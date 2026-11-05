/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include "util_sentry.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <windows.h>

#include "log/log.h"
#include "util_env.h"
#include "util_filesys.h"
#include "util_string.h"
#include "util_crash_report.h"

#ifndef REMIX_SENTRY_ENABLED
#define REMIX_SENTRY_ENABLED 1
#endif

// Sentry is disabled in debug builds, as the prebuilt Release Sentry/Crashpad will fail to link in debug builds.
#if REMIX_SENTRY_ENABLED

#define SENTRY_BUILD_STATIC 1
#include <sentry.h>

// When the build sets -Dremix_sentry_environment= (e.g. CI), that value is used. Otherwise it is "development".
#ifndef REMIX_SENTRY_ENVIRONMENT
#define REMIX_SENTRY_ENVIRONMENT "development"
#endif

#ifndef REMIX_SENTRY_FORCE_SMART_MINIDUMP
#define REMIX_SENTRY_FORCE_SMART_MINIDUMP 0
#endif

namespace dxvk {
namespace sentry {

  namespace {

    // Max time (ms) sentry_close() may keep the process alive when it is shutting down.
    static constexpr uint64_t kSentryShutdownTimeoutMs = 4000;

    // sentry_close() is not idempotent, and both WM_DESTROY and WM_ENDSESSION can request shutdown.
    static std::atomic<bool> s_shutdownDone{ false };
    static std::atomic<bool> s_sentryInitialized{ false };
    static bool s_crashedLastRun = false;

    static std::mutex s_sentryMutex;

    bool isSentryActive() {
      return s_sentryInitialized.load(std::memory_order_acquire)
        && !s_shutdownDone.load(std::memory_order_acquire);
    }

    std::string pathToUtf8(const std::filesystem::path& path) {
      return str::fromws(path.wstring().c_str());
    }

    bool isAutomationMode() {
      static const bool s_result = [] {
        const char* disableDialogs = std::getenv("RTX_AUTOMATION_DISABLE_BLOCKING_DIALOG_BOXES");
        return disableDialogs && std::string(disableDialogs) == "1";
      }();
      return s_result;
    }

    // Discards every envelope the SDK produces, so events, transactions and sessions never leave the process.
    // Crashpad minidumps do not pass through here.
    // They stay in the crash database because, with an empty DSN, the crashpad handler has no upload URL.
    void discardEnvelope(sentry_envelope_t* envelope, void*) {
      sentry_envelope_free(envelope);
    }

    void setContextList(const std::string& contextKey, const std::string& listKey, const std::vector<std::string>& values) {
      sentry_value_t list = sentry_value_new_list();
      for (const auto& s : values) {
        sentry_value_append(list, sentry_value_new_string(s.c_str()));
      }
      sentry_value_t context = sentry_value_new_object();
      sentry_value_set_by_key(context, listKey.c_str(), list);
      sentry_set_context(contextKey.c_str(), context);
    }

    std::string toLower(std::string s) {
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
      return s;
    }

  }  // namespace

  void initialize(const char* version) {
    if (s_sentryInitialized.load(std::memory_order_acquire)) {
      return;
    }

    const auto dllDir = std::filesystem::path(env::getDllDirectory());
    if (dllDir.empty()) {
      Logger::warn("Sentry: failed to get DLL directory for handler path");
      return;
    }

    const auto logsDir = util::RtxFileSys::path(util::RtxFileSys::Logs);
    const std::string sentryDbPathUtf8 = pathToUtf8(logsDir / ".sentry-native");
    const std::string handlerPathUtf8 = pathToUtf8(dllDir / "crashpad_handler.exe");
    sentry_options_t* options = sentry_options_new();
    sentry_options_set_database_path(options, sentryDbPathUtf8.c_str());
    sentry_options_set_handler_path(options, handlerPathUtf8.c_str());
    sentry_options_set_release(options, version);
    sentry_options_set_environment(options, REMIX_SENTRY_ENVIRONMENT);
    sentry_options_set_shutdown_timeout(options, kSentryShutdownTimeoutMs);

    // Crash reports stay on this computer. The DSN is empty, so the crashpad handler has no upload URL,
    // and the SDK's own envelopes go to a transport that discards them.
    // The empty DSN also overrides a SENTRY_DSN environment variable.
    // Revoked consent also turns off uploads in the crashpad database settings.
    sentry_options_set_dsn(options, "");
    sentry_options_set_transport(options, sentry_transport_new(discardEnvelope));
    sentry_options_set_require_user_consent(options, 1);
    sentry_options_set_enable_metrics(options, 0);
    sentry_options_set_enable_logs(options, 0);
    sentry_options_set_auto_session_tracking(options, 0);
    sentry_options_set_send_client_reports(options, 0);
    sentry_options_set_attach_screenshot(options, 0);

    // A development build has no symbols, so a dump beyond the call stack buys nothing there.
    // SMART adds the heap memory referenced from the stack and registers. There is no meson option for this.
    // Define REMIX_SENTRY_FORCE_SMART_MINIDUMP=1 in the compiler flags to force it.
    constexpr bool kIsDevelopmentEnvironment = std::string_view(REMIX_SENTRY_ENVIRONMENT) == "development";
    constexpr bool kForceSmartMinidump = REMIX_SENTRY_FORCE_SMART_MINIDUMP != 0;
    sentry_options_set_minidump_mode(options,
        (kIsDevelopmentEnvironment && !kForceSmartMinidump) ? SENTRY_MINIDUMP_MODE_STACK_ONLY
                                                            : SENTRY_MINIDUMP_MODE_SMART);

    // The crashpad handler copies these logs into the report's attachments folder at crash time,
    // so the copies hold the crashed session. Log text still buffered at crash time is missing from them.
    const auto attachLogFile = [&](const char* filename) {
      const auto logPath = logsDir / filename;
      if (std::filesystem::exists(logPath)) {
        const std::wstring logPathW = logPath.wstring();
        sentry_options_add_attachmentw(options, logPathW.c_str());
      }
    };
    attachLogFile("remix-dxvk.log");
    attachLogFile("bridge32.log");
    attachLogFile("bridge64.log");

    if (sentry_init(options) != 0) {
      Logger::warn("Sentry: failed to initialize local crash dumps");
      return;
    }
    s_sentryInitialized.store(true, std::memory_order_release);

    std::lock_guard<std::mutex> lock(s_sentryMutex);
    sentry_user_consent_revoke();
    Logger::info(str::format("Sentry: local crash dumps are saved to ", pathToUtf8(logsDir / ".sentry-native" / "reports")));

    const std::string appFolder = std::filesystem::path(env::getExePath()).parent_path().filename().string();
    sentry_set_tag("app_folder", appFolder.c_str());
  }

  void showCrashReportDialogIfNeeded() {
    if (!isSentryActive()) {
      return;
    }
    if (sentry_get_crashed_last_run() != 1) {
      return;
    }
    sentry_clear_crashed_last_run();
    s_crashedLastRun = true;

    // Under the bridge, the bridge client shows the crash dialog as soon as the server exits.
    if (env::isRemixBridgeActive() || isAutomationMode()) {
      return;
    }

    const std::wstring message = buildCrashReportMessage(CrashPromptKind::CpuNextLaunch, util::RtxFileSys::path(util::RtxFileSys::Logs));
    MessageBoxW(nullptr, message.c_str(), kCrashReportDialogTitle, MB_OK | MB_ICONINFORMATION);
  }

  bool crashedLastRun() {
    return s_crashedLastRun;
  }

  void showGpuCrashDialog() {
    // Under the bridge, the bridge client shows the crash dialog when the server exits.
    if (env::isRemixBridgeActive() || isAutomationMode()) {
      return;
    }
    const std::wstring message = buildCrashReportMessage(CrashPromptKind::GpuImmediate, util::RtxFileSys::path(util::RtxFileSys::Logs));
    MessageBoxW(nullptr, message.c_str(), kCrashReportDialogTitle, MB_OK | MB_ICONINFORMATION | MB_TOPMOST);
  }

  void shutdown() {
    if (s_shutdownDone.exchange(true)) {
      return;
    }
    if (s_sentryInitialized.exchange(false, std::memory_order_acq_rel)) {
      std::lock_guard<std::mutex> lock(s_sentryMutex);
      sentry_close();
    }
  }

  void addBreadcrumb(const std::string& category, const std::string& message, int level) {
    if (!isSentryActive()) {
      return;
    }
    sentry_value_t crumb = sentry_value_new_breadcrumb(nullptr, message.c_str());
    sentry_value_set_by_key(crumb, "category", sentry_value_new_string(category.c_str()));
    sentry_value_set_by_key(crumb, "level", sentry_value_new_int32(level));
    sentry_add_breadcrumb(crumb);
  }

  void setTag(const std::string& key, const std::string& value) {
    if (!isSentryActive()) {
      return;
    }
    sentry_set_tag(key.c_str(), value.c_str());
  }

  void setContext(const std::string& key, const std::string& value) {
    if (!isSentryActive()) {
      return;
    }
    sentry_value_t context = sentry_value_new_object();
    sentry_value_set_by_key(context, "value", sentry_value_new_string(value.c_str()));
    sentry_set_context(key.c_str(), context);
  }

  void setVulkanLayersAndExtensionsContext(const std::vector<std::string>& layerNames,
                                          const std::vector<std::string>& extensionNames) {
    if (!isSentryActive()) {
      return;
    }
    setContextList("vulkan_instance_layers", "layers", layerNames);
    setContextList("vulkan_instance_extensions", "extensions", extensionNames);

    bool hasSteam = false;
    bool hasAny = false;
    constexpr const char* steamOverlayKeyword = "steam_overlay";
    constexpr const char* anyOverlayKeywords[] = {
        "overlay", "reshade", "capture", "renderdoc"
    };

    for (const auto& name : layerNames) {
      const std::string nameLower = toLower(name);
      if (nameLower.find(steamOverlayKeyword) != std::string::npos) {
        hasSteam = true;
        break;
      }
      for (const char* kw : anyOverlayKeywords) {
        if (nameLower.find(kw) != std::string::npos) {
          hasAny = true;
          break;
        }
      }
    }
    if (!hasAny) {
      for (const auto& name : extensionNames) {
        const std::string nameLower = toLower(name);
        for (const char* kw : anyOverlayKeywords) {
          if (nameLower.find(kw) != std::string::npos) {
            hasAny = true;
            break;
          }
        }
        if (hasAny) {
          break;
        }
      }
    }

    setTag("has_steam_overlay", hasSteam ? "true" : "false");
    setTag("has_any_overlay", (hasSteam || hasAny) ? "true" : "false");
  }

} // namespace sentry
} // namespace dxvk

#else

// Sentry disabled (e.g. debug build): no-op implementations so callers link without Sentry libs.

namespace dxvk {
namespace sentry {

  void initialize(const char*) {}

  void showCrashReportDialogIfNeeded() {}

  bool crashedLastRun() { return false; }

  void showGpuCrashDialog() {}

  void shutdown() {}

  void addBreadcrumb(const std::string&, const std::string&, int) {}

  void setTag(const std::string&, const std::string&) {}

  void setContext(const std::string&, const std::string&) {}

  void setVulkanLayersAndExtensionsContext(const std::vector<std::string>&, const std::vector<std::string>&) {}

} // namespace sentry
} // namespace dxvk

#endif
