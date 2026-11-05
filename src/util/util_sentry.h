/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software and to permit persons to whom the
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

#pragma once

#include <string>
#include <vector>

namespace dxvk {

  // Local crash dumps through Sentry Native and Crashpad. Nothing is uploaded: a crash leaves a minidump
  // and copies of the logs in the .sentry-native folder under the Remix logs folder.
  // Sentry is only included in util_sentry.cpp so this header is safe to use from any target (e.g. libdxvk).
  namespace sentry {

    // Initializes Sentry. Call once at startup, after the Remix file system is initialized.
    void initialize(const char* version);

    // Tells the user where the crash report was saved if the previous run crashed.
    // Stays silent under the bridge, whose client shows the dialog when the server exits.
    void showCrashReportDialogIfNeeded();

    // Returns true when the previous run crashed, as showCrashReportDialogIfNeeded found it.
    bool crashedLastRun();

    // Tells the user about a GPU crash and which files to attach to a bug report.
    // Stays silent under the bridge, whose client shows the dialog when the server exits.
    void showGpuCrashDialog();

    // Shuts down Sentry. Call before process exit when possible.
    void shutdown();

    // Adds a breadcrumb, a step in the trail of what the app did recently.
    // Breadcrumbs are kept in a ring buffer of about 100 and are written into the next crash report.
    // Level: -2=trace, -1=debug, 0=info, 1=warning, 2=error, 3=fatal (Sentry sentry_level_e).
    void addBreadcrumb(const std::string& category, const std::string& message, int level = 0);

    // Sets a tag on the current scope. Tags are short, stable key/value pairs written into crash reports.
    void setTag(const std::string& key, const std::string& value);

    // Sets custom context on the current scope, for longer debugging text written into crash reports.
    void setContext(const std::string& key, const std::string& value);

    // Sets Vulkan instance layers and extensions as structured context, and sets overlay tags from the names.
    // Checks for the Steam overlay (e.g. VK_LAYER_*_steam_overlay) and other overlays (ReShade, RenderDoc, etc.).
    void setVulkanLayersAndExtensionsContext(const std::vector<std::string>& layerNames,
                                            const std::vector<std::string>& extensionNames);

  }

}
