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
#pragma once

// Shared crash-reporting definitions used by both the RTX Remix runtime and the 32-bit bridge, so
// the two stay in sync on how crashes are signaled and how the crash dialog is worded.

#include <cstdint>
#include <filesystem>
#include <string>

namespace dxvk {
  // Process exit code the runtime uses when it terminates itself after gracefully handling a GPU
  // crash (device loss) via env::killProcess(). The bridge client reads the server process's exit
  // code to distinguish a GPU crash (handled by the runtime) from a CPU crash (abrupt termination
  // caught by crashpad) or a normal exit, so it can show an appropriate dialog.
  static constexpr uint32_t kRemixGpuCrashExitCode = 0x60D0DEADu;

  // Named window message sent by the runtime to the bridge client as soon as a GPU crash is
  // detected, before it waits for the Aftermath dump. The bridge client registers a handler
  // via gpRemixMessageChannel so it can disable the Windows ghost dialog and prepare the crash
  // dialog window as early as possible, without waiting for the server process to exit.
  static constexpr const char* kGpuCrashNotifyMsgName = "UWM_REMIX_GPU_CRASH_NOTIFY";

  // Title used for all crash dialogs.
  static constexpr const wchar_t* kCrashReportDialogTitle = L"Remix Plus Crash Report";

  // Selects the opening line of the crash dialog.
  enum class CrashPromptKind {
    CpuImmediate,   // CPU crash, dialog shown at crash time (bridge server exit).
    CpuNextLaunch,  // Crash detected on the next launch (runtime, no bridge).
    GpuImmediate,   // GPU crash, dialog shown at crash time.
  };

  // Returns the folder that holds the crashpad minidumps (.dmp) under the Remix logs folder.
  inline std::filesystem::path crashReportsDir(const std::filesystem::path& logsDir) {
    return logsDir / ".sentry-native" / "reports";
  }

  // Returns the file the runtime leaves in the Remix logs folder when it ends the process after a GPU crash.
  // The problem notice of the next session reports it and deletes it.
  inline std::filesystem::path gpuCrashMarkerPath(const std::filesystem::path& logsDir) {
    return logsDir / "gpu-crash.marker";
  }

  // Builds the crash dialog message from shared parts so the runtime and bridge produce
  // identical wording. \p logsDir is the Remix logs folder.
  inline std::wstring buildCrashReportMessage(CrashPromptKind kind, const std::filesystem::path& logsDir) {
    std::wstring message;

    switch (kind) {
    case CrashPromptKind::CpuImmediate:
      message = L"The Remix Plus runtime has encountered an unexpected issue. The application will close.\n\n";
      break;
    case CrashPromptKind::CpuNextLaunch:
      message = L"The Remix Plus runtime crashed last time.\n\n";
      break;
    case CrashPromptKind::GpuImmediate:
      message = L"The Remix Plus runtime has encountered a GPU crash. The application will close.\n\n";
      break;
    }

    // A GPU crash ends the process on purpose, so crashpad writes no minidump for it.
    if (kind == CrashPromptKind::GpuImmediate) {
      message +=
        L"Please attach the log files from this folder to your bug report:\n" + logsDir.wstring() + L"\n\n"
        L"Adding 'dxvk.enableAftermath = True' to dxvk.conf makes future GPU crashes also write an Aftermath"
        L" crash dump (.nv-gpudmp), whose path is in remix-dxvk.log. Attach it as well."
        L" Aftermath may have a small performance overhead.";
    } else {
      message +=
        L"A crash report was saved to:\n" + crashReportsDir(logsDir).wstring() + L"\n\n"
        L"Please attach the newest .dmp file from that folder, and the log files from this folder,"
        L" to your bug report:\n" + logsDir.wstring();
    }

    message += L"\n\nCrash reports are never uploaded. They stay on this computer.";
    return message;
  }
}
