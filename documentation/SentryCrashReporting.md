# Local Crash Dumps (Sentry Native)

## 1. For Users

### What is saved

When the RTX Remix Runtime crashes on the CPU, Crashpad (through the Sentry Native SDK) saves a crash report on this computer:

- **Minidump**: a `.dmp` file with the stack trace and thread state from the moment of the crash.
- **Log copies**: if present, `remix-dxvk.log`, `bridge32.log` and `bridge64.log` from the Remix logs folder, copied at crash time.
- **Extra context**: breadcrumbs (recent actions), tags (e.g. GPU, game executable) and other debugging data that the runtime adds in code.

Nothing is ever uploaded. Remix sets no Sentry DSN, so there is no server to send to.

A GPU crash (device loss) ends the process on purpose, so it produces no minidump. Its details are in `remix-dxvk.log`. With `dxvk.enableAftermath = True` in `dxvk.conf`, a GPU crash also writes an Aftermath dump (`.nv-gpudmp`), whose path is logged.

### The crash dialog

The dialog only informs. It names the folder that holds the crash report and the logs folder, and asks you to attach the newest `.dmp` file and the logs to your bug report.

- **x86 games** (x86 game → bridge → x64 Remix): the bridge client shows the dialog as soon as the x64 Remix process exits.
- **x64 games** (game loads dxvk-remix directly): a CPU crash is reported on the next launch. A GPU crash is reported right away.

### Where the files are

- Minidumps: `rtx-remix/logs/.sentry-native/reports/*.dmp`.
- Log copies for each report: `rtx-remix/logs/.sentry-native/attachments/<report id>/`.
- The dialog never deletes them. You can delete the `.sentry-native` folder while the game is not running, and Remix recreates it. Crashpad itself prunes reports older than a year or beyond its database size limit.

---

## 2. For Remix Developers (Adding Data to Crash Reports)

Use `#include "util_sentry.h"` and the `dxvk::sentry::` helpers to add data that is written into the next crash report.

| Function | Use when |
|----------|----------|
| **addBreadcrumb(category, message, level)** | Recording a step the app took (e.g. "Loaded texture X"). Stored in a small ring buffer and written into the next crash report. Levels: -2=trace, -1=debug, 0=info, 1=warning, 2=error, 3=fatal. |
| **setTag(key, value)** | Short, stable key/value pairs (e.g. GPU vendor, game executable). |
| **setContext(key, value)** | Longer debugging detail (e.g. device strings, config snippets). |

The data is stored in the report's `__sentry-event` attachment next to the minidump.

---

## 3. Integration Reference

- **`src/util/util_sentry.cpp` and `.h`**: all Sentry code. `initialize()` puts the database under the Remix logs folder and the handler next to `d3d9.dll`. It sets an empty DSN, so the crashpad handler has no upload URL (this also overrides a `SENTRY_DSN` environment variable). It installs a transport that discards every SDK envelope, and it revokes user consent, which also turns off uploads in the crashpad database settings. `showCrashReportDialogIfNeeded()` shows the next-launch dialog and `showGpuCrashDialog()` the GPU crash dialog. Both stay silent under the bridge and in automation mode.
- **`src/util/util_crash_report.h`**: shared with the bridge. Holds the GPU crash exit code, the early GPU crash notification message and `buildCrashReportMessage()`, so the runtime and the bridge word the dialog the same way.
- **`src/dxvk/dxvk_instance.cpp`**: `DxvkInstance::getSharedInstance()` calls `sentry::initialize()` and `showCrashReportDialogIfNeeded()`. The Aftermath callback writes the `.nv-gpudmp` file and logs its path.
- **`src/dxvk/dxvk_queue.cpp`**: on device loss, `handleDeviceLost()` notifies the bridge client, waits for the Aftermath dump, calls `showGpuCrashDialog()` and exits with `kRemixGpuCrashExitCode`.
- **`src/d3d9/d3d9_swapchain.cpp`**: calls `sentry::shutdown()` on `WM_DESTROY` and `WM_ENDSESSION`. The bridge server calls the `RtxSentryShutdown` export when it exits.
- **`bridge/src/client/d3d9_lss.cpp`**: when the server exits unexpectedly, `showCrashDialogAndExit()` shows the dialog and terminates the game.

**Build and deployment**

- Sentry is not included in debug builds. The prebuilt Sentry/Crashpad libraries are Release-only and fail to link in debug, so the build turns Sentry off when `buildtype` is `debug` (`external-build/meson.build`).
- `RTX_AUTOMATION_DISABLE_BLOCKING_DIALOG_BOXES=1` suppresses the crash dialogs (CI).
- `crashpad_handler.exe` and `crashpad_wer.dll` must be deployed next to the dxvk-remix `d3d9.dll`, because the handler path is relative to it.

### File layout

```text
<game or .trex>/
  d3d9.dll
  crashpad_handler.exe
  crashpad_wer.dll

<rtx-remix>/logs/
  remix-dxvk.log, bridge32.log, bridge64.log
  .sentry-native/            (crash database)
    reports/*.dmp            (minidumps)
    attachments/<id>/        (log copies and the __sentry-event data)
    settings.dat, user-consent
```

---

## References

- [Sentry Native SDK](https://docs.sentry.io/platforms/native/) (MIT)
- [Crashpad Backend](https://docs.sentry.io/platforms/native/configuration/backends/crashpad/) (Crashpad is Apache-2.0)
