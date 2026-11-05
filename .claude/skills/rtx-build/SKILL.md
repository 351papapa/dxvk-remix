---
name: rtx-build
description: Build the dxvk-remix runtime (64-bit d3d9.dll) via the fork-side scripts/build.ps1 wrapper. Use when the user asks to build, compile, compile-check, verify, or smoke-test the runtime. Runs in the background, reports exit code + verifies artifacts. Do NOT invoke for doc/skill/config-only edits — those don't affect compilation.
---

# rtx-build

Wraps [`scripts/build.ps1`](../../../scripts/build.ps1), the fork-side
runtime build entry point. The script handles VS discovery via vswhere,
calls vcvarsall.bat by full path (avoiding the upstream SetupVS lying-
on-failure trap), pre-cleans the gitignored `nv-private/` and
`tests/rtx/dxvk_rt_testing/` dirs, runs `meson setup`/`compile`/`install`
in a single isolated cmd.exe shell, and verifies artifacts.

## When to invoke

- User asks to build, compile, compile-check, verify, or smoke-test
- Before pushing code changes
- After any code or shader change under `src/`

## When NOT to invoke

- Doc-only changes (`.md`, `CLAUDE.md`, `docs/`, `.claude/`
  entries) — no compile impact
- Skill / settings / config edits
- Bridge-only changes — invoke `bridge-build` instead
- Before the user has finished their code edits — building against a
  work-in-progress tree wastes time

## Steps

1. From the project root, kick off the build with
   `run_in_background: true` (expected duration: 30s incremental,
   10-15 min cold full):

   ```powershell
   & .\scripts\build.ps1
   ```

   For non-default flavor, add `-Flavor debug` or
   `-Flavor debugoptimized`. Add `-Clean` to wipe the build dir
   first. Add `-EnableTracy` to enable the Tracy profiler. `-Jobs N`
   changes the ninja job cap (default 4, see Known traps).

2. Wait for the task-completion notification — DO NOT poll, the
   runtime will wake you up when the background command exits.

3. Report back:
   - **Green (exit 0):** the script's own output ends with
     `=== Runtime build OK ===` and prints the artifact paths +
     mtimes. Quote those lines.
   - **Red (non-zero exit):** scan the captured output for
     `error C[0-9]`, `fatal error`, `FAILED:`, `ninja: build stopped`,
     or `Failed to run`. Quote the first ~10 error lines so the user
     sees the real cause before any fix attempt. If the failure
     pattern matches a known one (e.g. the
     `remix_api_multi_layer_audit` auto-memory entry), consult that
     before guessing.

## Flavors

| Flavor | `--buildtype` | Build dir |
|---|---|---|
| release (default) | `release` | `_Comp64Release` |
| debug | `debug` | `_Comp64Debug` |
| debugoptimized | `debugoptimized` | `_Comp64DebugOptimized` |

## Gotchas

- **`LNK4098: defaultlib 'LIBCMT' conflicts`** is a pre-existing
  upstream warning — not a failure.
- **`WARNING: msvc does not support C++11; attempting best effort`**
  is also upstream, not a failure.
- **Build produces `d3d9.dll`** in `_Comp64Release/src/d3d9/` and
  installs it to `_output/`, `public/bin/`, `nv-private/hdremix/bin/release/`,
  and the test app dirs.
- **Don't bypass the script** by calling `build_common.ps1::PerformBuild`
  (or `build_dxvk.ps1`, which calls it) directly. It works but its
  `SetupVS` function lies about success when vcvarsall.bat lookup
  fails — invisible until the next consumer hits a stale build. The
  script in `scripts/build.ps1` is the fork-side entry point that
  doesn't have that trap.

## Known traps

Each of these looks like a code error and is not.

- **`'meson' is not recognized`.** meson and ninja installed with
  `pip install --user` land in the user-site Python `Scripts` dir,
  which a spawned shell may not have on `PATH`. The script prepends
  the newest `%APPDATA%\Python\Python3*\Scripts` that holds
  `meson.exe` when meson is not already on `PATH`, and prints the
  meson it uses. If meson was upgraded since the build dir was
  configured, meson fails with "Coredata file ... references
  functions or classes that don't exist"; rebuild with `-Clean`.
- **The toolchain is fixed per build dir.** The script prefers VS 18
  (2026), Build Tools included, and falls back to VS 2019 / 2022.
  A build dir configured with one toolchain and built with another
  fails with `C1853 ... precompiled header file is from a different
  version of the compiler`, or with an unresolved
  `__std_regex_transform_primary_char` in a unit test that stops ninja
  before `d3d9.dll` relinks, leaving a stale DLL. So the script
  compares the dir's compiler (`meson-info\intro-compilers.json`,
  19.4x is VS 2022, 19.5x is VS 18) with the selected install's MSVC
  toolset and stops before building if they differ. Pass `-Clean` to
  reconfigure the dir with the selected toolchain. To keep building a
  dir with the other toolchain, use the manual chain below with that
  install's `vcvarsall.bat`.
- **The job count is capped at 4.** `cl.exe` adds its own `/MP`
  parallelism to ninja's, so on many-core machines an uncapped wide
  rebuild (touching `rtx_options.h` or `rtx_scene_manager.h`) fails
  with `C3859: Failed to create virtual memory for PCH` /
  `C1076: compiler limit: internal heap limit reached`, or `slangc.exe`
  exits `0xC0000005` (-1073741819) on a few shader permutations with
  no diagnostic. The script passes `-Jobs` (default 4) to
  `meson compile -j`. 4 is the only value proven to finish a full
  rebuild on a 32-core machine.
- **A shader change can need a second build.** After `.slang` /
  `.slangh` changes, the first build can exit 0 without relinking
  `d3d9.dll`. The script prints the artifacts' mtimes and warns when
  `d3d9.dll` was not relinked during the run. After a source or
  shader change, that warning means build again.
- **Deploy the whole `_output/` directory, never just `d3d9.dll`.**
  `d3d9.dll` imports dozens of DLLs that ship beside it (USD, DLSS,
  FidelityFX, CUDA), and their names change between branches. A
  partial copy fails `LoadLibrary` silently: an API host gets
  `REMIXAPI_ERROR_CODE_LOAD_LIBRARY_FAILURE` and no Remix window. The
  optional `scripts/auto-deploy.local.ps1` hook copies only `d3d9.dll`,
  so it is safe only when the support DLLs have not changed.
- **A fresh worktree has no submodules.** Run
  `git submodule update --init --recursive` before the first build,
  or meson fails on missing files under `submodules/`.

### Manual chain

This is the script's own chain with an explicit toolchain and job cap.
Use an absolute build dir path, because the shell's working directory
may not be the repo:

```powershell
$vcvars = '<VS install>\VC\Auxiliary\Build\vcvarsall.bat'
cmd.exe /c "call `"$vcvars`" x64 >nul 2>&1 && cd /d <repo>\_Comp64Release && meson compile -j 4 && meson install"
```

For a build dir that does not exist yet, run
`meson setup --buildtype release --backend ninja -Denable_tracy=false <repo>\_Comp64Release`
first, inside the same `cmd.exe` chain.
