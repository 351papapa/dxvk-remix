---
name: run-unit-tests
description: Build and run dxvk-remix unit tests. Use when the user asks to run tests, add new tests, or verify that code changes pass existing tests.
---

# Run Unit Tests

All commands run in PowerShell from the repo root.

Unit tests live in `tests/rtx/unit/`. New test files must be added to `tests/rtx/unit/meson.build`.

**IMPORTANT:** Unit tests use a **separate dedicated build directory** (`_Comp64UnitTest`). Do NOT run unit tests from game build directories (`_Comp64DebugOptimized`, `_Comp64Release`, etc.) — they will fail.

The build traps in the `rtx-build` skill (`.claude/skills/rtx-build/SKILL.md`) apply here too: meson must be on `PATH`, the toolchain is fixed per build dir, and the job count is capped at 4. Do not use `build_dxvk.ps1`; its VS setup can report success after failing.

## First-Time Setup

Configure and build the unit test directory with the same toolchain as your other build dirs. Use absolute paths:

```powershell
$vcvars = '<VS install>\VC\Auxiliary\Build\vcvarsall.bat'
cmd.exe /c "call `"$vcvars`" x64 >nul 2>&1 && cd /d <repo> && meson setup --buildtype release --backend ninja -Denable_tracy=false _Comp64UnitTest && cd _Comp64UnitTest && meson compile -j 4 unit_tests && meson install --tags unit"
```

## Running Tests

Once `_Comp64UnitTest` exists:

```powershell
cmd.exe /c "call `"$vcvars`" x64 >nul 2>&1 && cd /d <repo>\_Comp64UnitTest && meson test --verbose"
```

Append a test name to run one test.

## Incremental Rebuild and Retest

If test source files have changed, rebuild before running:

```powershell
cmd.exe /c "call `"$vcvars`" x64 >nul 2>&1 && cd /d <repo>\_Comp64UnitTest && meson compile -j 4 unit_tests && meson install --tags unit && meson test --verbose <name>"
```

## Test Guidelines

- Minimal dependencies. Refactor tested code if needed to reduce coupling.
- Full branch coverage with carefully designed test data.
- Return 0 on success, -1 on failure.
- Include descriptive logging on failure.

See also `documentation/UnitTest.md`.
