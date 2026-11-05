# Validated local runtime build

`scripts/build-validated.ps1` rebuilds the already-configured x64 Release runtime
and updates its existing `_output/d3d9.dll`. Run it from the source checkout:

```powershell
.\scripts\build-validated.ps1 -CheckOnly
.\scripts\build-validated.ps1
```

This entry point requires Visual Studio 18, the Meson version recorded in the
build directory, Python 3.10 or newer, an existing `_Comp64Release` configuration,
and the complete existing `_output` distribution. It does not configure a new
checkout from scratch or download dependencies. `-BuildDir` selects another
configured build directory inside the source tree. The configured compiler must
match the selected VS toolset.

A junction to this checkout is supported. Python `samefile` verifies both the
requested source and build against `meson-info.json`; the build then uses the
configured source spelling. A different physical checkout is rejected rather
than accidentally compiling its older source. The build-directory junction is
retained, including when its target is on another volume.

## Meson and local settings

The default is the normal `meson` executable on PATH. An explicitly supplied
Python package directory must contain `mesonbuild/mesonmain.py`:

```powershell
.\scripts\build-validated.ps1 -MesonPythonPath 'C:\BuildTools\meson-1.10.2'
```

`REMIX_MESON_PYTHON_PATH` provides the same default. The optional, gitignored
`scripts/build.local.ps1` can hold machine-local settings:

```powershell
$MesonPythonPath = 'C:\BuildTools\meson-1.10.2'
$Python = 'C:\Python313\python.exe'
```

Explicit command-line settings override this file. It is local PowerShell code,
so only keep trusted settings in it. No Meson package is vendored by this entry
point. When a package directory is selected, a small loader is generated in the
build cache. It disables `platform._wmi` in that Python process and imports the
specified Meson package. Meson's own regeneration uses that loader as well.
System Python, WMI configuration, and global PATH remain unchanged.

## Build and validation order

1. Verify physical source/build identity, MSVC compatibility and Meson version.
2. Record hashes of all existing supporting DLLs and executables in `_output`.
3. Reconfigure the existing Meson build. Normalize only that physical build
   directory's Ninja path prefixes to its verified source-local junction alias,
   including slash and Ninja-escaped forms, then build only `rtx_shaders`.
4. Verify every G-buffer generated array against its SPIR-V bytes and generate
   declaration headers plus a read-only AMD64 COFF object. Apply its Ninja rule.
5. Build the `d3d9` target with Ninja `-j1` and MSVC's trailing `/MP1` override.
   The configured `/O2` option and shader payloads are retained.
6. Recheck current source/shader inputs, declaration hashes, the COFF sections,
   symbols and COMDAT records, and the installed Ninja rule.
7. Check that the resulting DLL is newer than its COFF input. Audit AMD64 PE
   normal and delay-load imports for every existing package DLL/executable,
   substituting the new runtime for the old one. Dependencies must resolve in
   the module directory, package root, Windows System32, or Windows API sets.
   Supporting distribution files must still match their pre-build hashes.
8. Copy the new DLL to a temporary file, verify its SHA-256, and atomically
   replace `_output/d3d9.dll`. Write `validated-runtime/build-report.json` in
   the build directory.

There is no `meson install`, artifact cleanup, auto-deployment, or game launch.
The supporting distribution is retained rather than reconstructed. Missing or
changed supporting files stop publication. Import auditing cannot prove
arbitrary `LoadLibrary` calls or GPU/game behavior; those still need runtime
testing. This target build does not assert that every auxiliary target or test
suite has passed.

## Generated G-buffer rule

`scripts-common/gbuffer_blob_build.py` owns the generated Ninja changes. Repeated
`apply` calls preserve unchanged outputs and avoid duplicate dependencies.
Existing generated rules are replaced with the current repository helper path;
content-addressed backups retain changed Ninja files. A Meson regeneration can
remove the rule, so the controlled entry point always applies it after shader
generation. Its final validation fails if another regeneration removes or
changes the rule during linking.

The workaround accepts the expected MSVC Release Ninja edge and refuses unknown
layouts or missing `/O2`. Standalone `generate`, `apply`, `verify`, and read-only
`audit` modes accept explicit `--runtime` and `--build-dir` paths. Generated
headers, objects, manifests, loaders, reports, and Ninja backups remain in the
build cache; they are not source files to commit.

`-CheckOnly` performs read-only identity/tool-path/compiler checks. It does not
generate the Meson loader, compile shaders, audit imports, patch Ninja, or copy
the output DLL.
