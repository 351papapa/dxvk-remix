# Contributing to Remix Plus

Remix Plus is a hard fork of NVIDIA [`dxvk-remix`](https://github.com/NVIDIAGameWorks/dxvk-remix). The runtime has an API-neutral scene interface fed by the D3D9 frontend and by Remix API hosts.

**The rules are in [`AGENTS.md`](../AGENTS.md).** Read it before changing anything: working rules, upstream policy, git rules, the changelog rule, code style, RTX options, graphics presets, and documentation. This page covers setup and building only.

## Getting the code

Clone with submodules:

```
git clone --recursive <repo-url>
```

If you forgot `--recursive`, or you made a new `git worktree`, run `git submodule update --init --recursive` inside it. Worktrees do not get submodules on their own, and meson then fails on missing files under `submodules/`.

## Building

Remix Plus builds on Windows with MSVC, Meson and Ninja. A cold build takes 10-15 minutes; incremental builds take 1-3 minutes.

### Requirements

- Windows 10 or 11
- Visual Studio 2026 (VS 18) or its Build Tools, with the "Desktop development with C++" workload (x64 C++ tools). `scripts/build.ps1` prefers VS 18 and falls back to VS 2022 or 2019. A build dir keeps the toolchain it was configured with; after switching toolchains, build with `-Clean`.
- For the bridge: Visual Studio 2019 or 2022 with the C++ tools. `bridge/scripts/build.ps1` does not use VS 18 yet.
- [Windows SDK 10.0.19041.0](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/)
- [Meson 1.8.2+](https://mesonbuild.com/) and Ninja, for example `pip install --user meson ninja`. `scripts/build.ps1` finds a user-site install that is not on `PATH`.
- [Vulkan SDK 1.4.313.2+](https://vulkan.lunarg.com/sdk/home#windows)
- [Python 3.9+](https://www.python.org/downloads/) (not the Microsoft Store version)
- [DirectX Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=35)

Only x64 is supported. The first configure downloads the prebuilt dependencies in `packman-external.xml` into `external/`, so it needs internet access.

### First-time setup

PowerShell needs to allow local scripts. Run once in an elevated PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

### Build

From the project root, in PowerShell:

```powershell
.\scripts\build.ps1
```

This builds the release flavor into `_Comp64Release/` and installs the runtime and its support DLLs to `_output/`. Use `-Flavor debug` or `-Flavor debugoptimized` for the instrumented flavors, `-Clean` to wipe the build dir first, and `-EnableTracy` to enable the Tracy profiler.

Useful meson options, passed with `-MesonArgs` (for example `.\scripts\build.ps1 -MesonArgs '-Denable_d3d11=true','-Denable_dxgi=true'`):

| Option | Default | Effect |
|---|---|---|
| `enable_d3d11`, `enable_dxgi` | false | Build DXVK's D3D11 and DXGI modules. Remix API hosts that render with D3D11 (`RegisterD3D11Device`, `RenderD3D11`) need both. |
| `enable_d3d9_frontend` | true | False builds the API-only module: D3D9 draws are rasterized only and the Remix API is the only way into the path tracer. `-ApiOnly` builds this as a compile check. |
| `release_build` | true | No local paths in the binary, no PDB, EXP or LIB installed, and an extra SPIR-V optimization pass. Set it false for a development build with symbols. |
| `build_tag` | `dev` | A build name printed once at startup as `Build: <tag>` in `remix-dxvk.log`. Empty prints nothing. |
| `enable_tracy` | false | Tracy profiler support (`-EnableTracy` sets it). |

Changing an option on an existing build dir reconfigures it; `-Clean` starts over.

The build has known traps: the toolchain is fixed per build dir, many-core machines need the job count capped at 4, and a shader change can need a second build before `d3d9.dll` relinks. The [`rtx-build` skill](../.claude/skills/rtx-build/SKILL.md) describes each one and its fix.

The bridge builds separately with `.\bridge\scripts\build.ps1` (see the [`bridge-build` skill](../.claude/skills/bridge-build/SKILL.md)).

### Deploying

Copy the whole of `_output/`, not just `d3d9.dll`: into the game's `.trex` folder for a 32-bit game that runs through the bridge, or next to the game's executable for a 64-bit game. `d3d9.dll` imports dozens of DLLs that ship beside it, and their names and versions change between builds. A partial copy fails to load without a crash. The [`deploy-and-test-game` skill](../.agents/skills/deploy-and-test-game/SKILL.md) covers deploying through `gametargets.conf` (copy `gametargets.example.conf` and edit the paths).

### When dependencies change

If a dependency path changes (new Vulkan SDK, etc.), run `meson --reconfigure` inside the build folder. If meson itself was upgraded, rebuild with `-Clean`.

### Profiling

Tracy profiling is supported. Build with `-EnableTracy`, then connect [Tracy v0.8.7](https://github.com/wolfpld/tracy/releases/download/v0.8/Tracy-0.8.7z) to a running session.

## Commits

- Name branches as [`BRANCHING.md`](BRANCHING.md) says.
- Each commit is a self-contained step. The first line is a short summary in present tense. The body explains why; the diff shows what.
- Match the prefix style of recent commits (`git log --oneline`).
- Add a [`CHANGELOG.md`](CHANGELOG.md) entry in the same commit for every major change (see `AGENTS.md`, Changelog).
- Build clean before you merge: exit code 0 and zero compile errors.

## Bug reports

Bugs in Remix Plus go in this repo's issue tracker. Bugs in the broader RTX Remix toolkit belong in [NVIDIAGameWorks/rtx-remix](https://github.com/NVIDIAGameWorks/rtx-remix/issues).

## Credits

Remix Plus stands on:

- [DXVK](https://github.com/doitsujin/dxvk), the D3D9 to Vulkan translation layer this project descends from.
- [NVIDIA `dxvk-remix`](https://github.com/NVIDIAGameWorks/dxvk-remix), the path-traced remastering fork of DXVK (MIT, see `LICENSE-MIT`).
- The gmod-rtx community fork, where many of the Remix SDK API extensions originated.
