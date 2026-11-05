---
name: deploy-and-test-game
description: Deploy built dxvk-remix to a game and launch it. Use when the user asks to deploy, test, run, or debug a game with Remix, or when built changes need to be tested in a running game.
---

# Deploy and Test a Game

All commands run in PowerShell from the repo root unless otherwise noted.

**Ask the user before launching a game or running a GPU replay.** The GPU may be in use.

## Game Targets

`gametargets.conf` (INI-style, not checked into git; start from `gametargets.example.conf`) defines the user's local game targets. Each section has:
- `outputdir` — the `.trex` folder where built DLLs are deployed
- `workingdir` — the game's working directory
- `commandline` — executable and arguments

To list available targets:

```powershell
python vsgen/get_game_target_list.py
```

This outputs `GameName,outputdir` per line.

## Build, Deploy, and Launch

### Step 1: Build

Build with the `rtx-build` skill (`.claude/skills/rtx-build/SKILL.md`). Its known traps apply here: the per-dir toolchain, the `-j 4` job cap, and the second build after a shader change.

### Step 2: Deploy to a Game Target

Deploy the whole build output, never just `d3d9.dll`. `d3d9.dll` imports dozens of DLLs that ship beside it (USD, DLSS, NRD, FidelityFX, CUDA), and a partial copy fails to load without a crash.

Either install every artifact for a `gametargets.conf` target:

```powershell
meson install -C <repo>\_Comp64Release --tags <GameTarget> --only-changed
```

The tag is the section name from `gametargets.conf`. Or copy all of `_output/` into the game's `.trex` folder.

Then check that the deployed `d3d9.dll` has the build's timestamp.

### Step 3: Launch the Game

Only after the user agrees. Parse the target's `workingdir` and `commandline` from `gametargets.conf`, then launch:

```powershell
$env:DXVK_CONFIG_FILE = "<repo_root>\dxvk.conf"
Start-Process -FilePath "<workingdir>\<executable>" -ArgumentList "<args>" -WorkingDirectory "<workingdir>"
```

## Debugging Crashes

The runtime automatically generates `.dmp` crash dump files. When a crash occurs, analyze the dump to get the call stack and crash context.

When the Remix window does not appear but nothing crashes, check the host's log for a load error (`REMIXAPI_ERROR_CODE_LOAD_LIBRARY_FAILURE`) before suspecting the API. It usually means a partial deploy.

## The Bridge (32-bit Games)

For 32-bit games, a two-process architecture is used:

1. The game loads a **32-bit `d3d9.dll`** (bridge client) which intercepts D3D9 calls
2. The client automatically launches **`NvRemixBridge.exe`** (64-bit server) from the `.trex` folder
3. The server loads the **64-bit `d3d9.dll`** (the actual Remix runtime) and renders into the game window
4. Communication between client and server uses shared-memory circular queues

Build the bridge with the `bridge-build` skill (`.claude/skills/bridge-build/SKILL.md`). When debugging bridge-based games, both processes may need inspection. The bridge disables IPC timeouts when a debugger is attached, so breakpoints won't cause disconnects. Bridge-specific configuration is in `bridge.conf`.
