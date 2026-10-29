# Local runtime feature migration, 2026-10-29

Target: Remix Plus v2, commit `268fb5b2`, in a new recursive clone. Source:
the local runtime at `c17b36af`, including its uncommitted working-tree changes.
The original runtime remains the reference. This is an altered local fork;
upstream implementation and copyright notices remain in place.

The migration follows `RemixApi-Migration.md`, `api-host-path-migration.md` and
`documentation/RemixApiChangelog.md`, using the source of the pinned target
commit to resolve ambiguities in the prose.

## Order and acceptance

1. Establish an unmodified runtime build and matching x86/x64 bridge build.
2. Restore draw correlation capture, preserving queued draw identity and the
   DffCapture CSV contract. Exercise duplicate draws, session rollover and I/O
   retries. Keep backend code independent of D3D9 frontend types.
3. Restore per-draw wetness classification, then material shading, no-rain zones,
   DFF overrides, rain transitions and the current weather preset mechanism.
4. Restore window and street-lamp classification, UV islands, clock/day inputs
   and shader emission, with disabled defaults and menu controls.
5. Restore procedural grass placement, USD prototypes, CPU/GPU generation and
   scene lifetime integration, including original texture name lookup.
6. Restore water progressively: controls and surface geometry, simulation and
   waves, shore data, foam, caustics and underwater rendering.
7. Audit remaining fork features: emissive texture overrides, precipitation
   controls, lens rain, particle lighting and any still-needed capture fixes.
8. Rebuild plugin copies and run controlled game tests with a complete matching
   runtime/bridge deployment and a restorable backup.

Each feature is developed on a local feature branch and validated before moving
on. No public push is part of this work. A successful compile is not treated as
evidence of correct in-game output.

## Plugin contract

RemixCoreVC contains ColorFix, VehLights, RTXLights and WaterLab. Those historical
projects are not separate migration targets. Independent plugins are inventoried
from their actual source and game deployment.

All API consumers must use the target's 1.0 header consistently. The x86 bridge
has a smaller feature surface than the x64 runtime: retained instances, asset
key extensions and geometry refresh cannot be assumed available merely because
their x64 declarations exist. Each missing bridge feature must be implemented
and tested before a plugin depends on it.

Weather input uses `rtx.weather.gameWeather` and the new file-backed presets.
The removed `__weather.*` and old weather snapshot layout must not be restored
over the new weather implementation. Dynamic lights require source verification
because the migration and host-path documents disagree about the old dynamic
flag workaround.

## Checkpoints

- Recursive clone and pinned submodule verification: complete.
- Original uncommitted source and added-file checkpoint: saved outside the old
  repository, including the dirty RTXDI diff. RTXDI modifications are not copied
  into the new dependency submodule.
- Removed only three verified generated `_Comp64IndexMemoRemoved*` directories;
  retained their Meson metadata/logs and the active Release build.
- Runtime baseline: full x64 release build passed with D3D11/DXGI enabled. The default D3D9-only build has unresolved exports in the upstream definition file.
- Draw capture: integrated runtime build, x86/x64 bridge builds and isolated log tests passed. Native x64 and bridged x86 D3D9 GPU tests each produced exactly four records (two sessions, two tokens) with identical hashes. DffCapture plugin copy builds. Full Vice City traversal remains untested.
- Other modules: pending implementation and validation.
