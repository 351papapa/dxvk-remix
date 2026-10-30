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
- Draw capture: native and x86 bridge contract tests passed. The adapted DFF
  plugin also passed a real game capture of models 1000/1001, with nine distinct
  material draw hashes matching the runtime. Temporary capture settings were restored.
- Wetness and window lighting: native/bridge CPU and GPU suites passed, including
  no-rain zones, DFF rules, weather transitions, topology, clock and disable cases.
  Both are integrated into the complete runtime and Core deployment. Final route
  cost comparisons remain pending.
- Procedural grass: GPU instance emission and scene lifetime integration are
  implemented and validated. Live route logs show active GPU candidate batches.
  VegetationIPL is a separate plugin: its F5 toggle now changes both near and LOD
  vegetation and passed fixed-scene visual checks.
- Water: GPU surface deformation, persistent simulation, shoreline waves, wakes,
  foam, caustics and underwater behavior are integrated. Native/bridge fixtures
  passed. Signed world-axis ring indexing and divergent foam geometry loads were
  corrected; numeric and GPU regressions passed. The exact location in the user's
  foam video still needs a final visual replay.
- Precipitation: migrated with surface contacts, weather presets and GPU lens
  rain. The additional no-rain-zone shelter passed 655,360 GPU oracle queries,
  sixteen native/bridge scenarios and a real game on/off test with 565 boxes.
  Zone-only tests used zero scene rays. Particle simulation remains on the GPU.
- Additional runtime work: flare materials, nearest-sphere sampling, point-instance
  frustum culling, volumetric workgroup variants, particle-lighting compatibility,
  OMM idle trimming, retained instances, bridge payload bounds and HUD atlas
  upload have implementation and targeted validation evidence. An inactive module
  in a route is not counted as a zero-cost module.
- Integrated plugins: RemixCoreVC, WorldMeshVC, VegetationIPL and DffCapture are
  built in their `_rpv2rc2` copies and deployed with the matching x86/x64 bridge.
  ColorFix and VehLights remain inside Core. Six game weather IDs select editable
  v2 presets; unsupported artistic parameters retain the new defaults.
- Stability fixes: option-change snapshots now share the resolver mutex. The
  actual production regression failed before the fix and passed afterward,
  including 50,000 concurrent updates. A ten-minute D3D9 replay and a subsequent
  API session completed without another heap crash. Vehicle disk-light basis
  normalization passed 100,029 tests; subsequent game diagnostics showed no
  CreateLight or invalid-handle errors with disk lights active. Individual
  headlight appearance still needs final visual acceptance.

## Game performance acceptance, 2026-10-30

The tested bundle is `integration-20261030-light-basis`: runtime SHA-256 begins
`79458308`, bridge server `68794c54`, bridge client `6e59b065`, and Core `7db5b0d5`.
The final matching API and original D3D9 submission paths each completed one
discarded warmup and three measured 300-second routes. Ten binary, route and
frontend configuration identities match. Both paths still use Remix rendering.

| Rendering submission path | Median base FPS | Median base frame time |
| --- | ---: | ---: |
| WorldMesh API | 32.443 | 30.823 ms |
| Original D3D9 | 32.510 | 30.760 ms |

The 0.207% average-FPS difference is small. D3D9 had five frames above 100 ms
across the three routes, including two at route startup; API had one. All remain
in the statistics. No new crash or light-creation failure occurred in these
accepted routes. Sparse long-frame causes remain unconfirmed. Historical runs
from other bundles are not final matched evidence.

Individual module tests use the original D3D9 path and restore the same baseline
between cases. Current results are:

| Disabled module | Formal runs | Reference base FPS | Disabled base FPS | Disabled base ms | Raw ms difference | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| wetness | 3 | 32.510 | 40.339 | 24.790 | 5.970 | baseline-drift-limited |
| water | 1 | 32.510 | 34.274 | 29.177 | 1.583 | baseline-drift-limited |
| precipitation | 1 | 35.798 | 36.363 | 27.500 | 0.434 | provisional |
| procedural-grass | 1 | 32.510 | 32.938 | 30.360 | 0.400 | baseline-drift-limited |
| vegetation-ipl | 1 | 35.798 | 36.042 | 27.746 | 0.189 | provisional |
| nearest-sphere-sampling | 1 | 35.798 | 35.564 | 28.119 | -0.184 | provisional |
| precipitation-cullzone-shelter | 1 | 35.798 | 35.337 | 28.299 | -0.364 | provisional |
| lens-rain | 1 | 35.798 | 34.989 | 28.580 | -0.646 | provisional |
| omm-idle-trimming | 1 | 35.798 | 40.620 | 24.618 | 3.316 | coverage-unconfirmed |
| window-lighting | 1 | 35.798 | 36.353 | 27.508 | 0.427 | coverage-unconfirmed |
| point-instance-frustum-culling | 1 | 35.798 | 35.375 | 28.268 | -0.334 | coverage-unconfirmed |

Remaining cases use one measured run, as requested by the user. Existing results
are retained without further repetitions. From the OMM case onward, each uses
a 60-second warmup followed by the same full 300-second measured route, in the
requested order: OMM, VegetationIPL, PointInstancer culling, nearest spheres,
then lens rain. The user subsequently excluded flare material as well as volume
workgroup layout, and requested no further warmups. Lens rain uses a recorded
warmup-skip policy; earlier warmup records are retained as they actually ran.
The retained reference used a full-route warmup, so cache history may differ.
Baseline-drift-limited rows are raw
differences confounded by the unchanged baseline becoming faster later, and are
not isolated module costs. The later identical-configuration reference measured
35.798 FPS / 27.935 ms, a 9.18% reduction in frame time; its cause is unconfirmed.
Coverage-unconfirmed rows establish the configuration comparison but do not
prove active custom-module surfaces in the route. These are changes in the complete rendered
frame, including the effect on lighting and visibility, rather than isolated
shader costs. GPU and CPU interval timers overlap and cannot be added. Traffic,
pedestrians and the daily window seed are not deterministically restored.
Frame-generation display estimates are recorded separately.

The nearest-sphere off attempt at 20:24 stalled at about 213 seconds with a
`vector too long` worker exception. Its dumps and logs remain preserved and it
is excluded from performance results. One debugger replay completed without
recurrence and is also excluded. The user requested completing the remaining
tests before further diagnosis. A successful later run does not establish that
the intermittent failure has been fixed.

Remaining work: complete visual checks of the water-video scene and individual
headlights, resume the deferred freeze investigation, and restore temporary
timing controls before preparing the final matching deliverable.

The evidence workspace is
`G:/Codex/2026-10-29/ban/work/rpv2-source-migration`, with live timing records under
`game-performance`. `latest-migration-status.json` in that directory records the
current phase. Tests retain raw data, exclude warmup and interrupted routes, and
report base-frame FPS separately from generated/display FPS. No GPU computation
or instance generation has been replaced by a CPU implementation.
