## Remix API Changelog

## 2026-11-03 — Camera medium for external effects

The external frame buffer exposes cameraMediumFlags at byte 1356 (formerly
reserved), with bit 0 indicating water. Frame size and existing non-padding
offsets are unchanged. No exported function or bridge command is added.


## 2026-11-03 — Live post-processing tint

### Added

`__postfx.tint_v1` publishes validity and normalized RGB in one GameStateStore
value. External frame data appends a typed tint and validity flag at offsets
1360 and 1372; its size is 1376 bytes. Existing offsets, exported functions and
bridge commands are unchanged. See [post-processing tint](RemixPostFxAPI.md).

## 2026-11-02 — Private retained catalog v3 and static world batches

The private fixed-width interface negotiates v2 or v3 with an unchanged layout.
Version 3 adds HIDDEN unordered records and preserves their intended visibility
through missing commits and asset rebuilds. Transform/visibility-only changes
update existing retained children. Matching bridge endpoints negotiate support
through a runtime acknowledgement; v2 wrappers continue to reject v3 flags.

Eligible static opaque surfaces can be physically packed across records sharing
a 64-unit cell, original texture identity and compatible resolved
material/surface/cull/wetness state. Pure
opaque enhanced replacements expand through the existing resolver. Canonical
identities remain available for invalidation, per-record visibility and tools;
unsupported inputs or failed builds restore canonical rendering. Unchanged
membership/state reuses packed vertices, while a changed nonempty visibility
index set still requires a full BLAS build. All-hidden batches keep the last
nonempty index buffer/state and geometry revision while remaining hidden. Exact
mask reveal reuses that topology; release policy 0 still rebuilds the scene view
in the reveal frame using the same buffer. Different nonempty masks build new
indices. Retained visibility policy may still request BLAS work.
World-batch accounting is bounded at 128 MiB. This internal view adds no public
function or layout change.

The planner preflights complete connected input/bucket components before packing or GPU
allocation. A capacity-rejected component stays canonical without discarding
independent accepted batches. Final visibility is published once after a
successful transaction, preserving unchanged views; actual build/allocation
failures retain the full canonical fallback. Sparse CPU phase and reason counters
distinguish budget rejection from errors.

Fixed packed chunks retain unchanged source geometry across admissions. Full-child
proof retires every changed old part; only complete uncovered children enter new
delta packing. Stable physical slots and budgeted holes prevent regrouping churn
without reviving old geometry when a canonical handle is reused. Publication
remains staged, with canonical fallback on failure.

Synchronous preparation borrows the entry-owned resolved parts. Immutable cached
sources, physical ranges and members are shared. Cached and fast rendering plans
carry no ownership list; full output constructs active ownership from storage.
The full topology index vector was already shared; this change removes duplicate
metadata copies without changing whole-child ownership or GPU topology rules.
The CPU complete-child proof uses a compressed incidence index and rejection
queue instead of rebuilding coverage and rescanning all buckets to convergence.
Its fixed-point eligibility and whole-child ownership contract remain unchanged;
grouping, capacity, canonical fallback and GPU topology rules are preserved.
The R9 algorithm adds no phase fields and reuses R8 telemetry. Separate R10
diagnostics report six helper full phases including cleanup, then successful
compatibility publication. Work/Publish pair by runtime frame+prepareAttempt;
helper completion is distinct from compatibility publication and GPU completion.
Runtime frame and HUD source-frame IDs are independent; approximate wall-time
association does not establish a slow-frame cause. Helper log formatting is
inside outer prepareMs but outside helper totalMs, so diagnostic overhead is an
expected residual. Normal Perf also exposes the existing sampled BlasBuild,
TlasBuild and SceneSetup means with stageSamples; missing values stay n/a. No
new queries, ABI, settings, rendering or admission changes are introduced, and
global GPU stage times do not isolate world batching.

Separate cumulative full/fast successful-transaction phase counts, totals and
lifetime maxima supplement independently sampled sparse events. Failed work
remains in failure/resource counters; lifetime maxima cannot be differenced into
formal maxima, and absent older-run fields remain unavailable.

Full membership preparation is separate from visibility-only updates. The full
generation owns a canonical-child/member index; only affected batch masks and
active indices are staged together before upload, keeping immutable vertices.
Structural changes forbid the fast path. Missing commits require republishing
all cached presentations on resume, even for unchanged masks; hidden batches
never draw the last nonempty topology. Staging or prefix-publication failure
restores canonical sources. `[WorldBatchFast]` records this work separately.

Local old-child detachment and final visibility share one staged index
transaction. Successful publication releases old ownership and exposes canonical
children from the latest successfully published source, preserving stable batches.
Detached old ranges cannot revive through reused handles or show/republish.
Optional full readmission is pending-only at a one-second steady-clock interval;
global asset/texture/options/rule/tool invalidation still requires current
canonical state before upload. Failure and missing-commit behavior remain safe.

Index-build attempts/builds are logical transitions, including hidden no-allocation
changes. Separate buffer-resource/byte/triangle and successful reveal/hidden reuse
counters distinguish actual topology work; they do not measure device-memory
allocation calls or establish a GPU performance improvement.

Hidden instances skip wetness/material/name classification queries after clearing
their window class; reveal classification runs before the same frame's upload.
The window-disable pass also clears hidden surfaces. `[SurfaceClassify]` records
resetting interval CPU totals and timings separately from cumulative batch work.

A shared gate accepts disabled emission or exactly zero authored intensity without
changing material fields. Enabled nonzero/nonfinite intensities remain canonical;
the original texture hash is also retained in world grouping and cache equality.

The r2 runtime and matching bridge builds succeeded. All four targeted CPU tests
passed, including the r2 emission-policy cases and normal-transform rotation/shear
coverage. R2's valid formal route regressed to 25.41 average base FPS / 11.68
1% low against baseline 34.46 / 16.69, so it was rolled back. R3 CPU tests and
full build passed, but its valid formal route measured 33.86455 average base FPS /
14.75868 1% low versus baseline 34.46336 / 16.68940. R3 did not meet the performance
goal. R4 CPU tests and full build passed, but its valid formal route measured
34.1766608 average base FPS / 14.0138520 1% low against baseline 34.4633559 /
16.6894049; both gates failed. The observed call interval remained about 72% full
preparation and 28% fast. R5 CPU tests and build passed, but its valid formal route
measured 33.824186 average base FPS / 14.885688 1% low against 34.463356 /
16.689405. Its observed interval was about 3% full / 97% fast; complete formal
Perf windows averaged 15.20 ms rendering CPU and 28.74 ms GPU. Both FPS gates
remained unmet. R6 CPU tests and build passed, but its valid formal result was
34.2475767 average base FPS / 15.848298 1% low against 34.4633559 / 16.6894049.
Complete formal Perf windows averaged 14.42 ms CPU / 28.18 ms GPU. Its observed
work was 8049 real IB creations, 2712 reveal reuses and 6863 hidden no-allocation
transitions. Hidden transitions already did not allocate in R5; they are not an
allocation saving attributed to R6. Both performance goals remained unmet.
R7 CPU tests and full build passed. Its valid formal result was 34.8744584 average
base FPS / 15.8678106 1% low against original 34.4633559 / 16.6894049. Average met
the existing goal, while 1% low did not. Five complete Perf windows averaged
13.64 ms CPU / 27.44 ms GPU. Its observed work was 2125 real VB creations and
6987 real IB creations, with 2778 reveal reuses and 6500 hidden no-allocation
transitions. All 6286 raw frame samples, including the initial spike and 54
samples above 50 ms, remain included. R8 targeted CPU tests and full build passed. Its valid formal result was
34.66398557 average base FPS / 16.66741185 1% low versus original
34.46335595 / 16.68940488. The strict combined comparison failed on the measured
0.13178% low1 difference; the prior D485/A9BF working version was restored.
One small single-run difference does not prove a persistent causal regression.
Interior full transactions averaged 8.54503 ms total, with 6.83573 ms prepare;
this does not measure completeInputs separately. Five complete Perf windows
averaged 13.32 ms CPU / 27.82 ms GPU. All 6250 raw samples remain included.
R9 targeted CPU tests and full build passed. Its valid 180.025-second formal run
measured 34.91051504 average base FPS / 16.20685087 1% low versus original
34.46335595 / 16.68940488. Strict comparison failed on low1; D485/A9BF was
restored. All 6288 raw samples remained. Full mean was 7.69472 ms, including
5.95176 ms prepare versus R8's 8.54503 / 6.83573 ms; this does not isolate
completeInputs or establish a permanent causal FPS change. Five complete Perf
windows averaged 13.50 ms CPU / 27.64 ms GPU. R10 adds diagnostics only.
R10 targeted CPU tests and the full build passed. Its one valid 180.017-second diagnostic route retained all 6248 raw frames and measured 34.67515914 average base FPS / 16.25225565 1% low, against original 34.46335595 / 16.68940488. The strict combined comparison failed on low1. D485/A9BF, both original bridge halves and the original configuration/route were restored, as required before the diagnostic regardless of its metrics. The formal interval contained 178 exact Work/Publish pairs without duplicates. Helper full prepare averaged 6.05647 ms (P95 12.1179 ms, maximum 22.7939 ms); its maximum event included 17.317 ms in newPack. That stage includes source decoding, packing, buffer creation/mapping/copy, bounds/hash and initial indices, not just packing arithmetic. Five complete normal Perf windows contained 176 GPU samples: sample-weighted global BLAS 1.29553 ms, TLAS 0.49404 ms and SceneSetup 0.85229 ms. These are global sampled stages and do not identify world-batch or individual slow-frame causes.

R11 removes per-source decoded vertex/index arrays from new world-batch construction. A synchronous reader feeds the existing transform and packing rules directly from validated source buffers into the final CPU arrays. Geometry ordering, authored attribute defaults, transform/normal math, index bounds, output limits and failure atomicity remain checked. Source resources remain strongly retained for the synchronous read. This removes duplicate intermediate allocation/copy work; it does not introduce a pending build queue or change admission, visibility, material, topology or quality rules. R10 diagnostics stay enabled to compare the same measured phases. R11 CPU validation: four targeted CPU tests passed; full build: pending; live/GPU: pending. No FPS improvement or visual equivalence is established before the single formal comparison.
No R10 FPS benefit or visual equivalence is established. No
rendering options or presets change. See the
[compatibility contracts](../docs/RemixRetainedCompatibility.md).

## 2026-10-31 — Bridge light definition updates

Added a capability-gated command that keeps bridge light handles stable and
applies owned definitions in API command order before frame injection. The
ordered status query reports asynchronous runtime rejection. No options or
presets change. Deploy the matching bridge client/server pair together.
Compiled source: `c9e3cac7`; this validation follow-up changes documentation only.

One discarded 60-second warmup and one 300-second route per configuration gave
33.698 -> 34.028 base FPS (29.676 -> 29.388 ms), 1% low 18.370 -> 18.309 FPS,
P99 47.494 -> 47.791 ms, and 74 -> 68 frames over 50 ms.
This is a single-run observation, not an established percentage gain.
The comparison includes the matching Core host adaptation and keeps the
accepted WorldMesh change and rendering settings fixed.
See the [contract and validation limits](RemixBridgeLightDefinitions.md).

## 2026-10-30

### Fixed

The x86 bridge now serializes `remixapi_MeshInfoSkinning` arrays using their total element counts. This prevents out-of-bounds reads for valid skinned meshes. Public structs and API versions are unchanged; deploy the matching bridge client and server together.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- 2026-10-30 (local fork): the x86 bridge exports the private
  `remixapi_GetRetainedInterface` entry by its documented undecorated name,
  allowing retained-catalog hosts to negotiate the implemented interface.

### Added
- 2026-10-29 (local fork): typed per-weather global volumetric fog fields and
  structural wetness transitions, controlled by `rtx.weather.controlVolumetricFog`.
  Host weather inputs remain the existing NoSave current/outgoing/blend options.
  See [weather volumetrics](WeatherVolumetrics.md).
- 2026-10-29 (local fork): ordered `AutoInstancePersistentLights` forwarding in
  the matching bridge, advertised by the client-local
  `vc.bridge.autoInstancePersistentLightsVersion` capability (value `1`).
  Public function signatures and interface version are unchanged. See
  [bridge capability negotiation](RemixBridgeCapabilities.md).
- 2026-10-29 (local fork): restored the negotiated fixed-width retained v2 and
  legacy-surface V1/V2 host protocols on the native runtime and matching bridge;
  added real game-texture hash/material binding and GPU-instancing batch forwarding.
  The public table/version are unchanged. See [compatibility contracts](../docs/RemixRetainedCompatibility.md).
- 2026-10-29 (local fork): matching x86/x64 bridge forwarding for
  `SetCameraMediumMaterial`, and the private pointer-free GPU water protocol.
  NUMOS presets also accept ocean rain/storm weights. See `RemixWater.md`.
  The public C API layout and version are unchanged.
- 2026-10-29 (local fork): `__window.hour`, `__window.day` and `__window.badweather`
  conventions for window and street-lamp emission. See `RemixWindowLighting.md`.
  The API table and version are unchanged.
- 2026-10-29 (local fork): `__wetness.norain_zones_v1` and
  `__wetness.rain_ending` game-value conventions, plus wetness fields in NUMOS
  weather presets. See `documentation/RemixWetnessAPI.md`. The C API table and
  version remain unchanged.

### Removed
- 2026-10-21: the named export `remixapi_fork_ReplayIdentityTrace` and its
  result struct `remixapi_fork_IdentityReplayResult`, with the D3D9 identity
  trace it replayed (`rtx.d3d9Identity.traceFrames`). It was a diagnostics
  entry point for `remix_replay --identity-trace`, not part of the plugin
  contract; `remix_c.h` and `remixapi_Interface` are unchanged.

### Changed
- 2026-10-30: `CreateMaterial` with a live hash no longer ignores the new data. An
  identical description returns at once with no rebuilds, and different data
  replaces the material and rebuilds its users. `DestroyMaterial` then
  `CreateMaterial` still works. The API's signatures are unchanged.
- 2026-10-27: `REMIXAPI_VERSION` is `1.0.0` (was `0.1000.1`), for the release
  after `public-rpv2-rc1`. Major 1 marks the Remix Plus line as distinct from
  NVIDIA dxvk-remix (0.x) and other forks. Under the version check, the runtime
  and the bridge client accept 1.x callers whose minor is at most theirs, and
  also 0.1000.x callers (the header of `rebased-9-19` and rc1), because the
  interface table and all structs are unchanged. Every other 0.x caller, such as
  NVIDIA's 0.6.x or another fork's 0.1003.x, is rejected. Existing 0.1000.x
  binaries keep working unrebuilt, and the version-gated struct reads still
  apply to them. Rebuild against this header to use the new exports. See
  `docs/RemixApi-Migration.md`.
- 2026-10-25: `remixapi_GetMemoryPressure` reports the runtime's own pressure
  level, which now follows `usageBytes` (driver usage minus the free space inside
  the runtime's memory chunks) instead of the driver's usage. The struct is
  unchanged.
- 2026-10-25: the entry points check their input (docs/frontend-conformance.md)
  and count what they reject or fix in the runtime log's `[SceneInput]` line.
  `DrawInstance`, `CreateRetainedInstance(s)` and `UpdateRetainedInstance`
  return `REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS` for a non-finite transform or
  GPU-instance transform, or a bone or instance-transform array of null with a
  count. `CreateMesh` and `CreateMeshBatched` drop a surface without vertices
  or with too few blend indices or weights, and return
  `REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS` when no surface is left; non-finite
  positions become 0 with their triangles collapsed, a triangle with an index
  past `vertices_count` collapses, and a blend index of 256 or more becomes 0.
  An instance with fewer bones than its mesh's blend indices name gets identity
  for the rest, and bones past 256 are dropped. `UpdateRetainedInstance` of a
  destroyed handle no longer creates it again, and `DestroyMesh` drops a
  pending `CreateMeshBatched` of the same hash. The surface is unchanged.
- 2026-10-25: `remixapi_StartupInfo::combineGuiInFinalColor` takes effect. With
  `rtx.gui.enableExternalPresenter` set, false keeps the Remix GUI out of the
  final color, so the host composites it from `dxvk_CopyRenderingOutput`'s GUI
  output. It had no effect before, and the GUI was always in the final color.
- 2026-10-21: `remixapi_AssetKeyEXT` is always honored. The runtime option
  `rtx.scene.assetKeys` that gated it is removed, and so are the scene
  options that made retained instances use the scene's lanes and lock-free
  command queue: those paths are now always on. The API surface is unchanged.
- 2026-10-20: `remixapi_GetMemoryPressure` reports a `level` that never exceeds
  the runtime's own pressure level, which now follows the driver's usage. The
  reported `level`, `usageBytes`, `bytesToRelease` and `requestHostRelease` follow
  usage minus the free space inside the runtime's memory chunks.
  `requestHostRelease` is set from level 2 (High) by default.

### Added
- 2026-10-27: color grading, as named exports: `remixapi_SetColorGradePreset`,
  `remixapi_SetColorGrade`, `remixapi_MixColorGradePresets`, `remixapi_GetColorGrade` and
  `remixapi_SaveColorGradePreset`, with `remixapi_ColorGradeEXT`
  (`REMIXAPI_STRUCT_TYPE_COLOR_GRADE_EXT` = 0x4E560004, `REMIXAPI_COLOR_GRADE_VERSION` 1).
  Besides the grade (matrix, balance, saturation, contrast), the struct carries the tone curve's
  `anchor` (default 0.18) and `shoulder` (default 0.01). A 0 in either takes the default.
  The bridge exports and forwards them. `remixapi_Interface` is unchanged. docs/RemixApi.md,
  "Color grading".
- 2026-10-27: `REMIXAPI_HOST_SETTING_FLAG_USER_MENU` (0x4) in `remix_c.h`. A host
  setting with this flag also appears in the user menu (Alt+X) under Game Settings,
  with the same control as in the developer menu's Host Settings tab. The struct
  and the other flag values are unchanged.

- 2026-10-26: `remixapi_ConfigureExternalFrame` / `PFN_remixapi_ConfigureExternalFrame`
  and `remixapi_ExternalFrameEXT` (`REMIXAPI_STRUCT_TYPE_EXTERNAL_FRAME_EXT` =
  `0x4E560003`, `REMIXAPI_EXTERNAL_FRAME_VERSION` = 1): the presented frame is
  copied every `Present` into an exported R8G8B8A8_UNORM image with a ready and
  a released binary semaphore, all as OPAQUE_WIN32 NT handles, so another Vulkan
  device or process shows it without a CPU readback (docs/RemixApi.md,
  "External frame"). A named export, so `remixapi_Interface` and
  `REMIXAPI_VERSION` are unchanged.
- 2026-10-26: `REMIXAPI_ERROR_CODE_FEATURE_UNAVAILABLE` (`0x4E560001`), the first
  fork error code, in the fork range that the fork struct types use.
- `remixapi_DrawScreenOverlayD3D11` / `PFN_remixapi_DrawScreenOverlayD3D11`
  and `remixapi_ScreenOverlayLayerD3D11`: a native D3D11 host passes its UI
  render targets (up to 8 layers, with a premultiplied-alpha flag and a
  destination rectangle). Remix copies them on the GPU into persistent images.
  This is a named export like the other D3D11 host extensions, so
  `remixapi_Interface` and `REMIXAPI_VERSION` are unchanged.
- Runtime option `rtx.enableGpuScreenOverlay` (Developer Options, "GPU Screen
  Overlay (D3D11 Hosts)"), default on.
- 2026-10-03: `remixapi_TextureInfoHostImportEXT`
  (`REMIXAPI_STRUCT_TYPE_TEXTURE_INFO_HOST_IMPORT_EXT` = 33), chained on
  `remixapi_TextureInfo`. `CreateTexture` then carries what a D3D11 import
  carries: `encoding` (`REMIXAPI_TEXTURE_ENCODING_*`) and an optional
  `vkFormat` that overrides `format`. The API recorder's replayer uses it for
  recorded imports. Commit `2f2a32165`.
- 2026-10-03: `REMIXAPI_TEXTURE_HOST_IMPORT_FLAG_REPLACE` in
  `remixapi_TextureInfoHostImportEXT.flags`. It swaps the image of the live
  texture registered under `hash` in place and keeps its texture-table slot,
  as `ReplaceD3D11TextureEx` does. Without a live texture under that hash, the
  texture is created normally. Commit `1066c5580`.
- 2026-10-03: `remixapi_CameraInfoFrameEventsEXT`
  (`REMIXAPI_STRUCT_TYPE_CAMERA_INFO_FRAME_EVENTS_EXT` = `0x4E560001`) and
  the flags `REMIXAPI_CAMERA_FRAME_EVENT_CAMERA_CUT`, `_HISTORY_INVALID` and
  `_ORIGIN_SHIFT`, chained on the main camera's `remixapi_CameraInfo`.
  A cut or history invalidation restarts temporal history without clearing
  the scene. An origin shift moves the camera history by `originShift`, so the
  host's origin move is neither camera motion nor a cut. Fork sTypes start at
  `0x4E560000`, a range upstream never uses. Commit `e3e70114e`.
- 2026-10-04: `remixapi_AssetKeyEXT`
  (`REMIXAPI_STRUCT_TYPE_ASSET_KEY_EXT` = `0x4E560002`), chained on
  `remixapi_MeshInfo` or `remixapi_MaterialInfo`. It gives the asset the key
  that mods, captures and caches use instead of the mesh hash or material
  content hash, plus aliases (keys older host versions produced) and an
  optional name. Replacements are looked up by key, then by each alias in
  order. It is honored while `rtx.scene.assetKeys` is on (default off).
  Commit `0b31bedd5`.
- 2026-10-06: named export `remixapi_fork_ReplayIdentityTrace(const wchar_t*
  path, remixapi_fork_IdentityReplayResult* result)`, a diagnostics entry
  point for `remix_replay --identity-trace`. It replays a D3D9 identity trace
  (`rtx.d3d9Identity.traceFrames`) through a private draw-call tracker on the
  calling thread and presents nothing. It is declared in
  `src/dxvk/rtx_render/rtx_fork_d3d9_identity_trace_format.h`, not in
  `remix_c.h`, so `remixapi_Interface` and `REMIXAPI_VERSION` are unchanged.
  Commit `4408886d5`.

## [0.1000.1]

Remix Plus rebased onto NVIDIA dxvk-remix `0.6.5`, which grew
`remixapi_MaterialInfoOpaqueEXT` by four DLSS control-mask fields (112 -> 128
bytes on x64, 92 -> 108 on x86). The runtime and the bridge client read those
fields only from callers reporting API version `0.1000.1` or newer; binaries
built against `0.1000.0` keep working and receive upstream's defaults.

### Added
- remixapi_MaterialInfoOpaqueEXT.enableDlssControlMask (inherited from upstream 0.6.5)
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskIntensity (inherited from upstream 0.6.5)
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskToneStrength (inherited from upstream 0.6.5)
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskStructuralStrength (inherited from upstream 0.6.5)

### Changed
- `REMIXAPI_VERSION` is now `0.1000.1` (was `0.1000.0`).

## [0.1000.0]

Remix Plus adopts its own ABI version line (reserved MINOR `1000`), distinct
from stock NVIDIA dxvk-remix `0.6.x`. Because the runtime treats every minor as
breaking while MAJOR is 0, this version makes the runtime reject binaries built
against stock Remix `0.6.x` or against older Remix Plus `0.6.x` — the
`remixapi_Interface` layout and `remixapi_InstanceCategoryBit` ABI differ
between them. Rebuild plugins/hosts against this header.

### Added
- `REMIXAPI_INSTANCE_CATEGORY_BIT_SMOOTH_NORMALS` (bit 24) — the upstream name
  for the category previously exposed (under the fork) only as `LEGACY_EMISSIVE`.
  Use `SMOOTH_NORMALS`; the old alias has been removed (see below).

### Changed
- `remixapi_InstanceCategoryBit` bit values now match upstream NVIDIA exactly.
  An earlier fork build had shifted `IGNORE_ALPHA_CHANNEL` to bit 8 (cascading
  bits 8–20 up by one) to mirror the internal `InstanceCategories` order; this
  is reverted. The C↔internal mapping in `toRtCategories()` is by-name, so the
  public bit values are free to — and now do — match upstream. **Breaking** for
  any consumer that had serialized or hard-coded the shifted bit values.
- `remixapi_Interface.SetCameraMediumMaterial` moved from the middle of the
  struct (between `SetupCamera` and `DrawInstance`) to immediately after
  `Present`, mirroring upstream's canonical layout (upstream `2bac8874`). Fork
  extension functions remain appended after it. **Breaking** struct-offset
  change; rebuild consumers. The `sizeof(remixapi_Interface)` sentinel is
  unchanged (move, not add/remove).
- `REMIXAPI_VERSION` is now `0.1000.0` (was `0.6.4`).

### Fixed
- `remixapi_AutoInstancePersistentLights` no longer emits a per-frame
  `LockDevice` + empty `EmitCs` on the native-D3D9 present path when no external
  (C-API) light has ever been registered and no C-API scene work is queued.
  This empty per-frame dispatch disturbed the light pipeline for native-only
  consumers, manifesting as "persistent lights break all lights / heavy
  flicker." Genuine C-API light consumers are unaffected (the persistent
  re-instancing path is preserved).

### Removed
- `REMIXAPI_INSTANCE_CATEGORY_BIT_LEGACY_EMISSIVE` is **removed**. Its name
  implied emissive behavior, but it routed to bit 24 / `InstanceCategories::SmoothNormals`
  — so any caller using it got a silent wrong-category result. Removing it turns
  that into a compile error; use `REMIXAPI_INSTANCE_CATEGORY_BIT_SMOOTH_NORMALS`.
  Source-only break — the bit layout, enum size, and struct offsets are unchanged
  (bit 24 still exists as `SMOOTH_NORMALS`), so no binary/ABI change and the
  `0.1000.0` version line is unaffected.

## [0.6.5]

### Added
- remixapi_MaterialInfoOpaqueEXT.enableDlssControlMask
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskIntensity
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskToneStrength
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskStructuralStrength

## [0.4.2]

### Added
- MaterialInfoOpaqueEXT.displaceOut

### Changed
- renamed MaterialInfoOpaqueEXT.heightTextureStrength to MaterialInfoOpaqueEXT.displaceIn

### Fixed

### Removed


## [0.4.3]

### Added
- remixapi_MaterialInfoOpaqueSubsurfaceEXT.subsurfaceDiffusionProfile
- remixapi_MaterialInfoOpaqueSubsurfaceEXT.subsurfaceRadius
- remixapi_MaterialInfoOpaqueSubsurfaceEXT.subsurfaceRadiusScale
- remixapi_MaterialInfoOpaqueSubsurfaceEXT.subsurfaceMaxSampleRadius

### Changed

### Fixed

### Removed

### Fixed — retained empty replacements (2026-10-30)

A valid empty USD mesh replacement suppresses the original retained mesh without failing the catalog. Updating the record or reloading a mod can restore its geometry. Unanchored lights stay absent, matching the D3D9 replacement path. No API layout or option changes.
