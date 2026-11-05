# GPU precipitation migration

The internal weather emitter uses the v2 neutral `SceneDraw` interface and the
existing GPU particle spawn, evolve and geometry passes. It submits one emitter
transform; it does not update individual particle positions on the CPU.
The implementation adapts the previous local runtime's weather emitter, whose
design credits xoxor4d's NFS Carbon RTX rain. Existing repository licence notices
remain in place. NVIDIA-derived shader notices are retained.

## Configuration and weather

`rtx.weather.precipitation.enable` is off by default. `occludeUnderCover` and
`enableCollision` are optional and off by default. The global particle controls
include an editable base look and appearance multipliers. Each NUMOS preset can
override the ten scalar fields and color under `rtx.weather.precipitation.*`;
unset fields inherit the base look. Prefix a value with `x` to multiply it.
The preset editor exposes these fields without writing per-frame RTX options.
Legacy `precipitationIntensity`, `precipitationFallSpeed`, and other old field
names are imported; legacy intensity also retains the wet-surface alias.

Typed state blends over the game's explicit weather blend or the preset's
transition time, preserving the current look on an interrupted transition.
Copy, import, save, held sky seeds and sky tuning retain precipitation state.
Values are normalized before reaching particle descriptors. Sky tuning holds
precipitation steady; edit its preset fields separately.
`__weather.precipitation_enable` suspends new emission while existing drops finish
their authored lifetimes. Descriptor dwell and lifetime-based minimum intervals
bound the number of overlapping particle systems during slow snowfall blends.

Transparent precipitation materials retain the sky-lit flag through texture
preloading and surface encoding. Particle lighting uses the new sky provider and
froxel lighting conventions. The obsolete roughness and metallic controls are
excluded: the old emitter had already switched to a translucent material where
those values did not affect its appearance.

## Contact emitters

The mesh-hash lists `rtx.particles.waterContactEmitterMeshes`,
`wetRoadEmitterMeshes` and `puddleEmitterMeshes` are empty by default and editable
under Particle System. They identify API-created emitter meshes by their original
64-bit mesh hash, including when a 32-bit bridge transports the handle.

A dedicated GPU spawn shader projects those emitters onto water, wet tagged
world surfaces or rendered puddles. Other particle systems keep their ordinary
spawn shader. Queries use the completed scene's TLAS and surface/material tables;
they never dereference previous-frame geometry through current-frame bindless
vertex indices. All surface/material accesses are bounded. Water beneath solid
ground is rejected. Both Y-up and Z-up scenes are supported.

Contact wetness follows the same authored weather mixture as surface shading.
Spray fades as the ground dries and stays hidden after reaching dry state, while
its lifetime still follows FIFO retirement. Open-water spray does not inherit
ground drying. Particle records remain 48 bytes with full-float lifetime at
offset 44. Contact modes contribute to the particle system's identity.

## Validation

The `test_precipitation` unit suite checks state normalization, relative overrides,
interrupted blends, stored looks and GPU record contracts. `contactDiagnostics`
is off by default; when enabled it logs delayed GPU shelter/contact counters and
the emitter's typed weather state. These counters are diagnostic readback, not
particle computation or a CPU fallback.
The D3D9 fixture exercises rain, snow, roofs, host suppression, wet roads, puddles,
water over/under solid ground, drying, cullzones, legacy/new presets and Y-up,
with native and x86 bridge runs. Full-game performance acceptance is performed
after the remaining modules, separately from these functional tests.

The recorded migration run passed 20 native scenarios, eight x86 bridge scenarios,
one native fixture repeat and four water/grass/window regressions. All screenshot
components are finite. Grass and window masks match their accepted builds.
The windowed test pumps messages and drains final bridge work before destroying
its window. These are functional tests, not full-game frame-rate measurements.

## GPU no-rain-zone shelter

`rtx.weather.precipitation.occludeWithNoRainZones` defaults off and is exposed as
**Stop In No-Rain Zones** in the precipitation panel. It is independent of the
wetness master switch and graphics quality presets: the boxes describe authored
shelter, not a lower-quality rendering preset. Enable it in a game's config only
when its host publishes valid `__wetness.norain_zones_v1` data.

Weather particles reuse the existing wetness BVH and world-space boxes. Spawn
candidates inside a box, including its boundary, relocate up to eight times;
fully covered emitters produce sleeping particles. The evolve pass intersects
each live particle's complete movement segment with the BVH. This covers top
entry, sideways wind, stationary particles inside newly added boxes and fast
crossings with both endpoints outside. The protected region is the authored box,
not an infinitely tall shadow below it. Other particle emitters are unaffected.
Sleeping particles retain the existing FIFO counter retirement and generate no
visible geometry. Particle records remain 48 bytes; no positions or collision
decisions are computed on the CPU.

The option applies through the frame constants, including to fading weather
emitters. Zone data is independent of the scene TLAS and surface-wetness shader
variant. Simulation reads the previously completed frame's immutable zone buffer
and its matching count; a missing initial buffer means zero zones. CPU work is
limited to the already existing map-data parsing/BVH upload on changes.

Spawn and evolve select independent compiled zone variants only for weather
emitters with nonzero active zone counts. Disabled, empty-list and ordinary
emitters use the base variants, whose SPIR-V has no zone buffer or traversal.
The wetness variant remains an independent axis. All eleven particle shader
variants compiled and passed SPIR-V validation. Reflection verified absence
of the zone buffer in all four base variants and presence in all four zone
variants. The full runtime and extended precipitation unit test compiled
and passed on 2026-10-30.

The existing ray-traced **Stop Under Cover** is independent and unchanged outside
the boxes. With both enabled, covered candidates skip the scene ray and retry;
uncovered candidates retain the upward and landing traces. With only zone shelter
enabled, geometry absent from the authored boxes does not block rain. Screen-space
collision remains separately configurable. No performance saving is assumed until
measured with matched scenes and ray policy.

`rtx.particles.contactDiagnostics` also logs zone candidate rejections, birth and
movement kills, the active node count, and actual scene rays. These delayed GPU
counters are diagnostic readback only. The legacy `rays` field counts particles
entering the ray policy; `sceneRays` counts issued upward/landing rays.

Before implementation, the particle subsystem was compared against NVIDIA
upstream `0ba9973fd856e29fe9cf8dd2f2bc754bcbb2037a` (2026-10-28), fetched on
2026-10-30. Upstream has the same screen-space evolve collision and no equivalent
authored-zone or shelter-ray spawn implementation to reuse. No upstream merge or
third-party modification was performed.

Initial validation: all seven production particle shader variants compiled and
validated. On an RTX 5070 the production zone helper and production CPU BVH builder
passed 327,680 GPU segment queries and 327,680 GPU point queries against an
independent six-plane CPU oracle. Cases include empty/disabled zones, boundaries,
vertical/horizontal/diagonal/high-speed motion, overlapping boxes, two world-axis
orientations, and 4,096 boxes. Output canaries remained intact. This is algorithm
validation. Full-runtime integration subsequently passed sixteen native/bridge
cases: covered births, roof slabs, fast crossing, partial relocation, empty
lists, live disable/removal, Y-up, independent ray fallback, wetness off and
unaffected ordinary contact emitters. Zone-only cases issue zero scene rays.
All images are finite, and covered/empty images show a positive rain difference
well above the native/bridge covered-image difference. Game acceptance and
performance measurements remain pending.

The initial test attempt inherited ray shelter from its base configuration
until deferred option updates settled. It was excluded from acceptance. The
accepted run initializes each case configuration before creating the device
and retains the strict zero-ray checks; no runtime behavior was changed for
this fixture correction.
