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
