# Particle lighting compatibility migration

The v2 NUMOS sky provider and pristine RTXDI cache writer are retained. This port restores confirmed custom surface-consumer behavior from the old runtime; all illumination evaluation remains in GPU shaders. CPU work computes per-frame scalar controls, as the old constant setup did; no particle, ray, instance or sky sampling is moved to CPU.

## Controls and scope

- `rtx.particles.rescaleSkyLitFroxelGain` (false): sky-lit opaque particle materials undo the actual current frame's consumer gain. Translucent sky-lit particles already read the raw cache in v2, so they apply only the target gain. Unmarked particle/decal materials retain v2 behavior.
- `rtx.particles.skyAmbientHemisphere` (false): only opaque sky-lit materials use fixed zenith plus four45-degree upper-hemisphere taps, weighting each ring bycos45 and limiting each component to4times the zenith. It removes view-direction dependence but is not a scene-visibility test. New provider pre-exposure/radiometry stays intact. Translucent sky ambient retains v2's two taps, matching the scope of the old opaque-only custom change.
- `rtx.volumetrics.particleSkyLitFroxelGain` (1, UI0..32): target cache contribution for either compatibility gain control. The parameter alone does nothing until a control is enabled. The game already explicitly used1.
- `rtx.stochasticAlphaBlendRescaleFroxelGain` (false): independent original-name switch restoring the old rescale on all SAB radiance-volume fallback surfaces. AlphaBlendSurface carries no particle marker. Neighbor-derived lighting is untouched. UI explicitly says all alpha surfaces.

UI: Particle Surface Lighting (independent of the volumetric fog enable switch) exposes the two particle switches and target gain. Stochastic Alpha Blend exposes its independent rescale checkbox. No setting is applied through per-frame string writes.

## Safe gain math

For consumer-scaled cache `C*g*edgeFade`, the enabled scale is `min(target,32)/g`. For v2's raw translucent cache `C`, it is `min(target,32)`, never an inverse gain. Disabled controls, nonfinite/negative target, nonfinite gain, gain<=1e-6, or an inverse scale>4096 return1: they leave the current path unscaled rather than amplify a skipped/nearly disabled cache. Zero target is valid and removes only the cache term. At the game's nominal g=.008,target1 the consumer scale is125 and raw scale1; sky/emission/fog are not multiplied125x. The existing shader operation is skipped when scale==1, preserving the default and raw-target1 paths.

The hemisphere samples use world Y-up or Z-up from the renderer and call `skyProviderAmbient`, retaining current sky-provider semantics. The old white tint default has no effect and no explicit game tint exists, so no inactive tint parameter is reintroduced.

## Layout and verification

RaytraceArgs places the two scalar fields after softOmmFlags: consumer scale at8792 and raw scale at8796, with a total size of8800bytes. CompositeArgs places the particle froxel gain at5772 and the SAB average-search control at5776, with a total size of5792bytes. The dev NUMOS AtmosphereArgs is1440bytes (16bytes larger than the previous pin), shifting both structures' later fields; the SAB controls add another16bytes to CompositeArgs. CPU objects and the SPIR-V layout probe agree on these offsets. Rebuild binaries and shaders together.

The opaque hemisphere path is a compile-time GBuffer feature selected by `rtx.particles.skyAmbientHemisphere`; it needs no constant-buffer boolean. Each existing GBuffer variant has independent wetness and particle-sky axes. The selector uses actual wetness variants for slim, compact, validation, decal and PSR preparation paths as well as the standard path. Sky-lit surface resolution is enabled in the GBuffer resolver; passes that include only the generic resolver do not compile the sky branch.

Production shader compilation and spirv-val passed for SAB, eight GBuffer configurations (standard and PSR+NRC+WBOIT, each with both wetness and particle-sky axes), and the layout probe. The production gain helper passed352 finite/NaN/infinite/negative/zero/near-zero/threshold/on-off/raw/consumer cases. These checks do not establish whole-runtime image or game performance acceptance.

GPU acceptance must compare opaque/SAB on/off for gain0,.008,1; raw translucent target1 must be bit-exact on/off. Compare marked and unmarked materials, neighbor-found versus fallback, Y/Z-up camera rotations, a narrow bright sun-like direction and nighttime local lamps. The gain0 result is existing-path identity, not guaranteed darkness in the raw translucent path. The prepared plan is in the migration work directory's particle-lighting-port/gpu-acceptance-plan.json; it has not been executed here.


## Translucent particle resolve coverage

The three ordered/unordered resolve call sites preserve the old particle approximation flag. The raw-cache branch applies only when transmission approximation is requested and the GBuffer resolver compiles sky-lit handling. The scalar gain helper and compiled branch are not proof that a precipitation scene exercises this path.

Both the old runtime and this runtime return from translucent alpha-state derivation before assigning the Particle category to the surface's alpha state. Precipitation consequently retains the full translucent PBR path, including its diffuse layer and reflection, rather than the opacity-particle lighting approximation. Changing this classification just to exercise the cache test would change the original rain appearance and is outside this port.

The raw-cache precipitation fixtures are not acceptance evidence for the approximation branch: their gain response is no larger than repeated-run noise. Validate precipitation separately with an enabled/disabled positive control and a matched before/after full-PBR comparison. The missing call-site flags alone did not explain the fixture's lack of response. The already validated opaque/SAB and hemisphere controls have their own coverage and are unaffected by this limitation. All particle simulation and shading remain on the GPU.
