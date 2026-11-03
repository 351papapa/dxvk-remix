# GPU water migration

This opt-in subsystem ports the local runtime's ocean deformation, interaction
waves, hull pressure waves, wake flow, shore foam, caustics and underwater volume
passes to the v2 API-neutral scene. It uses the existing NUMOS sky provider.

## Compute ownership

The host supplies water coverage, initial state, time and interaction events over
the pointer-free private water protocol. The runtime owns device-local simulation
buffers. Reused packet inputs are uploaded in the graphics execution queue,
ordered immediately before their consuming dispatches, including multiple packets
received in one frame. Wave stepping, flow transport and pressure projection, hull deformation,
surface normals, foam placement and triangle emission run as compute shaders.
Ray queries perform shore/contact detection and light visibility on the GPU.
Scene foam directly supplies the generated GPU vertex buffers to the BLAS path;
there is no CPU geometry reconstruction. Other generated water buffers use the
existing GPU copy path. Diagnostic readbacks are optional and default off.

The protocol keeps its legacy names and layouts for the existing producer:
`water_controls.h` v3 (124 bytes), `water_gpu_surface.h` v1 (127484 bytes), and
`water_gpu_simulation.h` v3 (`Packet` 216800 bytes, `Shared` 4552504 bytes).
The generated ocean/interaction protocol is Z-up. Authored AnimatedWater surfaces
can use either supported world-up convention for foam, caustics and underwater
effects. No C API struct or interface-table layout changes are required.

## Configuration and assets

`rtx.water.enabled`, `gpuSurface`, `gpuSimulation`, all optional booleans and
diagnostics default off. Every setting has a Water menu control. Enable only the
desired features in the host's authored configuration. Previous runtime defaults
must be copied explicitly when preserving an existing game's look.

`waterMapOrigin` and `waterMapCellSize` describe the producer's fixed 128 by 128
water-level map. World XY coordinates are sampled in the shader; game map offsets
are authored configuration, not runtime constants. Both fields must match the
producer. The surface protocol uses independent coarse/fine grid parameters.

Set DDS paths through `foamColorTexture`, `foamNormalTexture`,
`foamCompositeTexture` and `causticsTexture`. `beachMaskPath` supplies the optional
eligibility mask. Paths resolve relative to the host executable; a `.trex`
bridge executable resolves relative to its parent game folder. Absolute paths
are supported. Asset changes reset their owned material/geometry state.

`foamLineMaterialPath` displays shore foam as scene geometry, with normal material
lighting and reflections. Otherwise the optional screen composite displays the
same GPU density. Simulation remains active for shore wetness even when this
display is disabled. The former fixed diagnostic sheet and height-toggle options
are not part of the migrated subsystem.

## Frame integration

NUMOS weather is resolved before water submits generated geometry. Generated
meshes retain their topology identity; a changed domain invalidates previous
vertex correspondence. Previous-frame density, world origin, clock and height
anchor are captured together for the next foam geometry dispatch.

Caustics operate around path integration. Foam simulation/composite runs after
upscaling, before HDR post-processing. Underwater light is integrated by the native
froxel volumetric renderer before the main composite; there is no separate
full-resolution underwater haze or shaft pass.
Sky-provider values are converted to the renderer's current pre-exposure once;
already encoded scene-light radiance is not scaled a second time. Shore wetness
uses the optional wetness shader variants even when rain wetness is disabled.

NUMOS presets accept `weather.oceanRainWeight` and `weather.oceanStormWeight`.
Weights are finite, clamped and normalized when their sum exceeds one. The clear
weight is the remainder. Ocean gain is `clear + rain * rainWaveMultiplier +
storm * stormWaveMultiplier`; explicit game blends and interrupted runtime
transitions preserve this typed state without per-frame option writes. Saving a
look preserves its ocean weights; tuning the sky holds the current ocean state.

## Camera medium bridge

The matching x86 bridge now exposes `SetCameraMediumMaterial`. It sends a 32-bit
material UID, resolves it on the x64 server, and returns the API result through a
response transaction. Null medium clears the override. Invalid structure types,
unsupported extension chains and unknown handles are rejected. The command is
appended so existing bridge command IDs remain unchanged. Deploy client and
server together.

RemixCoreVC's integrated WaterLab producer restores the GPU handshake. For live
water it waits while the GPU surface or interaction solver is unavailable instead
of invoking the old CPU fallback. GPU event mode leaves CPU height/density arrays
unchanged. Its historical explicit CPU snapshot diagnostic is not used by the
live migration or performance validation.

## Unified water medium

The Minecraft-style path selects a homogeneous camera medium. Enable
`rtx.water.enabled`, `overrideMaterial` and `underwaterEnabled`, and set
`cameraMediumMaterialHash` to the hexadecimal source hash passed through
`SetCameraMediumMaterial`. The default is empty and never identifies arbitrary
glass as water. The current VC integration uses `0x574154455243414D` with an
exactly white camera medium; its water surface remains colored. The host owns
waterline detection and hysteresis. API layouts and bridge commands are unchanged.

Surface transmission, receiver caustics and the underwater volume share
`transmittanceColor` (linear RGB), `transmittanceDistance` (world units), and
`particleExtinctionPerMeter` (inverse meters). The authored baseline extinction
is `sigma0 = -log(T) / distance`, with finite inputs clamped to T=[0.0001,1]
and distance=[0.1,100]. T and distance have no gamma or meter conversion.
The additional neutral particle coefficient beta is converted from inverse meters
to inverse world units using the scene scale. Total extinction is `sigma0 + beta`.
The particle option defaults to zero and exactly preserves the authored baseline
at zero. It describes the water material, so graphics presets do not change it.

`underwaterScattering` selects the scattering fraction f of neutral extinction:
`sigma_s = (min(sigma0) + beta) * f`, and `sigma_a = sigma_t - sigma_s`.
With f in [0,1], scattering cannot exceed extinction in any channel. This allows
visible suspended particles even when a baseline color channel is nearly clear,
without a separate underwater tint. `underwaterAnisotropy` controls the native
forward-scattering approximation. Both scattering controls default to zero.
Increasing beta also reduces water transmission above and below the surface.

The material encodes total extinction as a new T/d pair. Dense media use a shorter
encoding distance so transmission stays at least the smallest normal binary16
value, while distance stays at least .001 world units. Extremely large coefficients
are limited consistently before both material encoding and volume use. The
authored distance remains the fallback volume attenuation range. Changes of scene
scale invalidate translated water materials, and the material cache hashes the
effective encoded pair.
Water forces the native volumetric cache while active even if air fog is disabled.
It does not inherit air fog remapping, atmosphere, noise, ambient approximation
or weather visibility gains. Sun and local lights use the native volume light
sampling and translucent visibility. Entry, exit and optical changes invalidate
volumetric history. State is derived per frame without writing authored options.
The native volumetric quality settings control underwater quality/cost.

When the registered shared GPU water mesh covers the camera, its current base-Z
plane supplies a directional-light entry boundary. The snapshot must be fresh,
and the surrounding coarse vertices and cell mask must be valid and wet. The
CPU wave-height field is displacement, not a water-level map; it does not set the
plane. Missing or invalid coverage falls back to the homogeneous light path.
Changes in boundary availability or height invalidate volume history.

Water-specific shader variants shorten a distant-light sample's virtual position
to the water-entry plane for the existing Beer calculation and ReSTIR target PDF.
The native maximum attenuation distance (five authored measurement distances)
still caps this path; grazing sunlight and very deep samples retain that approximation.
Shader-local metadata retains the original direction and distance for the same
full-length visibility ray, preserving occluders above water. Finite local lights
retain their original sampling and distances. All four ReSTIR stages and three
integration workgroup layouts have water variants, with and without wetness.
The directional-distance correction adds no GPU allocation, pass, or visibility
ray. Atmosphere variants do not contain the water path. Light radiance and cloud shadows still follow the scene.

Receiver caustics keep the same world coordinates, scale, speed and dispersion
on both sides of the waterline. Optional ocean/pool scales describe different
water domains, not camera position. The old independent shaft pattern/scale,
manual fog color, extra density and per-pixel shaft sample controls are removed.
Reflected caustics read F0 from the actual hit water material, matching its IOR
for sun and local lights. Transmitted caustics modulate the native direct-light
solution without another refraction solve or a second medium attenuation.

This matches the homogeneous camera-medium approach, not bounded local water
volumes: the volume does not clip at the waterline or reconstruct refraction and
focused sunlight from the wave surface. It cannot model several water media in
one view. The exactly neutral camera material prevents duplicate camera-path
attenuation; integrations supplying a colored camera material must neutralize it
before enabling this profile. Surface materials still shade transmission normally.

## Unified receiver and volume field (2026-11-03)

The implementation uses the same source-normalized dynamic focusing field for the
projected ocean receiver and the selected directional light in the water volume.
Both evaluate the shared water-plane coordinate Q / causticsScale with the same
causticsSpeed clock, randomized setting and primary UV warp/advection. There is
no eightfold volume repeat or separate slow envelope translation. Pools, reflected
caustics and the disabled-feature path retain their existing fine-pattern
evaluation. No option or default is added or changed.

The static R16 cache still stores saturate(source.r)^4 and a full mip chain.
The shared sampler evaluates one primary field; randomized sampling blends
three tiles with normalized nonnegative triangular weights. It does not repeat
the fine pattern's RMS/min/square shaping. Let D be the guarded source-cache
mean. The sample is normalized by D, and its residual around sourceMean / D
is multiplied by exp2(0.5 * min(max(lod, 0), 4)), at most four. The result is
bounded to [0, min(8, 1 / D)]. This artistic mip-contrast recovery keeps a constant
source constant across LODs; it cannot reconstruct detail already fully averaged
away. A nonfinite or near-black source mean returns neutral one for volume
illumination. The UV warps and contrast adjustment do not establish global
dynamic energy conservation. Clipping contrast-expanded residuals can raise
the field average even when the source mean stays fixed.

The projected receiver uses only this field whenever volumetricCausticsStrength
is positive. At zero it retains the original pattern, so changing from zero to
a positive value can switch the receiver's appearance; there is no claim of
continuity at zero or a mixture of independently animated patterns.
Its constant amplitude is 2 * sourceMean, and a nonfinite or near-black source
mean produces no receiver enhancement. Scalar and RGB-average receiver pattern
bounds remain two. Dispersion samples the same field three times and retains
the middle sample's RGB mean; the shared peak protection gives a conservative
single-channel bound of six after receiver gain. No additional two-unit
per-channel clamp is applied. These bounds precede authored receiver strength
and depth fade. The surface enhancement still does not isolate the selected
directional source, and equal average brightness to the former pattern is not claimed.

The volume's isotropic footprint uses the larger projected X/Y axis extent
instead of their sum. Strongly sheared footprints can alias more under this
approximation. Its cookie projection uses the guarded, normalized center direction
of the selected distant light, matching the receiver; the native sampled light
direction, Beer distance and full visibility ray remain untouched. Depth, surface
and coverage fades and native integration remain in use. Surface cones and
froxel footprints have different filtering widths, and the volume integrates
along a ray, so shared coordinates do not imply identical final pixels.
At the authored fine scale, spatial or path integration can still average away
visible columns; the renderer does not compensate by widening the pattern.

No dynamic atlas, per-frame cache-generation pass, new visibility ray or resource
binding is added. Randomized volume evaluation needs three field samples plus
one source-mean sample; the nonrandomized path needs one plus the mean.
Receiver dispersion evaluates the same field three times. Source-cache reuse,
native volume quality and history policy remain unchanged.

### Validation (2026-11-03)

The two-layer 53F0 prototype did not retain clear empty-water columns with the
actual game cookie at scale 2.98. Final candidate 18C46994 uses the single-field
implementation above. Native VC capture vc-unified-scale298.png shows clear,
fine shafts at the original public scale 2.98 without retuning the user's water
parameters. The synthetic fixture is more restrictive: columns remain weak at
2.98, improve at 5.96, and are clear at 11.92 while the receiver pattern enlarges
with the same shared scale. Those larger fixture scales are not game deployment
requirements and do not establish equivalent appearance across views.

At fixture scale 11.92 and speed 3, all 21 requested native frames were saved
over a 2.829-second request interval. Columns persisted and moved with the
receiver pattern in the inspected frames, without an observed whole-image flash.
Fine grain remains. This short, every-eight-frame sample does not prove long-run
stability or cover every intervening frame; its variance includes intended
pattern motion and native stochastic/refraction effects. All four final fixture
runs produced finite captures and recorded exit 0. Existing common-device-object
disposal diagnostics remain, so these are not clean-teardown/all-pass results.

The first fixed-view pair, in execution order 638790EE -> 18C46994, gave
primary base FPS 50.4236 -> 48.6003 (-3.62%) and supplementary
49.1715 -> 50.2109 (+2.11%). The reference primary interval contained a mod
hot reload and incomplete scene loading, so it is not a clean performance
comparison. All original samples and both intervals remain retained.

A reverse-order repetition was declared before execution. In execution order
18C46994 -> 638790EE, primary base FPS was 49.0701 -> 48.9045 (-0.337%),
with mean frame time 20.379 -> 20.448 ms. The supplementary interval was
50.0601 -> 50.4134 FPS (+0.706%), or 19.976 -> 19.836 ms. Thus the candidate
was slightly faster in the primary interval and slightly slower in the supplement.
The controlled repetition did not show an obvious regression in this fixed
view. It does not establish statistical significance, universal performance,
or a new comparison with the original cf86 route baseline.

The final build audit covered 1,626 shader dependencies, all nine affected
variants and their layouts; 14 prior shader variants were unchanged.
Independent CPU dispersion/gain checks supplement the GPU evidence but do not
replace it. The final runtime is installed; the original test configuration and
saves were restored and the game closed. The following section preserves the
638790EE baseline and its historical eightfold-envelope implementation and
measurements without attributing them to this final candidate.

## Caustic-pattern transmission in the water volume

`rtx.water.volumetricCausticsStrength` controls underwater shaft contrast from
0 to 1 and defaults to 0. It requires `enabled`, `causticsEnabled`, positive
`causticsStrength`, a resident `causticsTexture`, and a fresh registered Z-up
GPU water surface. Underwater operation also requires the configured camera
water medium. Zero keeps the existing receiver projection and volume path.
The control describes an authored lighting effect and is independent of graphics
presets. The validation below reports the visual result and measured cost of
the VC contrast-one profile.

The renderer selects the strongest upward directional light by projected flux.
This is usually the sun, but a moon or another directional source can qualify;
the selection does not classify lights by a game-specific solar identity.
Only that light's existing froxel light sample receives the scalar focusing
multiplier. Local lights and sky/dome samples keep their normal contributions. An
above-water camera does not enable the mask for air fog; ocean receiver caustics
can still use the matching projected pattern while the option is enabled.

At world position P below the shared base plane H, the sample coordinate is
Q = P + L * ((H - P.z) / L.z), where L points toward the selected light.
Receiver caustics and froxels share a broad transmission envelope at Q,
using the same caustic texture, authored scale, animation speed and clock.
The receiver retains its fine web at the authored scale; the envelope spans
eight times that tile width and moves by time * (0.004, 0.003) in envelope UV.
Volume lighting samples only this broad field to avoid averaging many fine
filaments along the viewing ray. The receiver retains its geometry query to establish
the hit ocean domain and depth. The projected Q is checked against the registered
coarse/fine water coverage, including holes and domain edges. Authored pools
retain their separate receiver scale and are not assigned the ocean volume mask.
Low-angle light, excessive depth, insufficient spatial resolution and coverage
boundaries fade the mask's contrast. Invalid or above-plane volume samples retain
unmodified illumination. Coverage failure suppresses the projected receiver
caustic contribution. Coverage checks the current cell and only the nearer
neighbor on each axis: the opposite edge's fade is already one. This reduces
five cell checks to three while retaining the same coarse/fine masks, wet corners,
height validity and boundary fade. Grid-header values are loaded once per
coverage invocation. A local coarse-cell key reuses four-corner wet/height
validation only after every corner succeeds; crossing into another cell
performs the complete validation again. Each point still selects its own
coarse/fine domain and reads its own mask, preserving holes and shoreline fades.
No additional geometry or visibility ray is traced.

On activation, the source red channel is sanitized and shaped as
`saturate(r)^4` into a separate R16_SFLOAT cache. Each cache axis is rounded up
to a power of two; linear 2x2 mip reductions then provide an area mean in the
last 1x1 mip. Sampling returns F = min(shaped / guardedMean, 8). The denominator
includes a small conservative allowance for half-float rounding at each level.
An invalid or near-zero mean returns neutral F=1. The complete resampled tile's
mean is at most one; peak clipping only reduces it. This constraint does not
claim conservation over an arbitrary cropped scene with varying coverage,
depth fades, visibility and the receiver's existing artistic gain.

The volume multiplier is C = lerp(1, F, c), where the authored contrast and
validity fades give c in [0,1]. Local values can exceed one, concentrating the
selected source instead of only darkening it. This is texture-driven artistic
focusing, not wave-surface refraction or a photon transport solution. It does
not increase particle density or change water's absorption/scattering.
Receiver caustics keep their original fine web and only multiply it by min(F,1),
avoiding a second amplification of their existing gain. Contrast blends from
the original vertical web at zero to the projected, attenuated web at one.
Baseline surface lighting is not multiplied by this receiver envelope.

The shared sampler reads the broad field and its final-mip mean: two texture
samples instead of the fine web's randomized six-tap evaluation. Volume mip
filtering uses the far face's projected XY cross-section at the eightfold envelope
scale. Native per-frame froxel jitter and history integrate the depth interval,
which is excluded from the isotropic footprint. Temporal noise can increase
when looking across the source, at coarse depth slices or after camera motion.
The existing Beer path, SH phase approximation, cloud shadow and full visibility
ray remain in use; no additional per-frame fullscreen pass or ray is added.
ReSTIR proposals and reservoir targets retain the native lighting distribution;
the field changes only the final selected sample's radiance. Bright focusing
peaks are therefore not importance-sampled by the reservoir and can converge noisily.

All supported 2D sources, including authored multi-mip textures, receive this
shaped cache. A small compute dispatch writes mip0 and native mip generation
fills the chain once per activation or source-view change. The original texture
is retained for receiver fine detail. Continuous requests reuse the cache by
source cookie, including above-water receivers; disabling the effect or losing
its valid surface/source releases it. Unsupported views disable the projected
envelope. Receiver envelope filtering uses the actual cache dimensions.

While this pattern animates,
its radiance accumulation is capped at eight frames to reduce temporal smearing;
this can expose more noise and is not a free quality-preserving optimization.
Resource, medium or pattern-setting changes invalidate the affected history.

The froxel medium remains homogeneous, with one flat base plane and no wave-normal
refraction. Camera compositing clips the initial underwater segment at that plane
for upward views, including primary radiance, alpha surfaces and transmittance
control masks. A ray that exits the water uses a finite endpoint rather than the
native miss path's infinite attenuation. The host forwards the combined active,
valid-surface and Z-up flag plus the surface height through two former padding
words in `CompositeArgs`; its total size and all other field offsets are unchanged.
A compile-only check uses the real `CompositeArgs` and `RaytraceArgs` headers,
the previous layout, and the actual host assignments. Separately, the extracted
boundary helper passed 30,027 numerical CPU checks, including 10,000 randomized
cases. These checks do not establish GPU or visual acceptance. This compositing
boundary describes the initial camera segment.

### Resolved water and air segments (2026-11-03)

The geometry resolver intersects each finite or missed ray segment with the
half-space below the registered base water plane before evaluating native volume
attenuation. This includes PSR reflection/refraction branches, indirect rays and
unordered emissive surfaces. Ordinary surface-to-light visibility uses the
same water-segment calculation, keeping the complete occlusion ray and its
existing POM origin and distance. Portal visibility and queries that retain
attenuation at opaque hits keep the native fallback. A ray that has left the
water no longer attenuates sky or the sun disk as an additional maximum-length
water segment. Finite air
segments also contribute no water absorption; reflection and re-entry retain
the absorption for their actual water length.

The existing maximum attenuation distance applies after excluding air. An
upward miss from below the plane uses its finite exit distance; a downward miss
keeps the native capped deep-water tail. Inactive water, an invalid plane, a
non-Z-up scene or invalid ray inputs preserve the native fallback. Atmospheric
attenuation uses its existing path. No ray, texture sample, per-frame pass,
setting, constant-buffer field or material parameter is added.

The resolver and camera compositing share the same validated water-map plane.
They do not intersect the displaced wave surface or implement nested water
volumes. Small wave-height and numerical-origin offsets near the surface remain
an approximation. Material refraction, total internal reflection, caustic scale,
volume integration quality and accumulation settings retain their existing paths.
This boundary correction does not remove the wavelength-dependent absorption of
the water segment: a strongly red-absorbing transmittance setting still produces
cyan or blue light underwater.

The extracted shader helper and wrapper passed 300,094 numerical checks against
an independent endpoint-height reference, including segment reversal/additivity,
finite and infinite paths, crossings, parallel rays and fallback cases. A further
11 checks compile the actual visibility branch with CPU adapters to verify its
gates and unchanged shadow-ray/POM handling. These CPU checks are separate from
rendering validation. Build F2BE9418 completed, and the build audit verified all
1,502 affected dependency consumers and their complete SPIR-V arrays embedded in
the DLL. All 34 native volume integration/ReSTIR variants remain byte-identical
to build 18C46994; the changed kernels resolve surface and path attenuation.

Native VC captures with the original optical parameters show white/gray clouds
through the water surface and retain visible shafts. The brighter, greener
surface and floor reduce their apparent contrast; these tonemapped captures do
not measure a loss of volume scattering energy. The separately calibrated final
profile is recorded below; it is not part of the following DLL-only comparison.

Both fixed-view pairs used the same original configuration. The table reports
candidate F2BE9418 base-FPS change relative to 18C46994, irrespective of launch
order. Every declared window and its spikes remain retained.

| Cumulative measured samples | First pair, 18C46994 then F2BE9418 | Warm reverse pair, F2BE9418 then 18C46994 |
| --- | ---: | ---: |
| 2401-3600 | -3.44% | -0.194% |
| 4801-6000 | -4.78% | -0.782% |
| 8401-9600 | -5.30% | -1.677% |

The reverse pair declared samples 8401-9600 as its primary window before either
run. In baseline-to-candidate order, that window measured 50.6252 -> 49.7760 base
FPS, total GPU 17.640 -> 17.890 ms and Volumetrics 1.730 -> 1.600 ms. GPU values
are medians of ten logged window p50 values, not mean costs or pooled medians.
The first pair included cold compilation, a candidate primary-window hot reload
and larger scene/OMM differences. The warmed reverse scenes are closer but still
not identical; its smaller gap does not establish statistical equivalence or a
speedup. In particular, the first pair's late Volumetrics increase did not recur,
but the candidate's base FPS remains lower in all three reverse windows. Neither
pair is discarded or replaced by the other.

The accepted final VC profile uses F2BE9418 with exactly two changes to existing
game settings: `transmittanceColor.R` changes from 0 to 0.51138175, and
`causticsStrength` changes from 2.65 to 1.0. This is optical color and receiver
brightness calibration, not lower rendering quality or fewer samples. Green and
blue transmittance, distance 17.9, particle extinction 0.01/m, scattering fraction
0.9 and anisotropy 0.4 are unchanged. Neutral sigma_s remains 0.00958578/m;
material IOR 1.5, caustic scale 2.98, speed 3, volume contrast 1 and native volume
quality/accumulation settings are also unchanged.

Final native captures retain visible underwater columns, white/gray clouds seen
through the surface and a normal above-water view. The saved exit log records
both water medium and volume caustics becoming inactive at 13:24:33.624. The game
closed normally; temporary validation state was restored and save hashes were
unchanged before applying only the two final settings to the original config.

The fresh final-profile run retained all three declared windows:

| Cumulative measured samples | Base FPS | Total GPU ms | Volumetrics ms |
| --- | ---: | ---: | ---: |
| 2401-3600 | 48.723 | 18.685 | 2.370 |
| 4801-6000 | 50.403 | 17.730 | 1.615 |
| 8401-9600 | 50.073 | 17.595 | 1.600 |

Relative to the warmed reverse-run 18C46994 reference, the final profile's FPS
changes are -2.456%, -0.247% and -1.092%; the late primary comparison is
50.6252 -> 50.0726 FPS. Relative to the original first-run 18C46994 reference,
they are -3.664%, -3.160% and -3.140%. All earlier original-profile, reverse-pair
and red-only results remain retained. These final comparisons include two optical
setting changes and nonidentical scene/OMM populations, so they do not isolate
code cost or establish statistical equivalence, universal parity or a speedup.
GPU values retain the window-p50 convention above; no 1% low is inferred from
summary logs. `balanced-optics-review.json` and `final-profile.json` in the
validation artifacts record the complete measurements and exact deployed delta.

### Caustic transmission validation status (2026-11-03)

The five-to-three coverage change matched its extracted CPU reference in
221,472 cases. The final header/corner cache matched the actual preceding shader
bit for bit in 1,014,920 CPU cases, including NaN/infinity, invalid spacing,
coarse/fine boundaries and adjacent representable coordinates. Mask reads remain
unchanged. Source-level read counters are not GPU transaction counts or a speedup
measurement. The normalized field also passed 237,148 extracted CPU checks,
including endpoint/continuity tests and an independent binary16 mip-rounding model.

The isotropic-footprint, fine-web XY and bounded-envelope candidates did not
establish clear shafts. The normalized build `20A32924` produced distinct columns
through empty water with both a striped diagnostic and the actual game cookie;
the fixed live-game underwater camera also showed visible shafts. Final build
`638790EE` retained the columns in repeated OFF/ON fixture captures. Its matched
images closely agree with the preceding normalized build, without claiming
bit-identical rendered images. The final cache change does not alter density,
optics, peak limits, native volume quality or history length.

Short static-cookie and animated-cookie sequences on `20A32924` retained column
structure without a large whole-image flash in the sampled frames. Visible grain
remains; these approximately three-second observations do not establish flicker-free
gameplay, and final-image variance includes native stochastic shading/refraction.
The same build's no-selected-light cases created no water-volume field or light.
Above-water cases kept the camera water volume inactive while allowing the shared
receiver projection to change. Those temporal/negative tests were not rerun or
relabeled as `638790EE` tests. One no-light runner encountered a file-sharing error
and did not record the exit code; its render evidence remains valid, but the case
is not an exit-success result. Existing common-device-object disposal diagnostics
remain present, including in the final two fixtures, despite their recorded exit 0.

A matched 180-second Buffalo route compared original `cf86bb3e` with final
`638790EE`. Only the runtime DLL changed; plugins and rendering configuration
matched. Both formal runs were valid, and all 6,170 / 6,344 base-frame samples
were retained, including the initial long frame, after the planned warmup.
Average base FPS was 34.2172 -> 35.2166 (+2.92%), while 1% low was
16.9454 -> 16.2821 (-3.91%) and 0.1% low was 7.7974 -> 7.5341 (-3.38%).
The candidate therefore did not meet or exceed every metric. This single,
mostly above-water route measures the whole runtime and does not establish
statistical significance or isolate underwater shaft cost. Small route/clock
differences and dynamic scene work remain sources of variation.

A separate fixed underwater OFF/ON pair used final `638790EE` for both runs,
with rendering settings differing only in `volumetricCausticsStrength` 0 / 1.
Native final screenshots show multiple distinct water columns when enabled and
no such columns when disabled. Both runs retained 1114x626 rendering reconstructed
to 1920x1080, DLSS Balanced with Ray Reconstruction and native 4x frame generation.
The reported base FPS excludes generated display frames.

The preselected primary interval, measured-sample ordinals 2401-3600, retained
all ten windows (1,200 base frames and 600 GPU timestamp samples per run).
Base FPS was 51.3373 -> 49.0244 (-4.51%), with weighted mean frame time
19.479 -> 20.398 ms (+0.919 ms). Median window-p50 volumetric time was
1.380 -> 2.290 ms (+0.910 ms), and total Remix GPU time was
17.375 -> 18.490 ms. These GPU summaries are medians of window p50s, not
pooled percentiles or average stage times.

The preselected supplementary interval, ordinals 4801-6000, retained the same
window/sample counts: base FPS 51.5650 -> 48.9021 (-5.16%), volumetrics
1.415 -> 2.280 ms. Thus the observed fixed underwater switch cost was about
4.5-5.2% of base FPS. This cost is separate from the whole-runtime route's
+2.92% average-FPS result. Scene/OMM/BLAS variation remains in both intervals;
one pair does not establish statistical significance or guarantee the same cost
in other views. The runtime default remains zero.

## Validation

Unit coverage includes foam argument completeness/layout, production sheet
bounds, partial-triangle boundary rules and weather interpolation. The boundary
unit test is a CPU reference, so real GPU scenes are also required.
`tests/rtx/apps/D3D9Water` supplies a synthetic private-protocol producer for native
x64 and bridged x86 rendering: waves, interaction impulses, domain changes,
disable/re-enable, camera-medium transitions and long-run paired foam validation.
Full-game performance is measured separately after all local modules are ported.

### Particle and directional-light validation (2026-11-02)

The accepted DLL `51f6d0ef` with beta=0 was compared with `0c741ab3` at
beta=0.01/m. The 720p fixture used the same executable and normal, unsandboxed
launch conditions. Across three 60-sample windows, the median of window p50s
was 12.79 -> 12.85 ms for total GPU time and 0.28 -> 0.28 ms for volumetrics.

VC validation held noon and camera (-576, -1488, 3) fixed and used all nine
complete windows over frames 2400-3600. Median window-p50 GPU time was
17.50 -> 17.65 ms and volumetrics 1.55 -> 1.57 ms. Both runs rendered at
1114x626 and reconstructed to 1920x1080 with DLSS and Ray Reconstruction,
with unchanged water color, scattering 0.9, anisotropy 0.4, caustics scale
2.98 and graphics quality. OMM and BLAS work remained dynamic, so the small
total-GPU difference cannot establish a performance gain or isolate this path's
cost. No quality reduction was used. The validated beta=0.01/m profile is
deployed in the game; the runtime default remains zero.

The preselected late interval (frames 4800-6000) supplied nine windows and
540 GPU samples per run. GPU time was 16.95 -> 17.62 ms (+0.67 ms, 3.95%),
and volumetrics 1.61 -> 1.63 ms (+0.02 ms). Baseline OMM coverage was
66.87-67.47%, compared with 50.00-50.69% in the updated run. Volumetric
time stayed close, but scene work varied; the total-GPU increase cannot
be attributed solely to water cost.

Suspended scattering is visible, but this open-water camera did not show
distinct light shafts. The entry-plane correction retains the native phase
approximation and does not implement wave-surface refraction or focused solar
flux. It is not validation of volumetric caustics.

Crashpad's named-pipe access failures were reproduced only under the test
sandbox. Matching runs outside it exited with code 0 on both DLLs and completed
Shutdown. Both builds still emitted the diagnostic for 57 common objects;
normal process exit does not establish that resource-lifetime issue is resolved.
Use normal, unsandboxed launches for this graphical fixture.

## Stable world-cell addressing

The authored and live foam grids use a shared unsigned-magnitude Euclidean
remainder for their ring-buffer slots. Every cell maps to a unique output region
across either world axis, including negative coordinates. GPU geometry and the
CPU contract test share this helper; the CPU test does not generate production
geometry. The compiled-kernel check rejects signed remainder instructions and
verifies the 8 by 8 by 1 thread groups for all six foam variants.

Vulkan does not define negative operands for `OpSRem`/`OpSMod` without the
`maintenance8` feature. A signed `(cell % side + side) % side` expression can
therefore alias output slots and create concurrent vertex writes. See the
[Vulkan SPIR-V execution rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).

The optional paired-triangle diagnostic compares every word of all 1,181,616
vertices against a second execution of the same kernel and the other kernel
variant. Three asynchronous snapshots distinguish unstable execution from an
optimization difference. Diagnostics are excluded from performance measurements.

Shore wetness reserves common bindings 235/236. The SHARC deferred buffers keep
230 through 233. The water weather unit test checks the combined descriptor slot
set for duplicates, including configurations whose shaders do not use wetness.


## Divergent geometry loads

Each foam-field invocation can hit a different scene mesh. Both the descriptor and the element pointer consumed by `OpLoad` therefore need non-uniform resource access semantics. `foamGeometryLoad` emits that element access explicitly with `StorageBufferArrayNonUniformIndexing`. The scope is limited to the foam triangle helper; all index lookup, transforms, ray queries and field updates remain on the GPU.

The bundled Slang version is `2025.10.4-39-g496fab196`. Its ordinary StructuredBuffer-array expression decorates only the descriptor pointer. The affected generated `water_foam_update` and culled shaders contained 15 undecorated divergent load operands. Both optimized outputs now contain 15 correctly decorated operands. The issue matches the compiler's [NonUniform operand report](https://github.com/shader-slang/slang/issues/10525); no external toolchain or dependency was modified.

An actual game fault on 2026-10-30 reported misaligned accesses at GPU PCs `0x45c0` and `0x7e60`. The exact remapped shader hash `4698fd83c52ba07f` maps those PCs to 32-bit triangle-index loads. The fault happened after an F5 vegetation toggle, but a later same-build off/on session did not reproduce it. This repair addresses a confirmed shader-contract defect at those loads; it is not yet proof that every cause of the intermittent game crash is removed.

A standalone RTX 5070 test extracts the production helper and checks 2,097,152 uint32, uint16 and float reads across 524,288 divergent invocations and 32 descriptor permutations, including output canaries. It passes. The unfixed test also passes on this driver, so it verifies result preservation rather than serving as a deterministic crash reproducer. Full-game stress and final FPS acceptance remain required.

## World-axis foam regression

The supplied 2026-10-27 video was inspected at one-second intervals and in full
frames around the material-path switch. The scene-material foam becomes a thin,
elongated bright strip across the channel. This is consistent with geometry
aliasing, but the exact recorded game location has not yet been replayed.

The old signed remainder expression was tested directly on an RTX 5070 against
an independent 64-bit Euclidean remainder oracle. Across 1,200 grids and 30,894,400
cells (sides 60, 90 and 256), it produced 3,010,630 wrong mappings and 193,462
duplicate slots. Ordinary negative origins such as (-1024,-1024) reproduce the
problem; it is not limited to integer extremes. The migrated unsigned-magnitude
`waterFoamWorldCellSlot` returned zero mismatches, duplicate slots or out-of-range
slots on the same cases, including grids crossing either world axis.

Current full runtime 3C15A768 was also checked through native and x86 bridge hosts.
At frame 600 each compared all 1,181,616 foam vertices (10,634,544 words), with zero
non-finite words, repeated-dispatch mismatches or paired-variant mismatches. Both
output images were finite. These tests establish the addressing correction and
current GPU output consistency, not exact visual acceptance at the video's scene.

## Camera-following water lighting

Underwater camera height and projected caustic coverage query the producer's
existing 128 by 128 water-level map. They do not depend on the player-centered
local simulation mesh covering the camera. Both the camera cell and each
projected surface point must contain a finite water level; projected points must
match the camera plane. Invalid cells, land and map bounds reject coverage.
The existing short shoreline fade remains bounded by half a map cell.

The optical map stores four floats of metadata followed by 16,384 water levels
(65,552 bytes), with a matching CPU cache. Uploads occur when the producer map
epoch or coordinate metadata changes. Normal frames reuse it. Protocol and
surface-producer heartbeats expire after 500 ms; disabled local flow or an
interaction solver does not by itself invalidate water-level information.

The water simulation radius, cell count, froxel grid, ray count, caustic depth,
pattern scale and dispatch dimensions are unchanged. Volumetric lighting is
evaluated in the current camera's existing froxel volume. The level map is
lookup data, not a larger simulated or rendered area.

The Water panel groups existing controls into appearance, waves/interactions,
foam/wet shore and caustics/underwater lighting. Integration resources, maps,
simulation details and diagnostic controls are folded into Advanced water
settings. Existing option calls and defaults are unchanged. Detail wave strength
controls surface distortion; refraction also follows the material IOR.


## Underwater post-processing

Lens Rain is hidden when the camera's resolved starting medium matches the
configured nonzero water-medium material hash. Entering water releases its drop
state; leaving water initializes clean state and follows the existing weather,
shelter and fade settings. This classification is independent of the water
volumetric-lighting and material-override switches. Forced rain previews are
also hidden underwater.

External effects receive the same camera classification in cameraMediumFlags.
The supplied Trails effect passes the current RGBA image through underwater,
without feedback tint, darkening or history blending. Crossing the water
boundary clears feedback history; the authored effect resumes on land.

No underwater depth-of-field blur is added. Water absorption and volumetric
scattering retain their existing distance response and surface-exit clipping.
The camera's depth-of-field settings remain independent.
