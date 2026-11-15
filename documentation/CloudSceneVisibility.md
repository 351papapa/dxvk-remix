# Cloud scene visibility

The NUMOS screen cloud march stops at the primary surface. That depth clamp determines
which cloud lies in front of a building, but does not determine whether sunlight can reach
the cloud. Scene visibility is evaluated separately from the camera depth.

The host builds scene variants of NUMOS's normal and fast screen passes. Each nonempty
cloud sample traces toward the sun through the opaque scene TLAS. Its direct lighting is
multiplied by that visibility while its ambient lighting remains. Lighting LOD caches the
direct and ambient terms separately so a reused lighting sample cannot reuse another
sample's scene shadow. Empty density samples perform no shadow query.

The cloud diffraction aureole accounts for light travelling from the source to
each cloud sample and then to the camera. Clamping the camera march at a building
does not remove the incident cloud column behind that building. Both cloud layers
contribute to the incident optical depth, independently of the layer being shaded.

The host retains the library's analytic angular diffraction lobe and normalizes its
source using the sampled forward transport:

```text
w_i = deltaTau_i * exp(f * tauView_i)
scale = min(1, exp(tauVisible) * sum(w_i * visibility_i * exp(-tauView_i - tauSun_i)) / sum(w_i))
```

Here `tauView` is optical depth to the sample, `tauSun` is the combined cloud depth
from that sample toward the source, and `f` is the droplet forward fraction. The two
sums are accumulated independently in logarithmic form to avoid float32 underflow.
A complete homogeneous, collinear column retains unit scale when the density
fields agree. A finite visible column of depth L with total source-camera depth T
retains the missing attenuation `exp(L-T)`. Added incident extinction cannot raise
the correction when the camera column is held fixed. A fully scene-blocked source
contributes zero. Spatial density sampling, near-camera density fading and shadow
LUT filtering can make a real column differ from this homogeneous identity.

Ordinary lighting refinement reuses its already combined `dSun`. Samples skipped
by that LOD evaluate only the sun-depth path, avoiding sky and horizon lighting
work. A cached sample-to-sun depth is not reused: it changes along even a uniform
camera ray. A total-path-depth cache is also insufficient across changing cloud
shadows. After the ordinary march stops at low transmittance, the forward channel
continues over the finite camera segment and remaining shell crossings. Up to 16
growing segments cover each crossing, including its final remainder. Each nonempty
tail sample includes both incident depth and scene visibility. This remains a
quadrature approximation, not a multiple-scattering path tracer.

The screen path measures every remaining crossing for this forward source.
No quality setting is changed. The diagnostic reference pass, secondary cloud LUT
and embedded cloud-air irradiance retain library behavior.

The conversion to the TLAS render frame is:

```text
position = cameraPosition + numosKmYUpToWorld(sampleKm - cameraWorldPosYUpKm)
direction = normalize(atmosphereYUpToWorld(sunDirection))
```

This retains the cloud world scale, up-axis convention and moving world origin. Shadow
queries use the same opaque mask and forced-opaque treatment as NUMOS's aerial perspective.
Alpha-cutout texture transparency is therefore not sampled, and glass transmission is not
included. Geometry absent from the TLAS cannot cast a cloud shadow.

`scripts-common/numos_scene_shaders.py` generates modified sources only in the build output.
Exact replacement anchors fail on incompatible upstream changes. The build reflects both
SPIR-V modules and checks that all original bindings, workgroup dimensions and push constants
are unchanged, with only TLAS slot 30 added. Release builds use the existing host
SPIR-V optimization recipe, read from `compile_shaders.py` without executing its
command-line parser. The recipe retains the NonUniform access-chain workaround,
bindings and specialization constants. `spirv-val` and a second structural ABI
check run after optimization; only debug variable names are excluded from that
second comparison. Metadata hashes the final optimized payload. The checked-out
NUMOS submodule is unchanged. Original pass-table checks stage only `*.comp.spv`,
so host `*.comp.scene.spv` sidecars cannot contaminate the library's table.

The runtime verifies the original NUMOS pass hash and a payload hash from the companion
`.scene.meta` before loading each scene shader. Both files must be deployed beside the original
NUMOS SPIR-V. Missing or incompatible variants produce a warning and use the library shader;
views without a TLAS also use the library shader.

Runtime validation should compare the same camera, sun and weather with and without the
scene variants: a sunlit cloud, a low cloud in a building's shadow, a moving actor crossing
the sun, an elevated cloud outside the buildings, and a camera near or inside a cloud.
With `RTX_NUMOS_PASS_TIMINGS=1`, measure `cloud_render` (or `cloud_render_fast`) as well as overall GPU time. `CloudScreen` then contains only tail bookkeeping. The visibility query adds
work per nonempty sample; no performance improvement is claimed without those measurements.

## Camera composite diagnosis

The Debug View menu exposes camera-path contributions before tone mapping:

| Index | View | Interpretation |
| --- | --- | --- |
| 915 | Cloud Composite Change | Signed cloud-composite delta, including foreground and alpha coverage. |
| 916 | Aerial Perspective Composite Change | Signed camera-air delta, including foreground and alpha layers. |
| 917 | Sky Disk Composite Change | Sun/moon disk radiance before cloud and air extinction. |
| 918 | Sky Surface Classification | Red: sky miss; green: alpha-blend coverage; blue: foreground; white: PSR. Opaque and alpha-tested hits are black. |
| 919 | Before Sky Composite | Surface/shared/alpha radiance before the three camera sky operations. Native volumetrics and path-traced sky are included. |

The radiance views use the composite's pre-exposed units. Use the existing Debug View
min/max or GPU print controls to compare values. Negative deltas represent extinction
and require a negative display minimum. These are deltas at their respective stage
boundaries, not three independent additive terms of the final image: later extinction
also acts on earlier terms.

For live isolation, `cloudRenderRTEnable=false` removes the screen cloud composite
while retaining the cloud shadow map and reflection dome. `debugDispatchCloudRender=false`
only freezes the cloud buffer and cannot isolate its visual contribution.
`aerialPerspective=false` independently removes camera-air composition. Freeze camera,
sun and weather while comparing. Neither shader-load success nor a compiled binary
proves that a reported bright silhouette is fixed.

## One-frame cloud hit probe

The internal `NumosCloudProbe` observer is off by default. It requires ABI-1
instrumented `cloud_render.comp.scene-probe.spv` and
`cloud_render_fast.comp.scene-probe.spv` sidecars, with matching `.scene-probe.meta`
files, beside the normal scene variants. These diagnostic sidecars are separate
from the production shader build. Missing or stale probe payloads retain normal
scene visibility; setting an environment variable alone does not instrument it.

Set these variables before starting the renderer:

| Variable | Meaning |
| --- | --- |
| `RTX_NUMOS_CLOUD_PROBE=1` | Enable the observer for this renderer instance. |
| `RTX_NUMOS_CLOUD_PROBE_ARM_FILE` | Required absolute marker-file path. It must first be observed absent, then appear as a file. |
| `RTX_NUMOS_CLOUD_PROBE_DIRECTORY` | Required existing absolute output directory. Use a fresh directory for each run. |
| `RTX_NUMOS_CLOUD_PROBE_PIXELS` | Optional one to eight normalized positions, `u,v;u,v`; each component must be in `[0,1)`. Eight positions are selected by default. |

After the camera, weather and streamed scene settle, create the marker. The host
checks it approximately every 250 ms and leaves it unchanged. It captures one
eligible cloud frame, with no rearming in that instance. Positions map to the
actual cloud-output dimensions and the current interleaved march phase, rather
than directly to screenshot pixels. Repeated dispatches in that frame share the
same buffer. Readback is asynchronous, waits at least eight later frames and checks
GPU query/resource readiness; it does not call `waitIdle`. Exiting before completion
can drop a pending result, as reported in the log.

Outputs are `cloud-probe-frame-<deviceFrame>.bin` (33,024-byte ABI-1 query data),
`.constants.bin` (GPU atmosphere/camera constants), and `.metadata.json` (copied
same-frame CPU surface identities). `[NumosCloudProbe]` log records include the
cloud/device frame IDs, CPU/GPU header and constant comparisons, claimed pixels,
hit/miss totals and resolved surface slots. Each pixel retains the first 30 queries
plus its first hit and first miss; these extra records can duplicate the prefix.
`beyondPrefix` is the count beyond the first 30, not the number of lost records.

Require matching constants and a reproduced scene before interpreting results.
An unclaimed pixel is missing evidence, not a miss; duplicate snapped positions
can leave a slot unclaimed. An all-miss capture of an incompletely streamed scene
does not rule out a scene occluder. CPU metadata is capped at 65,536 surface slots
and is not GPU-transform verification; procedural grass and GPU-owned instance
slots may resolve only to a prototype. Capture/readback frames are unsuitable for
normal performance measurements.
