# Nearest sphere candidates

`rtx.di.localSphereInitialSampleCount` adds up to 64 nearby sphere-light candidates
to RTXDI initial sampling. Zero disables it. Custom graphics presets retain the authored
lighting value; the automatic quality presets reset it to zero. It is used
only with `rtx.lightSampling.mode = 0`; it is not an equivalent of light-BVH sampling.
The previous local game's explicit value of 32 must be migrated with that mode.

The compute selector reads the existing active light GPU buffer. Each 256-thread
group sorts 512 candidates by clamped squared camera distance and then light index.
Groups retain their best 64, and further GPU passes merge those lists until one
remains. Non-sphere or nonfinite positions are rejected. The header and final count
stay on the GPU; initial and temporal sampling consume the same count and advance
their random streams by the same number of samples. No light sorting, distance
calculation, or result readback is performed on the CPU.

Disabled shader variants omit the candidate loop and dispatch no selector. The
enabled path preserves the v2 active-pixel launch, portals, and independent wetness
variants. Temporary candidates use two device-local buffers that grow on demand.
The packed light contract is position in `data0.xyz`, type in `data3.w[31:29]`.

The standalone Vulkan validation compares actual shader outputs with a CPU test
oracle across 220 cases, including zero/65534 inputs, count clamping, ties,
non-spheres, camera changes, and NaNs. Runtime integration and game performance
acceptance are recorded separately; this is not a measured game FPS improvement.

## Estimator overlap

The regular RTXDI estimator already covers every direct light, including the selected nearest spheres. The nearest list draws one area sample per distinct sphere and estimates their sum, normalized by one rather than by a count of surviving samples. Both estimators receive weight one-half on the shared direct-light domain. The regular estimator keeps weight one outside that domain, including portal images of the same light. This partition of unity avoids double lighting and does not drop a sphere merely because its portal image was the previous best light. The existing best-light estimator remains part of the regular estimator.

Nearest sampling consumes three randoms per listed light and one combine random, matching the temporal pass even for zero-target samples. Selection, light samples, membership checks and reservoir weights are computed on the GPU. No readback or CPU per-light sort is used.

Automatic Ultra/High/Medium/Low presets set the additional candidate count to zero; Custom retains the authored count. This avoids enabling additional sample cost implicitly. Actual performance comparisons must retain the game's authored count and quality settings.
