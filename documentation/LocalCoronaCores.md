# Local corona cores

Status: experimental. Target-game route and switch checks are recorded below. They do not establish all-scene continuity, GPU fence completion, or a performance improvement.

## Controls

The post-processing list contains **Local Corona Cores**, a display-space filter with its own enabled checkbox, brightness, size, and submission status. It can be reordered among other display-space filters. Its default position is before Lens Rain, including migration of saved stacks that omit the new filter, so lens drops can refract cores. It remains reorderable within the display domain. The master PostFX and filter switches hide v2 core submissions, without restoring host IM3D geometry. Disabling after a frame has been accepted takes effect on the next submitted frame.

The filter is disabled by default (`rtx.localCoronas.localCoronasEnable=False`). Store this developer option in `rtx.conf`; a stronger `user.conf` override blocks direct edits until that individual option is unblocked. Brightness and size default to 1 and are preset-independent because the host supplies the original photometric and size parameters. These controls are independent of local starbursts, pickup/checkpoint sprites, emitting fixtures, and actual RTX lights.

The VC host has a separate display switch and `RTXLights.CoronaBoard.RenderBackend`: 0=IM3D, 1=Remix mesh, 2=RTX PostFX filter. If the new key is absent, it resolves the old CorePostFx/Backend values. `CorePostFxFallback=1` preserves the existing IM3D fallback on protocol rejection; set 0 for strict RTX mode. Explicitly disabling the runtime filter is a successful suppression, not a failure, so neither fallback setting restores the cores. Old keys remain readable for migration but the new backend key takes precedence once saved.

## Image and ownership

The host preserves lamp selection, near/far fade, distance growth, world half-size, color, and frustum checks. No range is reduced to fit the budget. The analytic radial profile is `(exp2(-falloff*r*r)-exp2(-falloff))/(1-exp2(-falloff))`, with falloff 1..32. It is not an exact reproduction of arbitrary DDS artwork.

Accepted cores contribute only to the screen image, not reflection/GI geometry. Physical emitting fixtures and RTX illumination remain independent. Material-aware center visibility includes alpha-tested geometry and translucent attenuation, with the existing verified emissive fixture endpoint rule. This is center visibility, not per-pixel foreground masking of the full halo.

## Protocol

The shared `remix_local_coronas.h` uses existing synchronous GetGameValue/SetGameValue transport; exported bridge ABI is unchanged.

* v1 remains available: `__remix.localCoronas.runtimeVersion` returns `1`; `__remix.localCoronas.submit_v1` accepts LC01 and at most 512 records. Its legacy passive behavior is retained.
* v2 capability `__remix.localCoronas.runtimeVersion2` returns `2`. Only after this exact handshake may a host send `__remix.localCoronas.submit_v2`: LC02, eight hexadecimal count digits, then up to **4096** little-endian 48-byte records encoded in hexadecimal. Maximum text length is 393228 bytes plus NUL. Old runtimes cannot be mistaken for v2 merely because they store arbitrary string keys.
* Records contain XYZ, positive half-size, RGB, opacity, falloff, kind 0..3, and unique nonzero 64-bit source ID. Nonfinite, out-of-range, malformed, duplicate, and oversized input rejects the whole batch. There is one producer and one complete snapshot per runtime frame. This is not a paging API.
* For v2, a disabled filter or master PostFX accepts ownership and records **suppressed**. Otherwise missing runtime/RTX/shaders rejects the batch, allowing the host's explicit fallback policy.
* Accepted state is established on the ordered CPU command stream. The API drains preceding CPU commands before replying, without waiting for GPU completion. A first submission after the frame's corona composition opportunity is rejected. Repeated same-frame submissions acknowledge existing ownership without replacing or drawing the batch again, including after dispatch. Explicit filter suppression takes precedence over both late and duplicate checks. The batch retains its frame/scene identity and is consumed once. Scene/frame changes discard it as stale.
* `__remix.localCoronas.status_v2` returns three decimal values: state, runtime frame, source count. State is 0 idle, 1 accepted, 2 dispatch recorded, 3 suppressed, 4 stale, 5 late rejection, 6 duplicate suppressed (ownership already accepted), 7 rejected because RTX or shaders were unavailable. It is diagnostic history, not a fence or proof of current visible pixels. Both menus explicitly label dispatch as distinct from GPU completion.

The synchronous acceptance closes the previous known late-submit acknowledgement gap. Full RTX renderer early-outs, invalid camera, scene clears, and loss of a post-processing opportunity after acceptance still cannot be restored by the host in that frame. There is no world-space late-present fallback or GPU completion guarantee. Keep IM3D for strict continuity until tested on the target scene.

## Cost and validation

The maximum matches the VC owner's 4096-board capacity, eliminating the 511/512/513 discontinuity. The 16x16 tile compositor uses a 4096-bit membership mask (512 bytes group shared memory), with every admitted source addressable. Each tile scans the complete source batch and each pixel evaluates only members of its tile. Worst-case overlapping sprites can still evaluate 4096 profiles per pixel. GPU source/sprite buffers total 327680 bytes, allocated lazily. One encoded maximum frame carries 393229 bytes including NUL; this is a bounded capacity increase, not a measured speedup.

`tests/rtx/unit/test_local_coronas.cpp` covers x86/x64 ABI and roundtrips at 0/1/511/512/513/1032/4095/4096, whole-batch corrupt/duplicate/oversize rejection, v1 compatibility and complete tile-mask indexing. It does not emulate Vulkan execution. Runtime shader/full-link results and game acceptance must be reported separately. Required game checks include dense fog scenes, 4096 sources, moving camera, alpha/glass occlusion, filter and RTX toggles, scene reload, zero sources, first shader compilation, reflection comparison, and GPU timing.

## Target-game check, 2026-11-08

The final x64 runtime was fully linked and checked for missing imports (none), with 1060 generated shader headers verified. Runtime SHA256: `527bfaddc440963e5279a107c90c83291d8de1bdd9978809130b798e2c456931`. The x86 host and protocol tests were built separately.

On the installed Vice City scene (RTX 5070, 1114x626 to 1920x1080 DLSS-RR, 4x frame generation), both 150-second replays completed with identical route/vehicle/time/weather and valid base-frame telemetry. IM3D measured 25.297 base FPS / 39.531 ms, and the filter 25.634 / 39.010 ms. This single pair includes dynamic traffic, clouds, streaming and cache differences; it is not evidence of a speedup. GPU sampled whole-frame means across the four complete logging windows were 38.2/33.7/33.8/35.2 ms and 38.7/34.4/33.0/33.2 ms respectively. The whole PostProcessing stage p50 ranged 0.31-0.33 ms versus 0.37-0.52 ms; this does not isolate the corona pass.

The filter run produced 43 bounded host diagnostic samples: all accepted, zero overflow, 35-2514 candidates, with 28 samples above 512. A current-frame dispatch was logged. These are diagnostic samples and CPU dispatch evidence, not all-frame GPU fences. The menu showed the independent filter before Lens Rain. Switching the filter or master PostFX off showed suppressed status and hid its cores; switching back restored them. Fresh-process scene loads and moving-camera playback succeeded. Full 4096-source GPU stress, material-by-material occlusion/reflectance comparisons, and accepted-frame renderer early-out continuity remain unproven.
