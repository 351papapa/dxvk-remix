# Native screen droplets

The optional `__lensDrops.*` convention lets a game adapter replace its existing
screen water and blood droplets with RTX lens refraction. It describes a complete
current-frame snapshot. It never spawns gameplay particles and is independent of
weather rain. No exported function, interface table, or binary struct changes.

Both kinds default off. Post FX > Lens Rain provides **Game Water Droplets**,
**Game Blood Droplets**, **Game Drop Size** (0.25–3, default 1), and **Blood
Absorption** (0–4, default 1). The options are
`rtx.lensRain.gameWaterDropsEnable`, `gameBloodDropsEnable`, `gameDropSizeScale`,
and `bloodAbsorption`. Existing rain intensity and shelter controls affect weather
droplets only. Turning off weather Lens Rain does not disable accepted native drops.
The resolved camera water medium suppresses all lens droplets, including native
water and blood. A submitted underwater frame is consumed without drawing, so
late composition cannot revive it. After leaving water, only a newly accepted
current native snapshot can appear; no underwater snapshot is replayed.
The normal injection records its resolved medium for this device frame before
scene cleanup. Empty scenes and frames that skip preparation instead read the
current valid typed external-medium request, with the same cache bounds check as
scene preparation. A same-frame resolved result has priority over later external
changes. No previous-frame camera decision or water-height approximation is used.

## Capability and transactional handoff

Call `GetGameValue("__lensDrops.capability_v1", ...)` before suppressing any native
draw. A supported registered runtime returns this ASCII value:

```text
1 enabledKinds renderFrame deviceToken
```

`enabledKinds` is a bit mask: water = 1, blood = 2. It is zero if Post FX or ray
tracing is disabled, or asynchronous shader compilation currently bypasses RTX. `renderFrame` is the runtime
frame observed by the API, not the game's simulation frame. `deviceToken` is an
opaque device-lifetime token for invalidating producer-side duplicate caches.
Unknown runtimes may accept arbitrary GameValue strings; ordinary storage SUCCESS
does not establish support for this convention.

For a nonempty native frame, use the corresponding
`GetGameValue("__lensDrops.orderedCapability_v1", ...)` before deduplication or
submission. It has the same fields but places a command in the producer's command
stream and waits for preceding CPU command-stream work, without waiting for GPU
completion. This prevents a queued Present from exposing an old frame number.
This synchronization has a real CPU cost. Use the inexpensive capability key for
setup, and perform no ordered query in stable empty/default-off frames.

Submit one full batch with `SetGameValue("__lensDrops.frame_v1", wire)`. This exact
key is handled by the runtime API instead of generic GameStateStore storage.
Submission synchronously validates every record, the requested-kind mask, current
options, shader readiness and strictly increasing sequence number within a producer
epoch. Only SUCCESS transfers drawing ownership. On any failure the producer must
draw the entire submitted native set. If only water is enabled, submit water and
keep native blood drawing; do not hide a kind excluded from the submitted mask.

A success queues an owned snapshot in the same command stream as the game's draw
commands. At execution it is stamped with the actual runtime frame and scene epoch.
The accepted frame is drawn once even if a user turns off Post FX or a kind later
in that frame. A changed option rejects subsequent submissions. An old snapshot
never carries forward to a later present or a new device object. Normal composition
also checks the scene epoch. A scene synchronization that advances the epoch during
this same present can still finish the accepted pure screen-space snapshot in the
late path. Malformed
or duplicate/descending submissions are rejected and do not revoke an already
accepted frame; frame and scene guards prevent retaining it for a later present.
If normal composition is skipped or submission arrives after the stack, the actual
D3D9 scene-present target is used at `RtxContext::endFrame`, after fallback RTX
injection. A pending, unconsumed current-frame snapshot is composed once; an already
consumed or absent snapshot adds no GPU dispatch. This does not use old RTX output.
Device loss or a failed Present cannot be made into a visible drawing guarantee.

The late path copies the current color target (resolving MSAA first), shades into
an RGBA16F scratch image, and blits back. A 32-bit floating target uses RGBA32F to
preserve its range. MSAA writeback uses DXVK's existing framebuffer blit path,
with the destination's sample count. Scratch resources are reused while this path
is active and released when no late snapshot is needed. Target dimensions, sample
count, usage and color format are checked; native D3D9 scene backbuffers supply
the required transfer/sample/color-attachment capabilities.

Color encoding describes the sampled values. Normal RTX blits record their actual
gamma-conversion flag for this target/frame; native UNORM and HDR-float targets
normally contain gamma-2.2 values, while native true-SRGB sampling already yields
linear values. The filter decodes/encodes gamma 2.2 only when required, retains HDR
values above one, and never applies an extra paper-white scale. Hardware SRGB
sampling and final blit conversion remain Vulkan's responsibility. Late composition
occurs after the game's HUD/menu/fade drawing but before the Remix GUI; normal
composition keeps the configured lens-pass location. This ordering difference is
intentional and must be included in visual acceptance.

## Snapshot encoding

Numbers are whitespace-separated ASCII with a decimal dot. Header:

```text
1 epoch sequence gameFrame gameTimeMs paused width height kinds count
```

`epoch` and `sequence` are nonzero unsigned 64-bit values. Other integers are
unsigned 32-bit. `paused` is 0 or 1; `width` and `height` are 1–32768; `kinds` is
1–3. `count` is 0–1024, and the entire value is limited to 192 KiB. The game frame,
time, and paused fields describe the native simulation; they do not schedule RTX
lifetimes. A producer changes epoch on its simulation reset or device change.

Each record has ten fields:

```text
id kind u v radiusX radiusY axisX axisY opacity variant
```

IDs must be unique within the snapshot. Kind is water = 1 or blood = 2 and must be
included in `kinds`. Coordinates use normalized top-left screen UV; the accepted
offscreen guard range is −4 through 5. Both radii are fractions of screen height,
strictly positive and at most 1. The axis is a finite direction of length 0.5–1.5
and is normalized by the runtime. Opacity is 0–1; variant is 0 or 1. Nonfinite
fields, malformed/trailing tokens, duplicate IDs, and partial batches are rejected.

The renderer stores up to 1024 weather and 1024 native slots. Tile lists have room
for every active slot, including all slots overlapping one tile; there is no
overflow prefix truncation. On first native use the buffers grow to full external
capacity and subsequently reuse that allocation. Stable empty native frames need no API
submission, serialization, or additional GPU pass. Native refraction shares the
existing lens pass; blood adds thickness-dependent color absorption. Overlap uses
the existing strongest-coverage selection rule, not a fluid-merging simulation.

## Vice City adapter

The adapter reads complete active native particle lists for types 40 and 41 at
the existing verified CParticle::Render hook. Native allocation, trigger conditions,
positions, opacity, frame variant, destruction, and paused lifetime remain under
game control. This covers existing water-drop entry points, including applicable
vehicle and boat splashes, without synthesizing splash events from rain intensity.
Blood drops depend on actual native blood-drop creation; ordinary damage or every
kill is not converted into a new screen-blood event.

The plugin's existing particle settings page exposes **RTX screen water droplets**
and **RTX screen blood droplets**. The logical configuration is
`ParticleEffects.ini`, section `ScreenDrops`, keys `WaterEnabled` and `BloodEnabled`,
both default zero; the unified configuration layer resolves their physical location.
Saving these controls updates the corresponding runtime booleans once. Subsequent
runtime UI changes are respected. Unsupported hooks or runtimes retain native drawing.
Only a successfully accepted snapshot temporarily detaches the relevant native draw
lists; the original list heads are restored even on a structured exception.

The new optical profile approximates the native screen size and uses deterministic
smooth tapered lenses, rather than reproducing the native framebuffer texture and
random stretch. Native HUD/radar overlap exclusions are not reimplemented. Final
size, appearance, actual splash/blood trigger delivery, and frame cost require
in-game acceptance; source and shader checks alone do not establish those results.

## Bounded timing diagnostics

Set `RTX_SCREEN_LENS_DROPS_TIMINGS=1` before launch to log at most 64 device-local
records for `OrderedGet` and successful `Submit`. Values are CPU microseconds and
submitted counts. OrderedGet includes CPU command-stream synchronization; Submit
includes validation/reservation, not GPU execution or client-side IPC. The diagnostic
is off by default and its counter is atomic. Measure real active-event frames in
addition to isolated one-drop/750-drop serializer/parser tests; do not call the
ordered handoff zero cost or infer its latency from parser timing alone.
