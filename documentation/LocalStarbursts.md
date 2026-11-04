# Local starbursts

A host with visible lamp identities can submit cosmetic star sources before the frame's RTX injection. Select **Local Starbursts** under the display post-processing effects. The master switch defaults off. Set **Star mask texture** to a game-relative or absolute texture path. RGB supplies the star shape. The display pass preserves output alpha.

## Host contract

Resolve `remixapi_SubmitLocalStarbursts` from the D3D9 runtime or 32-bit bridge client. The optional export and types live in `public/include/remix/remix_local_starbursts.h`. Version 1 takes the version, 48-byte record stride, unsigned count and pointer. The limit is 256. Empty accepts a null pointer. An initialized bridge checks its server and registered runtime on the first successful submission. Subsequent batches travel asynchronously on the device queue. A successful return acknowledges queue admission and ownership of these optics even when their display switch is off.

Submit one complete list per game frame. An empty snapshot clears the list. Record pointers need only remain valid through the call. The current device and scene epoch own the snapshot. A new submission replaces it. A skipped frame, recreated device or cleared scene cannot inherit the previous stars. Native callers receive an error for malformed batches. The bridge validates before queue admission and the runtime verifies the copied records. A host with an older installation can keep drawing its own star cards when the export or capability is absent.

Positions use the current scene's world coordinates. Give each source a nonzero unique 64-bit ID and a kind from 0 through 3. Supply normalized display RGB and opacity including source-specific direction, distance and edge attenuation. Half width is relative to half the display height and may range from above zero through four. Aspect is width over height in [1/16,16]. Submit main-view visible centers. The renderer reprojects them with the active camera.

## Visibility and compositing

Each selected source sends one finite visibility ray from the camera, including opaque, alpha-blended and translucent materials. Endpoint bias is in metres. A wall beyond the lamp cannot block the star. Visible RGB is multiplied by the mask, host opacity and display brightness. A tile shares its candidate list and overlaps are accumulated in stable packet order. A manually reduced source budget selects the highest opacity-weighted luminance first with source ID as the tie breaker. The default accepts all 256, independently of the lighting preset.

The host remains responsible for the lamp definitions and ordinary coronas. The pass consumes neither the sun's direction nor its ghost layout. Existing base lamp and solar effects have separate controls. No mesh or material is created by submitting the optics packet.

A successful draw emits `[Starbursts] Native local starburst pass dispatched` in the runtime log. A missing mask emits its path. The effect renders into the displayed frame and does not appear as a mesh in toolkit captures.
