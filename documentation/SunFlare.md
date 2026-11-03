# Solar lens flare implementation

Sun Flare adds solar lens ghosts to the runtime's display-domain post-processing stack. It uses the active NUMOS sun rather than capturing the game's Im2D flare or submitting camera-facing Im3D meshes. It appears in the final rendered image, but is not exported as three-dimensional flare geometry in a USD scene capture. The source tables provide the 26-entry SA and 7-entry VC layouts. Layout and texture selection are independent.

This is a visual adaptation, not a pixel-exact recreation of either game's renderer. The runtime supplies sun tint, scene/cloud visibility, temporal filtering and compositing. No GPU acceptance or performance measurements are claimed here. See [usage](SunFlareUsage.md) for configuration and integration checks.

## Ownership and integration

`DxvkPostFx` owns the five texture references, visibility image, last update time and history state. Implementation is in `src/dxvk/rtx_render/rtx_postFx_sun_flare.cpp`; options and members are declared in `rtx_postFx.h`. It uses the existing device-owned `metaPostFx()` object and does not expose a new bridge API.

`RtxForkPostProcessing` registers the `Sun Flare` entry with configuration ID `sun_flare` and display color domain. Its default order is after NTSC/VHS and before Lens Rain/Lens Effects. When adding the effect to an older saved order, insertion prefers the Lens Rain position and otherwise Lens Effects. The tonemapping and terminal sRGB/dither stages remain pipeline anchors.

The feature requires both `rtx.postfx.enable` and `rtx.sunFlare.sunFlareEnable`. Its compiled defaults are opt-in: the feature is disabled and its texture directory is empty. The supplied configuration fragment enables a concrete SA-layout/VC-texture preset.

## GPU passes

The state pass, `sun_flare_state.comp.slang`, dispatches one thread and writes a 2×1 RGBA32F image:

- Texel 0 stores sun screen UV, filtered visibility and camera water depth.
- Texel 1 stores display sun tint; its fourth component encodes validity and remaining resurfacing recovery time.

The source direction is `skyProviderSunDirection()`. `skyProviderActive()` currently requires `rtx.skyMode=1`, the NUMOS provider. Projection uses the runtime camera and its handedness, then `cameraNDCToScreenUV`. A behind-camera sun is rejected. Horizon and screen-edge gates attenuate visibility; a zero horizon gate suppresses the effect immediately rather than leaving old fade history below the horizon. Ordinary geometry/cloud occlusion uses the falling visibility time constant.

Source tint is `sunIlluminance / peak` multiplied by atmospheric transmittance. It does not read VC's timecycle RGB and does not scale proportionally with the absolute solar illuminance value. A zero peak disables contribution. `sunFlareBrightness` is an artistic display-domain intensity control.

Eight visibility rays sample the solar disk, using the common visibility path with opaque, alpha-blended and translucent masks. Maximum ray distance is 10,000 world units. Optional cloud attenuation samples the secondary dome LUT when enabled, otherwise the existing sky-provider cloud shadow path. These choices avoid a dependency on the legacy game's corona fade state, but need GPU validation for the target scene and materials.

The composite pass, `sun_flare.comp.slang`, uses 8×8 groups over the final output. It reads the state and five texture samplers, adds the selected ghosts to the output RGB, and preserves output alpha. There is no CPU visibility readback and no flare geometry to capture or include in secondary rays.

## Layout, textures and blending

Each ghost is centered at `screenCenter + (sunScreen - screenCenter) * position`. Its pixel half-height is `4 * size * sizeScale`, optionally multiplied by output height/480. SA uses a square half-size. VC additionally scales half-width by output width/height; this does not reproduce every third-party widescreen sprite patch.

The table color coefficients retain their original /256 scale. The composite applies 255/256 intensity for SA and 200/256 for VC, without reproducing every original byte truncation. Optional SA flicker uses a frame-index hash shared by all ghosts, in the range 0.7–1.0. VC ignores that toggle.

| Layout | Texture set | Selected masks |
|---|---|---|
| SA / 0 | VC / 0 | VC `coronastar` for all 26 ghosts |
| SA / 0 | SA / 1 | SA `coronastar` for all 26 ghosts |
| VC / 1 | VC / 0 | VC hex, circle and ring, following the seven-entry table |
| VC / 1 | SA / 1 | SA `coronastar` for all seven ghosts |

The extracted assets are BGRA8 legacy DDS files. PNG copies are inspection references. No resizing, recoloring, transfer-function conversion or premultiplication is applied during extraction. All supplied assets decode to alpha 255; their black RGB background contributes zero under the original ONE/ONE additive convention. The shader samples RGB and must not use opaque-alpha replacement of the background. Texture loading uses the existing texture manager with `ColorSpace::AUTO`; final appearance under the selected output pipeline still needs visual calibration.

The directory must contain `vc_coronastar.dds`, `vc_coronahex.dds`, `vc_coronacircle.dds`, `vc_coronaringa.dds` and `sa_coronastar.dds`. The existing `WaterSystem::resolveAssetPath` utility resolves relative paths from the game root, including the `.trex` executable-directory case. Missing assets emit warnings. Missing individual slots use the first loaded texture; if none loads, the feature does not dispatch. The UI reload button clears the directory cache for the next dispatch.

## Water classification

Water suppression reads camera immersion independently of whether underwater absorption, fog or light shafts are enabled. `rtx.water.cameraMediumMaterialHash` identifies the water camera medium published through the existing integration. The CPU compares that parsed source hash with `SceneManager::getStartInMediumMaterialSourceHash()` and passes a typed state:

- 0: no parsed configured water identity; permit a geometric fallback.
- 1: a configured identity exists but does not match the camera medium.
- 2: the configured water identity matches the current camera medium.

For unknown classification, a GPU upward ray queries captured triangles marked `isAnimatedWater` within 100 world units. Matching water commits the intersection. This is a fallback surface query, not a general closed-volume solver: missing water geometry, multiple surfaces or unusual underground layouts can defeat it. An explicit configured mismatch does not run that fallback. Existing `cameraWaterVolumeActive` shader state can also identify immersion.

With `sunFlareHideUnderwater`, positive immersion forces visibility to zero immediately, overriding previous temporal history. Therefore turning off water fog alone does not restore flare if the camera water identity is still published. On resurfacing, the effect uses at least `sunFlareWaterTransition` as its rising visibility time constant during the recovery window.

Do not substitute an arbitrary water texture hash for the camera medium source identity. Preserve the target game's existing integration value. The supplied fragment intentionally does not assign it.

## History and boundaries

Visibility uses exponential time-based smoothing with separate rise/fall time constants. History resets after a camera cut, a gap over 250 ms, backward time, feature disablement or texture-directory reload. Disabling the feature invalidates history; state is not persisted through configuration options. Sun position behind the camera and inactive NUMOS state suppress contribution rather than retaining an old flare.

The optional VC 1.0 EN companion `SunFlareSuppressVC.asi` only clears solar corona ID 2's `flareType` during the original corona render CALL and restores it afterwards. It renders nothing and preserves the game's corona state maintenance. Do not install the earlier Im3D flare plugin alongside it. Solar disks, corona cores and water reflections are outside this feature.

## Validation scope

Asset extraction reopens PNG and DDS outputs and byte-compares decoded RGBA against the source. Source-table checks and CPU reference projection/temporal checks provide development evidence, not execution of the Slang kernels. Runtime build status belongs in the build report. Required GPU acceptance includes all four layout/texture combinations, geometry/cloud occlusion, camera cuts, surface crossings with water fog disabled, output resolution/HDR, resource reload and performance measurement.
