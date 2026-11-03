# Sun Flare usage

Use a runtime build that includes the `Sun Flare` post-processing entry. This feature follows the NUMOS sun and requires the existing NUMOS sky mode (`rtx.skyMode=1`). It does not automatically replace your sky settings. The feature is disabled by default in code and has no default texture directory.

The effect is composed directly into the final image. It does not become a three-dimensional flare object in a USD scene export.

No GPU gameplay acceptance or measured performance claim accompanies this usage document. Shader/source/CPU checks are separate from actual runtime execution.

The complete delivery includes the feature's runtime DLL for the supplied current VC runtime baseline. It is not a fresh Remix installation; retain the matching support DLLs. Consult the build report for the actual build and acceptance status.

## Install the assets and configuration

1. Place all five DDS files under the game's `sunflare/textures` directory: `vc_coronastar.dds`, `vc_coronahex.dds`, `vc_coronacircle.dds`, `vc_coronaringa.dds`, `sa_coronastar.dds`.
2. Merge the provided `sunflare.rtx.conf` fragment into the active `rtx.conf`. Do not replace the entire configuration. The fragment filename is not loaded automatically.
3. Ensure the global post-processing toggle is enabled. In the post-processing stack, enable `Sun Flare`.
4. For VC 1.0 EN, install `SunFlareSuppressVC.asi` with its adjacent INI set to `Enabled=1`, unless another known mechanism already suppresses the native solar ghosts. Remove or disable `SunFlareIm3D.asi`; the two approaches must not run together.

The suppressor ASI provides no replacement drawing by itself. If disabling the runtime effect to restore the stock look, also disable the suppressor and restart the game. Its INI is read at startup.

Relative texture paths are resolved from the game root. Absolute directories are supported. Use `Reload flare textures` after replacing files. Install all five textures: partial loading falls back to the first available texture and changes the requested appearance; an empty directory or total load failure disables rendering. Check runtime warnings for `[SunFlare] Texture missing`.

## Choose the effect independently of its texture

The UI's `Effect layout` maps `Dense (26 ghosts)` to SA and `Classic (7 ghosts)` to VC. `Texture set` maps `Mixed shapes` to the VC texture source and `Soft star` to the SA `coronastar`.

| `sunFlareStyle` | `sunFlareTextureSet` | Result |
|---:|---:|---|
| 0 | 0 | SA layout with VC `coronastar`; supplied preset |
| 0 | 1 | SA layout with SA `coronastar` |
| 1 | 0 | VC layout with VC hex/circle/ring |
| 1 | 1 | VC layout with SA `coronastar` |

Every option below has the `rtx.sunFlare.` prefix. The repeated `sunFlare` in the option name is intentional; for example, the full enable key is `rtx.sunFlare.sunFlareEnable`.

| Option | Compiled default | Meaning |
|---|---|---|
| `sunFlareEnable` | False | Enable this effect; also requires `rtx.postfx.enable` |
| `sunFlareStyle` | 0 | 0 SA layout, 1 VC layout |
| `sunFlareTextureSet` | 0 | 0 VC textures, 1 SA star texture |
| `sunFlareTextureDirectory` | Empty | Directory containing the five DDS files |
| `sunFlareBrightness` | 1.0 | Display-domain intensity multiplier, 0–20 |
| `sunFlareSize` | 1.0 | Ghost half-size multiplier, 0.05–8 |
| `sunFlareResolutionScale` | False | Scale original pixel sizes by output height/480 |
| `sunFlareFlicker` | False | SA-only shared frame variation, 0.7–1.0 |
| `sunFlareHideUnderwater` | False | Hide when the camera is in water |
| `sunFlareCloudOcclusion` | False | Apply existing runtime cloud transmittance |
| `sunFlareFadeIn` | 0.15 | Rising visibility time constant in seconds, 0.001–5 |
| `sunFlareFadeOut` | 0.10 | Falling visibility time constant in seconds, 0.001–5 |
| `sunFlareWaterTransition` | 0.15 | Minimum resurfacing rise time constant, 0.001–2 |

The provided fragment explicitly enables the feature, resolution scaling, underwater suppression and cloud occlusion. It leaves flicker disabled and selects SA layout/VC textures. These preset choices are not changes to the compiled defaults. Fade values are exponential time constants, not exact durations after which visibility reaches its target.

Sun tint uses solar illuminance normalized by its peak, followed by atmospheric transmittance. Absolute solar illuminance changes therefore do not proportionally scale flare brightness; a zero peak disables it. Use `sunFlareBrightness` to adjust the artistic intensity. A sun below the horizon is suppressed immediately; ordinary building/cloud occlusion uses fade-out.

`Sun Flare` belongs after tonemapping and before terminal sRGB/dither. The default stack position is after NTSC/VHS and before Lens Rain/Lens Effects. It can be reordered within the display color domain. The configuration fragment does not replace a saved stack order. Old Im3D-plugin hotkeys do not operate this runtime effect.

## Hide flare underwater

Enable `rtx.sunFlare.sunFlareHideUnderwater`. Keep `rtx.water.cameraMediumMaterialHash` consistent with the source identity published by the game's water camera-medium integration. This is not an arbitrary visible water texture hash. The target VC configuration inspected for this work uses `0x574154455243414D`; the fragment deliberately leaves the existing value untouched.

A matching configured camera medium hides flare immediately. A configured mismatch is treated as outside that medium. If no usable identity is configured, the shader falls back to an upward ray against captured `AnimatedWater` geometry, up to 100 world units. This surface-based fallback needs correctly classified water geometry and does not solve all enclosed or layered water volumes.

Hiding is independent of underwater fog/absorption/light-shaft toggles. If the water camera identity remains published, disabling water fog should still leave flare hidden. On exit, visibility recovers with the configured rise and resurfacing time constants. Classification uses the camera, so a character being in water does not imply that an above-water camera should hide flare.

## Acceptance checklist

- Compare all four effect/texture combinations while facing the sun.
- Test buildings, vegetation, clouds, screen edges, the horizon and looking away from the sun.
- Enter water with the camera; confirm no residual flare, including after disabling underwater fog.
- Resurface and check recovery. Test rapid crossings, camera cuts and teleportation.
- Test output resolution/FOV/HDR, texture reload and the target plugin set.
- Measure GPU cost before making a performance claim.

If no flare is visible, first check the runtime version, NUMOS mode, both enable toggles, asset paths, sun visibility and camera medium classification. The feature affects lens ghosts only; it does not replace the solar disk, corona core or water reflection.
