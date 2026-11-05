# Flare cards through standard material replacements

The host plugin draws camera-facing cards. A USD material replacement supplies the
diffuse mask, emissive mask, emission color and intensity. The runtime uses its
normal material and transparency paths; no flare identity cache or feature toggle
is needed.

Author a material named `mat_<submitted texture hash>` under `/RootNode/Looks`.
Set `enable_emission = true`, `emissive_color_constant = (1, 1, 1)`,
`use_legacy_alpha_state = false`, `blend_enabled = true`, and `blend_type = 1`
(alpha-emissive). Set both texture inputs and the desired `emissive_intensity`.

The host puts normalized lamp tint in RGB and its brightness/fade in vertex alpha.
The fixed-function alpha stage selects diffuse alpha. This retains intensity even
when `rtx.vertexColorIsBakedLighting` normalizes RGB. Use SRCALPHA/ONE blending for
the original raster renderer as well. Keep card size proportional to camera-space
depth for constant angular size.

The removed `rtx.fork.lensFlare.*` settings have no effect. Material strength belongs
in the USD file; plugin visibility and geometry controls remain in its own config.
