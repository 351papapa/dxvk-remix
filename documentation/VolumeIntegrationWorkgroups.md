# Froxel integration workgroups

`rtx.volumetrics.integrateWorkgroup8x8` selects an 8x8 compute group for volume
integration. `integrateWorkgroup8x4` takes priority and selects 8x4. Both default
off, leaving 16x8. These are scheduling choices independent of quality presets;
they do not reduce samples, froxel resolution, history, or shading quality.

Each shader variant calls the same v2 integrator with the same global dispatch
coordinate. The bounds check, portal-volume packing, random seed, sample count,
history reprojection, and NUMOS radiometry are unchanged. The CPU dispatch dimensions
use the selected group dimensions. Wetness has a separate variant for each size.
The previous local preset explicitly enabled 8x8; retain that setting in migration.
Performance depends on GPU and scene, and requires an equal-quality in-game comparison.
