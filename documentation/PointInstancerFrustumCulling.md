# Point instancer frustum culling

`rtx.pointInstancer.frustumCull` enables GPU rejection outside the four camera side
planes. It defaults off. `frustumSpherePad` expands the accepted region in world
units; `frustumNearSkip` retains nearby instances regardless of the side planes.
The culling master and host-instancer opt-in continue to apply. This setting can
remove off-screen shadows/reflections, so it is authored per scene rather than
silently enabled by quality presets. The previous local preset used true with a
600-world-unit near exemption, which must be migrated explicitly.

The CPU computes four normalized camera planes once per dispatch setup. All
per-instance distance, fade, frustum tests, transforms, TLAS writes, and surface /
material replication remain in the v2 compute pass. Its frame-wide batch table,
single-dispatch path, unchanged-input upload skipping, and SSS twin handling are
retained. Every rejected instance writes a complete TLAS entry with a zero mask.
The disabled shader variant omits the side-plane code.

Both compiled variants use 112-byte constants, 176-byte batches, 256-byte surfaces,
and 64-byte materials. On an RTX 5070, the standalone production-shader fixture
passed 336 cases and checked 150336 instance records with zero failures. It covers
Y/Z-oriented planes, near exemption, margins, host exemption, radius/density fade,
workgroup boundary sizes, SSS twins, and buffer guards. Single and multiple dispatch
outputs match byte for byte. Full-game performance acceptance remains separate.
