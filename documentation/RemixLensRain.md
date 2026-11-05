# Lens rain

Enable Lens Rain in the post-processing stack. It runs after tonemapping in the
display domain and can be reordered with other display effects. Existing saved
stacks gain it immediately before Lens Effects. Its master and Respect Cullzones /
Interiors switches default off. The old `rtx.lensRain.*` option names remain valid.

Weather-driven coverage uses the typed NUMOS rain/storm weights multiplied by
precipitation intensity. Force Amount supplies an explicit preview instead.
Respect Cullzones / Interiors stops spawning in host-published no-rain volumes or
when `__weather.precipitation_enable` is zero. Existing drops fade using Dry Time.
These volumes do not detect arbitrary roofs or underwater cameras.

`__lensRain.motion_v1` contains `1 frame vx vy vz`, with velocity in metres per
second. Finite velocities up to 120 m/s are accepted. A frame token that stops
advancing expires after 500 ms. Once a host has provided motion, stale or invalid
packets give zero speed; the camera fallback applies only before any packet.
Camera cuts and long frame interruptions reset the smoothed velocity.

## GPU execution

The device owns a fixed 1024-slot state buffer. Three compute passes clear tile
headers, update/drop/spawn independent droplets and build tile lists atomically,
then composite the refracted scene. CPU work supplies weather, camera/host motion,
authored settings and dispatch dimensions. It does not update droplet positions,
lifetimes, visibility or tile membership, and there is no production readback.
The optical and motion formulas derive from the previous local runtime's lens
rain implementation. That implementation's per-droplet CPU loops are replaced
with GPU state and index buffers. Existing repository licence notices apply.

Each tile can contain all 1024 slots, so overlapping drops cannot overflow a tile.
Only headers need clearing. Atomic insertion order does not affect the image:
equal-opacity overlaps select the lower drop ID. The largest supported tile grid
uses about 8 MiB for indices; state and visible records add about 112 KiB.
Changing aspect ratio rescales existing horizontal positions. Changing population
initializes newly active slots without reading uninitialized memory. Disabling
Lens Rain releases its buffers; drying to zero resets the next population.

CPU layout assertions and compiled SPIR-V offsets agree on 96-byte arguments,
80-byte states and 32-byte visible records. Functional GPU tests and their exact
runtime hashes are recorded with the migration outputs. Full-game frame-rate
acceptance follows the remaining module migrations.
