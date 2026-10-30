# GPU water migration

This opt-in subsystem ports the local runtime's ocean deformation, interaction
waves, hull pressure waves, wake flow, shore foam, caustics and underwater volume
passes to the v2 API-neutral scene. It uses the existing NUMOS sky provider.

## Compute ownership

The host supplies water coverage, initial state, time and interaction events over
the pointer-free private water protocol. The runtime owns device-local simulation
buffers. Reused packet inputs are uploaded in the graphics execution queue,
ordered immediately before their consuming dispatches, including multiple packets
received in one frame. Wave stepping, flow transport and pressure projection, hull deformation,
surface normals, foam placement and triangle emission run as compute shaders.
Ray queries perform shore/contact detection and light visibility on the GPU.
Scene foam directly supplies the generated GPU vertex buffers to the BLAS path;
there is no CPU geometry reconstruction. Other generated water buffers use the
existing GPU copy path. Diagnostic readbacks are optional and default off.

The protocol keeps its legacy names and layouts for the existing producer:
`water_controls.h` v3 (124 bytes), `water_gpu_surface.h` v1 (127484 bytes), and
`water_gpu_simulation.h` v3 (`Packet` 216800 bytes, `Shared` 4552504 bytes).
The generated ocean/interaction protocol is Z-up. Authored AnimatedWater surfaces
can use either supported world-up convention for foam, caustics and underwater
effects. No C API struct or interface-table layout changes are required.

## Configuration and assets

`rtx.water.enabled`, `gpuSurface`, `gpuSimulation`, all optional booleans and
diagnostics default off. Every setting has a Water menu control. Enable only the
desired features in the host's authored configuration. Previous runtime defaults
must be copied explicitly when preserving an existing game's look.

`waterMapOrigin` and `waterMapCellSize` describe the producer's fixed 128 by 128
water-level map. World XY coordinates are sampled in the shader; game map offsets
are authored configuration, not runtime constants. Both fields must match the
producer. The surface protocol uses independent coarse/fine grid parameters.

Set DDS paths through `foamColorTexture`, `foamNormalTexture`,
`foamCompositeTexture` and `causticsTexture`. `beachMaskPath` supplies the optional
eligibility mask. Paths resolve relative to the host executable; a `.trex`
bridge executable resolves relative to its parent game folder. Absolute paths
are supported. Asset changes reset their owned material/geometry state.

`foamLineMaterialPath` displays shore foam as scene geometry, with normal material
lighting and reflections. Otherwise the optional screen composite displays the
same GPU density. Simulation remains active for shore wetness even when this
display is disabled. The former fixed diagnostic sheet and height-toggle options
are not part of the migrated subsystem.

## Frame integration

NUMOS weather is resolved before water submits generated geometry. Generated
meshes retain their topology identity; a changed domain invalidates previous
vertex correspondence. Previous-frame density, world origin, clock and height
anchor are captured together for the next foam geometry dispatch.

Caustics operate around path integration. Foam simulation/composite and
underwater fog/light shafts run after upscaling, before HDR post-processing.
Sky-provider values are converted to the renderer's current pre-exposure once;
already encoded scene-light radiance is not scaled a second time. Shore wetness
uses the optional wetness shader variants even when rain wetness is disabled.

NUMOS presets accept `weather.oceanRainWeight` and `weather.oceanStormWeight`.
Weights are finite, clamped and normalized when their sum exceeds one. The clear
weight is the remainder. Ocean gain is `clear + rain * rainWaveMultiplier +
storm * stormWaveMultiplier`; explicit game blends and interrupted runtime
transitions preserve this typed state without per-frame option writes. Saving a
look preserves its ocean weights; tuning the sky holds the current ocean state.

## Camera medium bridge

The matching x86 bridge now exposes `SetCameraMediumMaterial`. It sends a 32-bit
material UID, resolves it on the x64 server, and returns the API result through a
response transaction. Null medium clears the override. Invalid structure types,
unsupported extension chains and unknown handles are rejected. The command is
appended so existing bridge command IDs remain unchanged. Deploy client and
server together.

RemixCoreVC's integrated WaterLab producer restores the GPU handshake. For live
water it waits while the GPU surface or interaction solver is unavailable instead
of invoking the old CPU fallback. GPU event mode leaves CPU height/density arrays
unchanged. Its historical explicit CPU snapshot diagnostic is not used by the
live migration or performance validation.

## Validation

Unit coverage includes foam argument completeness/layout, production sheet
bounds, partial-triangle boundary rules and weather interpolation. The boundary
unit test is a CPU reference, so real GPU scenes are also required.
`tests/rtx/apps/D3D9Water` supplies a synthetic private-protocol producer for native
x64 and bridged x86 rendering: waves, interaction impulses, domain changes,
disable/re-enable, camera-medium transitions and long-run paired foam validation.
Full-game performance is measured separately after all local modules are ported.
## Stable world-cell addressing

The authored and live foam grids use a shared unsigned-magnitude Euclidean
remainder for their ring-buffer slots. Every cell maps to a unique output region
across either world axis, including negative coordinates. GPU geometry and the
CPU contract test share this helper; the CPU test does not generate production
geometry. The compiled-kernel check rejects signed remainder instructions and
verifies the 8 by 8 by 1 thread groups for all six foam variants.

Vulkan does not define negative operands for `OpSRem`/`OpSMod` without the
`maintenance8` feature. A signed `(cell % side + side) % side` expression can
therefore alias output slots and create concurrent vertex writes. See the
[Vulkan SPIR-V execution rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).

The optional paired-triangle diagnostic compares every word of all 1,181,616
vertices against a second execution of the same kernel and the other kernel
variant. Three asynchronous snapshots distinguish unstable execution from an
optimization difference. Diagnostics are excluded from performance measurements.

Shore wetness reserves common bindings 235/236. The SHARC deferred buffers keep
230 through 233. The water weather unit test checks the combined descriptor slot
set for duplicates, including configurations whose shaders do not use wetness.


## Divergent geometry loads

Each foam-field invocation can hit a different scene mesh. Both the descriptor and the element pointer consumed by `OpLoad` therefore need non-uniform resource access semantics. `foamGeometryLoad` emits that element access explicitly with `StorageBufferArrayNonUniformIndexing`. The scope is limited to the foam triangle helper; all index lookup, transforms, ray queries and field updates remain on the GPU.

The bundled Slang version is `2025.10.4-39-g496fab196`. Its ordinary StructuredBuffer-array expression decorates only the descriptor pointer. The affected generated `water_foam_update` and culled shaders contained 15 undecorated divergent load operands. Both optimized outputs now contain 15 correctly decorated operands. The issue matches the compiler's [NonUniform operand report](https://github.com/shader-slang/slang/issues/10525); no external toolchain or dependency was modified.

An actual game fault on 2026-10-30 reported misaligned accesses at GPU PCs `0x45c0` and `0x7e60`. The exact remapped shader hash `4698fd83c52ba07f` maps those PCs to 32-bit triangle-index loads. The fault happened after an F5 vegetation toggle, but a later same-build off/on session did not reproduce it. This repair addresses a confirmed shader-contract defect at those loads; it is not yet proof that every cause of the intermittent game crash is removed.

A standalone RTX 5070 test extracts the production helper and checks 2,097,152 uint32, uint16 and float reads across 524,288 divergent invocations and 32 descriptor permutations, including output canaries. It passes. The unfixed test also passes on this driver, so it verifies result preservation rather than serving as a deterministic crash reproducer. Full-game stress and final FPS acceptance remain required.

## World-axis foam regression

The supplied 2026-10-27 video was inspected at one-second intervals and in full
frames around the material-path switch. The scene-material foam becomes a thin,
elongated bright strip across the channel. This is consistent with geometry
aliasing, but the exact recorded game location has not yet been replayed.

The old signed remainder expression was tested directly on an RTX 5070 against
an independent 64-bit Euclidean remainder oracle. Across 1,200 grids and 30,894,400
cells (sides 60, 90 and 256), it produced 3,010,630 wrong mappings and 193,462
duplicate slots. Ordinary negative origins such as (-1024,-1024) reproduce the
problem; it is not limited to integer extremes. The migrated unsigned-magnitude
`waterFoamWorldCellSlot` returned zero mismatches, duplicate slots or out-of-range
slots on the same cases, including grids crossing either world axis.

Current full runtime 3C15A768 was also checked through native and x86 bridge hosts.
At frame 600 each compared all 1,181,616 foam vertices (10,634,544 words), with zero
non-finite words, repeated-dispatch mismatches or paired-variant mismatches. Both
output images were finite. These tests establish the addressing correction and
current GPU output consistency, not exact visual acceptance at the video's scene.
