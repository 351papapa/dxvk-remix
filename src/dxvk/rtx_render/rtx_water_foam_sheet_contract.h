#pragma once
#include "../shaders/rtx/pass/water_foam_topology.h"
// Layout and topology contract for the GPU foam sheet.
//
// Deliberately dependency-free: this header includes nothing from the renderer, so
// a unit test can assert the same constants the shader and the runtime use without
// linking the engine. An earlier attempt put these in rtx_water_foam_geometry.h,
// whose CommonDeviceObject base drags in RtxOption statics and Logger, which a unit
// test cannot link. Splitting them keeps ONE definition rather than a test copy,
// which matters because a drifted constant here produces a torn or garbage sheet
// rather than a visible failure.
//
// The values are a contract with two other places and must be changed together:
//   * water_foam_surface.comp.slang     - FoamSurfaceVertex and WaterFoamSurfaceArgs
//   * rtx_water_foam_geometry.cpp       - buildGeometry()'s RasterBuffer views
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace dxvk {

// One foam sheet vertex. Written by compute, read as four separate RasterBuffer
// views, so this layout is a contract with that shader.
// stride 36: position@0, normal@12, texcoord@24, color@32.
struct WaterFoamGpuVertex {
  float position[3];
  float normal[3];
  float texcoord[2];
  uint32_t color;
};
static_assert(sizeof(WaterFoamGpuVertex) == 36, "WaterFoamGpuVertex stride must stay 36 bytes");
static_assert(offsetof(WaterFoamGpuVertex, position) == 0, "position must be at offset 0");
static_assert(offsetof(WaterFoamGpuVertex, normal) == 12, "normal must be at offset 12");
static_assert(offsetof(WaterFoamGpuVertex, texcoord) == 24, "texcoord must be at offset 24");
static_assert(offsetof(WaterFoamGpuVertex, color) == 32, "color must be at offset 32");

// Push constants for the sheet generator. Must match WaterFoamSurfaceArgs in
// water_foam_surface.comp.slang field for field and in order.
struct WaterFoamSurfaceArgs {
  float worldToClip[16];     // offset 0, 64 bytes, unjittered camera
  uint32_t vertexCount;      // offset 64
  uint32_t sheetSide;        // offset 68
  float sheetExtent;         // offset 72
  // Up-axis, passed here rather than read from the frame's common constant buffer.
  // This pass runs during scene preparation, BEFORE RtxContext fills and uploads the
  // raytrace constants (updateRaytraceArgsConstantBuffer), so cb.isZUp would be
  // stale or unset at this point. Carrying the axis in the push constant keeps the
  // pass self-contained and independent of the main ray tracing setup.
  uint32_t upAxisIsZ;        // offset 76, 1 when the up axis is Z
  float gridOrigin[2];       // offset 80, world-space origin of the density field
  float gridSpacing[2];      // offset 88, world units per density cell
  float coverageScale;       // offset 96, user foam strength
  float surfaceOffset;       // offset 100, lift above the recorded water surface
  float time;                // offset 104, seconds since start; drives erosion and UV drift
  float edgeFadeWidth;       // offset 108, domain edge fade in cells
  float cameraUp;            // offset 112, camera height of the density snapshot
  float textureScale;        // offset 116, world-space texture repeat distance
};
static_assert(sizeof(WaterFoamSurfaceArgs) == 120, "WaterFoamSurfaceArgs must stay 120 bytes");
static_assert(offsetof(WaterFoamSurfaceArgs, textureScale) == 116,
  "textureScale must be at byte offset 116");
static_assert(offsetof(WaterFoamSurfaceArgs, cameraUp) == 112,
  "cameraUp must be at byte offset 112");
static_assert(offsetof(WaterFoamSurfaceArgs, coverageScale) == 96,
  "coverageScale must be at byte offset 96");
static_assert(offsetof(WaterFoamSurfaceArgs, surfaceOffset) == 100,
  "surfaceOffset must be at byte offset 100");
static_assert(offsetof(WaterFoamSurfaceArgs, time) == 104,
  "time must be at byte offset 104");
static_assert(offsetof(WaterFoamSurfaceArgs, edgeFadeWidth) == 108,
  "edgeFadeWidth must be at byte offset 108");

// Carrier heights and validity live in their own R32G32B32A32_SFLOAT resource:
//   .xy water height/validity, .zw eligible static terrain height/validity.
//
// Not a channel of the density field, and not a biased half. The density field is
// R16G16B16A16_SFLOAT, where adjacent representable values near 1024 are 1.0 apart and
// 0.5 apart below that: storing a biased world height there quantises the water to whole
// or half units, which is coarser than the surface offset and makes the foam step and its
// normals stair-step. A biased encoding also collides with the sentinel whenever a legal
// world height equals minus the bias.
//
// Separating validity from height means no value is reserved out of the height range at
// all: every finite float is a legal height, and absence is its own flag.
inline constexpr float kWaterFoamHasWater = 1.0f;
inline constexpr float kWaterFoamNoWater = 0.0f;

// Tolerance for the height regression: the format is 32-bit float, so round trip error is
// at the float epsilon scale, orders of magnitude below the surface offset.
inline constexpr float kWaterFoamHeightTolerance = 1.0e-4f;

// Probe reach relative to the camera: the contact probe fires from cameraUp + 64 downward
// with its ray clamped to 256, so any recorded surface lies within [-192, +64] of the
// camera. Used to size the display bounding box and to bound the bias. MUST correspond to
// ray.Origin / ray.TMax in water_foam_update.comp.slang.
inline constexpr float kWaterFoamProbeUpReach = 64.0f;
inline constexpr float kWaterFoamProbeDownReach = 192.0f;

// Domain edge fade, in cells, matching the composite's smoothstep(1, 9, edge).
inline constexpr float kWaterFoamEdgeFadeCells = 8.0f;

// The contact probe's ray clamp, in world units: the largest distance that probe can
// ever report, and therefore a conservative bound on how far the water surface can be
// from the camera level. Used to size the display bounding box without a GPU readback.
// Must match ray.TMax in water_foam_update.comp.slang.
inline constexpr float kWATER_FOAM_PROBE_CLAMP = 256.0f;
static_assert(offsetof(WaterFoamSurfaceArgs, vertexCount) == 64,
  "vertexCount must follow the matrix at byte offset 64");
static_assert(offsetof(WaterFoamSurfaceArgs, sheetSide) == 68,
  "sheetSide must be at byte offset 68");
static_assert(offsetof(WaterFoamSurfaceArgs, sheetExtent) == 72,
  "sheetExtent must be at byte offset 72");
static_assert(offsetof(WaterFoamSurfaceArgs, upAxisIsZ) == 76,
  "upAxisIsZ must be at byte offset 76");
static_assert(offsetof(WaterFoamSurfaceArgs, gridOrigin) == 80,
  "gridOrigin must be at byte offset 80");
static_assert(offsetof(WaterFoamSurfaceArgs, gridSpacing) == 88,
  "gridSpacing must be at byte offset 88");

// Foam line display surface: one vertex per density-field cell, so the display mesh
// and the simulation domain cannot drift apart. The field is 257x257 cells, which is
// 256x256 quads.
//
// This replaces the earlier fixed 32x32 / 24-unit prototype sheet. That sheet existed
// only to prove the pipeline; it is not the foam line and must not be shipped as a
// fix. The display surface now covers exactly the simulated contact-foam domain and
// takes its coverage from the density field.
inline constexpr uint32_t kWaterFoamFieldSize = 257;
// Six vertices per density quad: standalone triangles, so an invalid triangle can be
// collapsed by writing one position three times. A shared corner grid cannot express
// that at all - its boundary triangles always have non-zero area, which is exactly how
// foam geometry ended up stretched from the foam edge to the world origin.
inline constexpr uint32_t kWaterFoamSheetSide = kWaterFoamFieldSize - 1u;   // 256 quads
inline constexpr uint32_t kWaterFoamVertsPerQuad = 6u;

// Workgroup shape. MUST match [numthreads(...)] in water_foam_surface.comp.slang, and the
// CPU dispatch below divides the quad count by kWaterFoamThreadsPerGroup.
//
// These exist because of a real failure: the shader had "[threadnum(8,8,1)]" - a
// misspelling, not a compile error. Slang silently ignored the unknown attribute and the
// shader kept the default workgroup size of 1x1x1, so the CPU dispatched 32x32 groups
// that produced 1024 threads for 65536 quads: one sixty-fourth of the domain, all in the
// low-index corner, and the foam line vanished once nothing else drew it. Declaring the
// shape here and asserting the coverage arithmetic makes that class of error visible to a
// test instead of only to a GPU run.
inline constexpr uint32_t kWaterFoamThreadsPerGroupX = 8u;
inline constexpr uint32_t kWaterFoamThreadsPerGroupY = 8u;
inline constexpr uint32_t kWaterFoamThreadsPerGroup =
  kWaterFoamThreadsPerGroupX * kWaterFoamThreadsPerGroupY;

// Groups the CPU must dispatch so every quad gets a thread.
inline constexpr uint32_t kWaterFoamDispatchGroups =
  (kWaterFoamSheetSide + kWaterFoamThreadsPerGroupX - 1u) / kWaterFoamThreadsPerGroupX;
inline constexpr float kWaterFoamFieldSpacing = 0.4f;                       // world units per cell
// Extent of the simulated domain, and therefore of the display surface.
inline constexpr float kWaterFoamSheetExtent = float(kWaterFoamSheetSide) * kWaterFoamFieldSpacing;

// Density at which the display coverage reaches full. The update pass stores density
// up to 1.6; the old screen-space composite used 1 - exp(-density * 2.8). Using the
// same curve here keeps the foam line's shape and fade identical to what the
// composite drew, so this is a change of lighting path, not a change of appearance.
inline constexpr float kWaterFoamDensityToCoverage = 2.8f;

// Vertical offset above the water surface, purely to avoid depth fighting with the
// water it sits on. Deliberately tiny: the surface must not float.
inline constexpr float kWaterFoamSurfaceOffset = 0.02f;

// Derived counts, so the buffer sizing and the topology loop cannot disagree.
inline constexpr uint32_t kWaterFoamSheetVertexCount =
  kWaterFoamSheetSide * kWaterFoamSheetSide * kWaterFoamVertsPerQuad;
inline constexpr uint32_t kWaterFoamSheetIndexCount = kWaterFoamSheetSide * kWaterFoamSheetSide * 6u;

// Stable geometry identity hashes.
//
// The internal geometry path keys its caches on these and compares them to decide
// between an instance update and a BVH update. They must be initialised (an
// uninitialised hash makes the update decision meaningless) and must stay CONSTANT
// for this fixed topology: rotating them per frame would grow the geometry cache and
// the BLAS without bound. Values are arbitrary but distinct from the caustics and
// foam materials' hashes so the three cannot collide.
inline constexpr uint64_t kWaterFoamSheetHashIndices    = 0x5746544D5348544Aull; // "WFTMSHTI"
inline constexpr uint64_t kWaterFoamSheetHashPosition   = 0x5746544D53485053ull; // "WFTMSHPS"
inline constexpr uint64_t kWaterFoamSheetHashTexcoord   = 0x5746544D53485458ull; // "WFTMSHTX"
inline constexpr uint64_t kWaterFoamSheetHashDescriptor = 0x5746544D53484454ull; // "WFTMSHDS"
inline constexpr uint64_t kWaterFoamSheetHashLayout     = 0x5746544D53484C59ull; // "WFTMSHLY"

// Stable mesh and material handles for the single prototype sheet. Shared here so the
// geometry module, the submit path and the cache diagnostics all compare against one
// constant instead of repeating a magic number in several files.
inline constexpr uint64_t kWaterFoamPrototypeMeshHandleValue = 0x57464D50524F5431ull;     // "WFMPROT1"
inline constexpr uint64_t kWaterFoamPrototypeMaterialHandleValue = 0x57464D4750554D54ull; // "WFMGPUMT"

struct WaterFoamBox {
  float minX, minY, minZ;
  float maxX, maxY, maxZ;
};

// The density domain supplies the horizontal footprint and the observed carrier
// range supplies the vertical bound. Until observed, use the contact probe reach.
inline WaterFoamBox waterFoamPlacedSheetBox(bool zUp, float originX, float originY,
    float heightCentre, float heightHalfRange) {
  const float margin = heightHalfRange > 0.0f
    ? heightHalfRange + kWaterFoamSurfaceOffset + 8.0f : kWATER_FOAM_PROBE_CLAMP;
  if (zUp) {
    return {originX, originY, heightCentre-margin,
      originX+kWaterFoamSheetExtent, originY+kWaterFoamSheetExtent, heightCentre+margin};
  }
  return {originX, heightCentre-margin, originY,
    originX+kWaterFoamSheetExtent, heightCentre+margin, originY+kWaterFoamSheetExtent};
}

inline bool waterFoamVertexAttributesValid(const WaterFoamGpuVertex& vertex) {
  for (float value : vertex.position) { if (!std::isfinite(value)) { return false; } }
  for (float value : vertex.normal) { if (!std::isfinite(value)) { return false; } }
  for (float value : vertex.texcoord) { if (!std::isfinite(value)) { return false; } }
  const float normalLengthSquared = vertex.normal[0]*vertex.normal[0]
    + vertex.normal[1]*vertex.normal[1] + vertex.normal[2]*vertex.normal[2];
  return std::abs(normalLengthSquared-1.0f) <= 0.02f;
}

} // namespace dxvk
