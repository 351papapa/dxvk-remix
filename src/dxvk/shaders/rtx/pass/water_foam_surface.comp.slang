//!variant water_foam_surface.comp
//!variant water_foam_surface_culled.comp
//!>       FOAM_VIEW_CULLING
//!variant water_foam_surface_live.comp
//!>       FOAM_LIVE_WATER
//!variant water_foam_surface_live_culled.comp
//!>       FOAM_LIVE_WATER FOAM_VIEW_CULLING
//!variant water_foam_surface_live_pair_reuse.comp
//!>       FOAM_LIVE_WATER FOAM_TRIANGLE_PAIR_REUSE
//!variant water_foam_surface_live_culled_pair_reuse.comp
//!>       FOAM_LIVE_WATER FOAM_VIEW_CULLING FOAM_TRIANGLE_PAIR_REUSE
//!end-variants
// Foam line display surface generator.
//
// Coverage comes from the persistent density field the contact pass produces, the surface
// follows the current generated ocean (or recorded authored water), and the resulting mesh is lit by the
// ordinary material path exactly like the WaterLab wake foam. Nothing here computes a
// light value.
//
// TOPOLOGY: six vertices per density quad, NOT a shared corner grid.
//
// A shared grid cannot express "this triangle has no water". An earlier per-vertex attempt
// only borrowed a neighbour's height, leaving the three corners at their own in-plane
// positions: the triangle kept full area and was still drawn, because alpha controls
// blending, not whether the triangle exists for shadow and reflection rays. With
// standalone triangles an invalid triangle is collapsed by writing the SAME position for
// all three vertices, which really is zero area and cannot be hit by anything.
//
// The index buffer stays static: quad k owns vertices [6k, 6k+6), so no dynamic index
// generation and no cache/hash/topology resynchronisation is needed.
//
// Self-contained by design. This runs during scene preparation, before RtxContext fills
// and uploads the raytrace constants, so cb and the ray tracing bank are not usable.
// Bindings 222 (vertex output) and 223 (density field) are bound directly by
// WaterFoamGpuGeometry::dispatchSheetUpdate.

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/water_foam_noise.slangh"
#include "rtx/pass/water_foam_topology.h"
#include "rtx/pass/water_foam_sentinel.slangh"
#include "rtx/pass/water_foam_culling.slangh"

// Push constant layout is owned by the CPU side (WaterFoamSurfaceArgs in
// rtx_water_foam_sheet_contract.h). Field order and types must match exactly; the C++
// header carries offsetof asserts for the same contract.
struct WaterFoamSurfaceArgs {
  float4x4 worldToClip;     // offset 0, current unjittered camera
  uint     vertexCount;     // offset 64
  uint     sheetSide;       // offset 68
  float    sheetExtent;     // offset 72
  uint     upAxisIsZ;       // offset 76
  float2   gridOrigin;      // offset 80, world origin of the density field snapshot
  float2   gridSpacing;     // offset 88, world units per density cell
  float    coverageScale;   // offset 96, user foam strength
  float    surfaceOffset;   // offset 100, lift above the recorded water surface
  float    time;            // offset 104, clock of the density snapshot
  float    edgeFadeWidth;   // offset 108, domain edge fade in cells
  float    cameraUp;        // offset 112, camera height of the density snapshot
  float    textureScale;    // offset 116, world-space texture repeat distance
};

layout(push_constant) ConstantBuffer<WaterFoamSurfaceArgs> args;

// Byte-identical to WaterFoamGpuVertex: stride 36, position@0, normal@12, uv@24, color@32.
struct FoamSurfaceVertex {
  float3 position;
  float3 normal;
  float2 texcoord;
  uint   color;      // BGRA8; alpha is coverage
};

layout(binding = BINDING_WATER_FOAM_SHEET_VERTICES)
RWStructuredBuffer<FoamSurfaceVertex> FoamSheetVertices;

// .x water foam density, .yz advection velocity, .w terrain wetness (never foam).
layout(binding = BINDING_WATER_FOAM_SURFACE_FIELD) Texture2D<float4> FoamDensity;

// .xy water height/kind (0 absent, 1 authored, 2 ocean), .zw terrain height/validity.
layout(binding = BINDING_WATER_FOAM_SURFACE_HEIGHT) Texture2D<float4> FoamHeight;
#ifdef FOAM_LIVE_WATER
#include "rtx/pass/water_mesh_sample.slangh"
layout(binding = 225) StructuredBuffer<float> LiveWaterInput;
layout(binding = 226) StructuredBuffer<float3> LiveWaterPositions;
layout(binding = 227) StructuredBuffer<float4> LiveWaterDepth;
#endif

// Coverage keeps the curve the replaced composite used, so the foam line's silhouette and
// fade do not change just because the lighting path did.
float foamDensityToCoverage(float density) {
  return 1.0f - exp(-max(density, 0.0f) * 2.8f);
}

// World position of a density cell at a given height.
float3 foamCellWorld(int2 c, float height) {
  const float2 w = float2(args.gridOrigin) + float2(c) * float2(args.gridSpacing);
  const float up = height + args.surfaceOffset;
  return args.upAxisIsZ != 0u ? float3(w.x, w.y, up) : float3(w.x, up, w.y);
}

// Anchor used to collapse an invalid triangle. It must be finite and inside the domain so
// a degenerate triangle never contributes stray geometry anywhere in the scene.
float3 foamAnchor(int2 c) {
  // Prefer a cell that actually has water, so the collapsed triangle sits on the water
  // rather than at an arbitrary spot.
  const int maxCell = int(args.sheetSide);
  const int2 probes[5] = { int2(0, 0), int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };
  [unroll] for (uint k = 0; k < 5; ++k) {
    const int2 q = clamp(c + probes[k], int2(0), int2(maxCell));
    const FoamCarrier carrier = foamSelectCarrier(FoamHeight.Load(int3(q, 0)), float4(0));
    if (carrier.valid) { return foamCellWorld(q, carrier.height); }
  }
  // No water anywhere near: anchor on the snapshot's camera height, which is finite and
  // known to lie within the probe's reach of any surface this field could have recorded.
  return foamCellWorld(c, args.cameraUp);
}

// Everything one corner of a triangle contributes. Computed PER CORNER, from that corner's
// own density, flow and position.
//
// This used to be computed once from the first corner and copied to all three vertices,
// with the UV offsets hardcoded for the first triangle's corner order. Both were visible
// bugs: coverage became blocky per quad instead of interpolating, and the second triangle
// (00,11,01) got offsets that belong to (00,10,11), so its UVs did not match its positions
// and the texture sheared across it.
struct FoamCornerAttr {
  float3 position;
  float2 uv;
  uint   alpha;
  float3 normal;
  bool   valid;
  bool   live;
  float  height;
};

FoamCornerAttr foamBuildCornerAttributes(int2 c) {
  const int maxCell = int(args.sheetSide);
  const float4 state = FoamDensity.Load(int3(c, 0));
  float4 heights = FoamHeight.Load(int3(c, 0));
  float3 liveGradient = 0;
  bool live = false;
#ifdef FOAM_LIVE_WATER
  if (heights.y > 1.5f) {
    float2 world=args.gridOrigin+float2(c)*args.gridSpacing;
    float2 local=world-float2(LiveWaterInput[0],LiveWaterInput[1]);
    // The generated-water overlay owns this domain, including its clipped dry edge.
    if(all(local>=0.0f) && all(local<=LiveWaterInput[12])) { heights.xy=0.0f; }
  }
#endif
  const FoamCarrier carrier = foamSelectCarrier(heights, state);

  FoamCornerAttr r;
  r.valid = carrier.valid;
  r.live = live;
  r.height = carrier.height;
  r.position = foamCellWorld(c, carrier.height);
  float2 derivative = float2(0);
  [unroll] for (uint axis = 0; axis < 2; ++axis) {
    int2 step = axis == 0 ? int2(1, 0) : int2(0, 1);
    int2 lo = max(c - step, int2(0));
    int2 hi = min(c + step, int2(maxCell));
    FoamCarrier a = foamSelectCarrier(FoamHeight.Load(int3(lo, 0)), FoamDensity.Load(int3(lo, 0)));
    FoamCarrier b = foamSelectCarrier(FoamHeight.Load(int3(hi, 0)), FoamDensity.Load(int3(hi, 0)));
    float distance = float(hi[axis] - lo[axis]) * args.gridSpacing[axis];
    if (a.valid && b.valid && a.water == carrier.water && b.water == carrier.water
        && foamCarrierContinuous(a.height, b.height, distance)) {
      derivative[axis] = (b.height - a.height) / max(distance, 0.001f);
    }
  }
  if (live) { derivative = liveGradient.yz; }
  r.normal = normalize(args.upAxisIsZ != 0u
    ? float3(-derivative.x, -derivative.y, 1)
    : float3(-derivative.x, 1, -derivative.y));

  const float2 world2 = float2(args.gridOrigin) + float2(c) * float2(args.gridSpacing);
  // Invalid/dry carriers contribute no foam; wet terrain history is separate.
  const float time = carrier.water ? args.time : 0.0f;
  const float scale = max(args.textureScale, 0.1f);
  const float detailScale = 5.0f / scale;
  const float noise = foamNoise(world2 * (0.42f * detailScale) + float2(time * 0.13f, -time * 0.09f)).x;
  const float fine = foamNoise(world2 * (1.1f * detailScale) + float2(-time * 0.21f, time * 0.16f)).x;
  const float densityCurve = foamDensityToCoverage(carrier.density);
  const float erosion = saturate((densityCurve - (noise * 0.38f + fine * 0.18f)) / 0.5f);
  const float gridX = float(c.x);
  const float gridY = float(c.y);
  const float edge = min(min(gridX, gridY), min(float(maxCell) - gridX, float(maxCell) - gridY));
  const float edgeFade = smoothstep(1.0f, 1.0f + args.edgeFadeWidth, edge);
  float coverage = densityCurve * erosion * edgeFade * args.coverageScale;
  if (!isfinite(coverage)) { coverage = 0.0f; }
  r.alpha = uint(saturate(coverage) * 255.0f + 0.5f);

  // UV from this corner's own world position and flow, so a shared spatial corner gets the
  // same UV in both triangles that use it and the texture does not shear.
  // A continuous rotation and low-frequency warp break axis-aligned texture repeats.
  const float2 rotated = float2(0.8f * world2.x - 0.6f * world2.y,
    0.6f * world2.x + 0.8f * world2.y);
  const float2 warp = foamNoise(world2 * (0.09f * detailScale)) - 0.5f;
  r.uv = rotated / scale + warp * 0.65f
    + float2(-0.045f, 0.035f) * time
    - (carrier.water ? state.yz : float2(0)) * 0.14f
    + (float2(noise, fine) - 0.5f) * 0.22f;
  return r;
}

void foamWriteVertex(uint index, float3 position, float3 normal, float2 uv, uint alpha) {
  FoamSurfaceVertex v;
  v.position = position;
  v.normal = normal;
  v.texcoord = uv;
  v.color = (alpha << 24) | 0x00ffffffu;
  FoamSheetVertices[index] = v;
}

// Emits one triangle: three standalone vertices, or three coincident ones when the
// triangle cannot be drawn. Each vertex uses the attributes of ITS OWN corner.
void foamEmitTriangle(uint base, int2 c0, FoamCornerAttr a, FoamCornerAttr b, FoamCornerAttr d, bool zUp) {
  const float3 upNormal = zUp ? float3(0, 0, 1) : float3(0, 1, 0);

  // Wet and dry corners may meet on a continuous beach. Only invalid carriers,
  // height discontinuities, or wholly transparent triangles are collapsed.
  const float2 ab = zUp ? b.position.xy - a.position.xy : b.position.xz - a.position.xz;
  const float2 ad = zUp ? d.position.xy - a.position.xy : d.position.xz - a.position.xz;
  const float2 bd = zUp ? d.position.xy - b.position.xy : d.position.xz - b.position.xz;
  const bool continuous = foamCarrierContinuous(a.height, b.height, length(ab))
    && foamCarrierContinuous(a.height, d.height, length(ad))
    && foamCarrierContinuous(b.height, d.height, length(bd));
  if (!a.valid || !b.valid || !d.valid || !continuous || (a.alpha | b.alpha | d.alpha) == 0u) {
    const float3 anchor = a.valid ? a.position : foamAnchor(c0);
    foamWriteVertex(base + 0u, anchor, upNormal, float2(0, 0), 0u);
    foamWriteVertex(base + 1u, anchor, upNormal, float2(0, 0), 0u);
    foamWriteVertex(base + 2u, anchor, upNormal, float2(0, 0), 0u);
    return;
  }


  // Shared spatial corners use identical normals, hiding the density-grid diagonal.
  foamWriteVertex(base + 0u, a.position, a.normal, a.uv, a.alpha);
  foamWriteVertex(base + 1u, b.position, b.normal, b.uv, b.alpha);
  foamWriteVertex(base + 2u, d.position, d.normal, d.uv, d.alpha);
}

#ifdef FOAM_LIVE_WATER
#include "rtx/pass/water_foam_live.slangh"
#endif

[shader("compute")]
// MUST be numthreads, not threadnum. The misspelling is not a compile error: the unknown
// attribute is silently ignored and the shader keeps the default workgroup size of 1x1x1.
// The CPU dispatches ceil(quads/8)^2 groups, so with 256 quads that is 32x32 groups but
// only 1 thread each - 1024 of the 65536 quads, one sixty-fourth of the domain, all in the
// low-index corner. That is what made the foam line disappear once the old screen-space
// composite stopped covering for it.
// 8x8 must equal kWaterFoamThreadsPerGroupX/Y in rtx_water_foam_sheet_contract.h, which the
// CPU dispatch and its test both use. A misspelled attribute here is silently ignored and
// reverts the workgroup to 1x1x1, which is how the foam line disappeared once.
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if (tid.z == 1u) {
    uint triangle=tid.y*args.sheetSide+tid.x;
#ifdef FOAM_LIVE_WATER
    foamEmitWater(triangle);
#else
    // Switching off live water also clears its previous geometry.
    if (triangle<WATER_FOAM_LIVE_TRIANGLES) {
      bool coarse=triangle<WATER_FOAM_COARSE_TRIANGLES;
      uint count=coarse?3u*WATER_FOAM_COARSE_SUBDIVISIONS*WATER_FOAM_COARSE_SUBDIVISIONS:3u;
      uint dst=WATER_FOAM_AUTHORED_VERTICES+(coarse?triangle*count:
        WATER_FOAM_COARSE_VERTICES+(triangle-WATER_FOAM_COARSE_TRIANGLES)*3u);
      float3 anchor=foamCellWorld(int2(args.sheetSide/2),args.cameraUp);
      float3 normal=args.upAxisIsZ!=0u?float3(0,0,1):float3(0,1,0);
      [loop] for(uint i=0;i<count;++i) { foamWriteVertex(dst+i,anchor,normal,float2(0),0u); }
    }
#endif
    return;
  }
  const bool zUp = args.upAxisIsZ != 0u;
  const uint quads = args.sheetSide;
  if (tid.x >= quads || tid.y >= quads) { return; }
  // A stable world-cell slot preserves previousPosition correspondence while
  // the camera moves the window; wrapped slots are hidden by the domain fade.
  int2 worldCell = int2(round(args.gridOrigin / args.gridSpacing)) + int2(tid.xy);
  uint2 slot = uint2(waterFoamWorldCellSlot(worldCell.x, quads),
    waterFoamWorldCellSlot(worldCell.y, quads));
  const uint base = (uint(slot.y) * quads + uint(slot.x)) * 6u;
  if (base + 5u >= args.vertexCount) { return; }

  const int2 c00 = int2(tid.xy);
  const int2 c10 = int2(tid.x + 1u, tid.y);
  const int2 c01 = int2(tid.x, tid.y + 1u);
  const int2 c11 = int2(tid.x + 1u, tid.y + 1u);
#ifdef FOAM_VIEW_CULLING
  if (foamTileOutsideView(args.worldToClip, tid.xy, args.gridOrigin,
      args.gridSpacing.x, args.cameraUp, zUp, abs(args.surfaceOffset))) {
    const float3 anchor = foamCellWorld(c00, args.cameraUp);
    const float3 normal = zUp ? float3(0, 0, 1) : float3(0, 1, 0);
    [unroll] for (uint i = 0; i < 6; ++i) {
      foamWriteVertex(base + i, anchor, normal, float2(0), 0u);
    }
    return;
  }
#endif
  if (max(max(FoamDensity.Load(int3(c00,0)).x,FoamDensity.Load(int3(c10,0)).x),
      max(FoamDensity.Load(int3(c01,0)).x,FoamDensity.Load(int3(c11,0)).x)) <= 1e-6f) {
    float3 anchor=foamCellWorld(c00,args.cameraUp);
    float3 normal=zUp ? float3(0,0,1) : float3(0,1,0);
    [unroll] for (uint i=0;i<6;++i) { foamWriteVertex(base+i,anchor,normal,float2(0),0u); }
    return;
  }
  const FoamCornerAttr a00 = foamBuildCornerAttributes(c00);
  const FoamCornerAttr a10 = foamBuildCornerAttributes(c10);
  const FoamCornerAttr a01 = foamBuildCornerAttributes(c01);
  const FoamCornerAttr a11 = foamBuildCornerAttributes(c11);
#ifdef FOAM_VIEW_CULLING
  bool culled;
  {
    const bool valid = a00.valid || a10.valid || a01.valid || a11.valid;
    float3 lo = a00.valid ? a00.position : foamCellWorld(c00, args.cameraUp);
    float3 hi = lo;
    const float3 p10 = a10.valid ? a10.position : foamCellWorld(c10, args.cameraUp);
    const float3 p01 = a01.valid ? a01.position : foamCellWorld(c01, args.cameraUp);
    const float3 p11 = a11.valid ? a11.position : foamCellWorld(c11, args.cameraUp);
    lo = min(min(lo, p10), min(p01, p11));
    hi = max(max(hi, p10), max(p01, p11));
    culled = !valid || foamOutsideView(args.worldToClip, (lo + hi) * 0.5f, (hi - lo) * 0.5f);
  }
  if (culled) {
    // Clear both triangles: early return alone would expose stale ping-pong vertices.
    const float3 anchor = foamCellWorld(c00, args.cameraUp);
    const float3 normal = zUp ? float3(0, 0, 1) : float3(0, 1, 0);
    [unroll] for (uint i = 0; i < 6; ++i) {
      foamWriteVertex(base + i, anchor, normal, float2(0), 0u);
    }
    return;
  }
#endif
  // Two triangles per quad, matching the static index buffer: (00,10,11) and (00,11,01).
  foamEmitTriangle(base, c00, a00, a10, a11, zUp);
  foamEmitTriangle(base + 3u, c00, a00, a11, a01, zUp);
}
