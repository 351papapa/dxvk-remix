// Adapted from xoxor4d/dxvk-remix, commit 7f5210e1a7cd1356bf0271230f82e6132b08ad8c.
// Modified for world orientation, units, surface bounds and dynamic surface classes.
// The original zlib/libpng license is retained in ThirdPartyLicenses.txt.

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/rain_map/rain_map_bindings.slangh"
#include "rtx/pass/instance_definitions.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/concept/surface/surface.slangh"
#include "rtx/pass/rain_map/rain_map_water.slangh"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 threadId : SV_DispatchThreadID) {
  const WetnessArgs args = cb.wetnessArgs;
  if (any(threadId >= args.rainMapResolution)) {
    return;
  }
  const float units = max(cb.metersToWorldUnitScale, 0.0001f);
  const float2 ground = float2(args.rainMapOriginX, args.rainMapOriginY)
    + (float2(threadId) + 0.5f) * args.rainMapTexel;
  RayDesc ray;
  ray.Origin = cb.isZUp ? float3(ground, args.rainMapTraceTop) : float3(ground.x, args.rainMapTraceTop, ground.y);
  ray.Direction = cb.isZUp ? float3(0.0f, 0.0f, -1.0f) : float3(0.0f, -1.0f, 0.0f);
  ray.TMin = 0.0f;
  ray.TMax = max(args.rainMapTraceTop - args.rainMapTraceBottom, units);

  // Nothing is committed, so the extrema include every eligible hit regardless of traversal order.
  const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
  RayQuery<0> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE, ray);
  float top = args.rainMapTraceBottom - units;
  float bottom = args.rainMapTraceTop + units;
  while (query.Proceed()) {
    if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
      continue;
    }
    const uint id = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CandidateGeometryIndex();
    if (id >= cb.surfaceCount) {
      continue;
    }
    const Surface surface = surfaces[nonuniformEXT(id)];
    if (!rainMapStaticGround(surface)) {
      continue;
    }
    const float height = args.rainMapTraceTop - query.CandidateTriangleRayT();
    top = max(top, height);
    bottom = min(bottom, height);
  }

  if (bottom > args.rainMapTraceTop) {
    rainMapHeightOutput[threadId] = float4(args.rainMapTraceBottom - units, args.rainMapTraceBottom - units, 0.0f, 0.0f);
    return;
  }
  const float minCoverHeight = max(args.rainOcclusionMinHeight, 0.01f * units);
  const float cover = top - bottom >= minCoverHeight ? top : bottom;
  const float3 groundPosition = cb.isZUp ? float3(ground, bottom) : float3(ground.x, bottom, ground.y);
  float waterHeight;
  const float waterReach = max(args.rainMapTraceTop - bottom, max(100.0f, args.puddleDetailDistance) * units);
  const bool waterValid = rainMapWaterAbove(groundPosition, waterReach, waterHeight);
  rainMapHeightOutput[threadId] = float4(bottom, cover, waterHeight, float(waterValid));
}
