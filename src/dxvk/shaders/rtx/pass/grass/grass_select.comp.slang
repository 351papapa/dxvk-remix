//!variant grass_init.comp
//!> GRASS_INIT
//!variant grass_cull.comp
//!> GRASS_CULL
//!variant grass_histogram.comp
//!> GRASS_HISTOGRAM
//!variant grass_radix.comp
//!> GRASS_RADIX
//!variant grass_count.comp
//!> GRASS_COUNT
//!variant grass_prefix.comp
//!> GRASS_PREFIX
//!variant grass_scatter.comp
//!> GRASS_SCATTER
//!end-variants

#include "rtx/pass/grass/grass_args.h"

layout(binding = GRASS_CONSTANTS) ConstantBuffer<GrassConstants> cb;
layout(binding = GRASS_ROOTS) StructuredBuffer<GrassRoot> roots;
layout(binding = GRASS_CANDIDATES) RWStructuredBuffer<uint2> candidates;
layout(binding = GRASS_STATE) RWStructuredBuffer<uint> state;
layout(binding = GRASS_GROUPS) RWStructuredBuffer<uint2> groups;
layout(binding = GRASS_SELECTED) RWStructuredBuffer<GrassSelection> selected;
layout(binding = GRASS_PREVIOUS) StructuredBuffer<GrassSelection> previous;

groupshared uint histogram[GRASS_GROUP_SIZE];
groupshared uint2 scan[GRASS_GROUP_SIZE];

uint2 scanGroup(uint lane, uint2 value) {
  scan[lane] = value;
  GroupMemoryBarrierWithGroupSync();
  for (uint offset = 1; offset < GRASS_GROUP_SIZE; offset *= 2) {
    uint2 add = lane >= offset ? scan[lane - offset] : uint2(0);
    GroupMemoryBarrierWithGroupSync();
    scan[lane] += add;
    GroupMemoryBarrierWithGroupSync();
  }
  return scan[lane] - value;
}

float rootFade(GrassRoot root) {
  float2 delta = cb.zUp ? root.positionScale.xy - cb.cameraRadius.xy : root.positionScale.xz - cb.cameraRadius.xz;
  float distanceSquared = dot(delta, delta);
  if (distanceSquared >= cb.cameraRadius.w * cb.cameraRadius.w) { return 0; }
  float distance = sqrt(distanceSquared);
  float scale = root.positionScale.w;
  if (cb.frustumCull != 0 && distance > 6.0) {
    float sphereRadius = scale * (cb.bounds.x * cb.dimensions.x + cb.bounds.y * cb.dimensions.y) + 2.0;
    for (uint i = 0; i < 5; ++i) {
      if (dot(cb.viewPlanes[i].xyz, root.positionScale.xyz) + cb.viewPlanes[i].w < -sphereRadius) { return 0; }
    }
  }
  float upNormal = cb.zUp ? root.normalAngle.z : root.normalAngle.y;
  float tilt = sqrt(max(0.0, 1.0 - upNormal * upNormal));
  float margin = scale * (cb.bounds.x * cb.dimensions.x + cb.bounds.y * cb.dimensions.y * tilt)
    + abs(cb.dimensions.z) * tilt;
  float available = cb.cameraRadius.w - distance;
  if (cb.frontRear != 0) {
    available = max(cb.cameraRadius.w * 0.5 - distance, min(available, dot(delta, cb.zUp ? cb.forwardFade.xy : cb.forwardFade.xz)));
  }
  return saturate((available - margin) / max(0.01, cb.forwardFade.w));
}

float4x4 rootTransform(GrassRoot root, float fade) {
  float3 normal = root.normalAngle.xyz;
  float3 helper = abs(normal.z) < 0.99 ? float3(0, 0, 1) : float3(0, 1, 0);
  float3 tangent = normalize(cross(helper, normal));
  float3 bitangent = cross(normal, tangent);
  float s, c;
  sincos(root.normalAngle.w, s, c);
  float3 x = (tangent * c + bitangent * s) * (cb.dimensions.x * root.positionScale.w);
  float3 y = (-tangent * s + bitangent * c) * (cb.dimensions.x * root.positionScale.w);
  float3 z = normal * (cb.dimensions.y * root.positionScale.w * fade);
  float3 p = root.positionScale.xyz + normal * cb.dimensions.z;
  return float4x4(x.x, y.x, z.x, p.x, x.y, y.y, z.y, p.y, x.z, y.z, z.z, p.z, 0, 0, 0, 1);
}

[shader("compute")]
[numthreads(GRASS_GROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint index = tid.x;
  uint lane = lid.x;
#if defined(GRASS_INIT)
  if (index < 256) { state[index] = 0; }
  if (index == 0) {
    state[256] = 0;
    state[257] = cb.budget;
    state[258] = 0;
    state[259] = 0;
  }
  if (index < cb.budget) {
    GrassSelection empty = (GrassSelection)0;
    empty.identityLo = empty.identityHi = GRASS_INVALID;
    empty.previousSlot = GRASS_INVALID;
    selected[index] = empty;
  }
#elif defined(GRASS_CULL) || defined(GRASS_HISTOGRAM)
  histogram[lane] = 0;
  GroupMemoryBarrierWithGroupSync();
  uint key = GRASS_INVALID;
  if (index < cb.rootCount) {
#if defined(GRASS_CULL)
    GrassRoot root = roots[index];
    float fade = rootFade(root);
    float2 delta = cb.zUp ? root.positionScale.xy - cb.cameraRadius.xy : root.positionScale.xz - cb.cameraRadius.xz;
    key = fade >= 0.01 ? asuint(dot(delta, delta)) : GRASS_INVALID;
    candidates[index] = uint2(key, asuint(fade));
#else
    key = candidates[index].x;
#endif
    uint prefixMask = cb.radixShift == 24 ? 0 : (GRASS_INVALID << (cb.radixShift + 8));
    if (key != GRASS_INVALID && (key & prefixMask) == state[256]) {
      InterlockedAdd(histogram[(key >> cb.radixShift) & 255], 1);
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (histogram[lane] != 0) { InterlockedAdd(state[lane], histogram[lane]); }
#elif defined(GRASS_RADIX)
  if (index != 0) { return; }
  uint rank = state[257];
  uint bucket = 255;
  for (uint i = 0; i < 256; ++i) {
    if (rank <= state[i]) { bucket = i; break; }
    rank -= state[i];
  }
  state[256] |= bucket << cb.radixShift;
  state[257] = rank;
  for (uint i = 0; i < 256; ++i) { state[i] = 0; }
#elif defined(GRASS_COUNT) || defined(GRASS_SCATTER)
  uint key = index < cb.rootCount ? candidates[index].x : GRASS_INVALID;
  bool valid = key != GRASS_INVALID;
  uint2 value = uint2(valid && key < state[256], valid && key == state[256]);
  uint2 localPrefix = scanGroup(lane, value);
#if defined(GRASS_COUNT)
  if (lane == GRASS_GROUP_SIZE - 1) { groups[gid.x] = localPrefix + value; }
#else
  if (!valid) { return; }
  uint2 prefix = groups[gid.x] + localPrefix;
  bool accepted = value.x != 0 || (value.y != 0 && prefix.y < state[259]);
  if (!accepted) { return; }
  uint slot = prefix.x + min(prefix.y, state[259]);
  if (slot >= cb.budget) { return; }
  GrassRoot root = roots[index];
  GrassSelection result = (GrassSelection)0;
  result.transform = rootTransform(root, asfloat(candidates[index].y));
  result.previousTransform = result.transform;
  result.previousSlot = GRASS_INVALID;
  result.valid = 1;
  result.identityLo = root.identityLo;
  result.identityHi = root.identityHi;
  // Compaction preserves root identity order, including across patch cache changes.
  uint lo = 0, hi = cb.previousBudget;
  while (lo < hi) {
    uint mid = (lo + hi) / 2;
    GrassSelection old = previous[mid];
    bool less = old.identityHi < root.identityHi || (old.identityHi == root.identityHi && old.identityLo < root.identityLo);
    if (less) { lo = mid + 1; } else { hi = mid; }
  }
  if (lo < cb.previousBudget) {
    GrassSelection old = previous[lo];
    if (old.valid != 0 && old.identityLo == root.identityLo && old.identityHi == root.identityHi) {
      result.previousTransform = old.transform;
      result.previousSlot = lo;
    }
  }
  selected[slot] = result;
#endif
#elif defined(GRASS_PREFIX)
  if (index != 0) { return; }
  uint2 prefix = uint2(0);
  for (uint i = 0; i < cb.groupCount; ++i) {
    uint2 count = groups[i];
    groups[i] = prefix;
    prefix += count;
  }
  state[259] = min(cb.budget - min(prefix.x, cb.budget), prefix.y);
  state[258] = min(cb.budget, prefix.x + prefix.y);
#endif
}
