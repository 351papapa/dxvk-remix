#include "rtx/utility/shader_types.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/instance_definitions.h"
#include "rtx/concept/surface/surface.h"
#include "rtx/concept/surface_material/surface_material.h"
#include "rtx/pass/water_surface_args.h"
#include "rtx/pass/water_beach_mask.slangh"

layout(push_constant) ConstantBuffer<WaterSurfaceArgs> args;
layout(binding = WATER_SURFACE_INPUT) StructuredBuffer<float> input;
layout(binding = WATER_SURFACE_BEACH_MASK) StructuredBuffer<uint> beachMask;
layout(binding = WATER_SURFACE_SHORE_DEPTH) RWStructuredBuffer<float4> shoreDepth;
layout(binding = WATER_SURFACE_SHORE_TERRAIN) RWStructuredBuffer<float2> terrain;
layout(binding = WATER_SURFACE_SHORE_SEEDS) StructuredBuffer<float2> seeds;
layout(binding = WATER_SURFACE_SHORE_NEXT) RWStructuredBuffer<float2> nextSeeds;
layout(binding = BINDING_ACCELERATION_STRUCTURE) RaytracingAccelerationStructure sceneTlas;
layout(binding = BINDING_SURFACE_DATA_BUFFER) StructuredBuffer<Surface> surfaces;
layout(binding = BINDING_SURFACE_MATERIAL_DATA_BUFFER) StructuredBuffer<MemoryPolymorphicSurfaceMaterial> materials;

uint shoreIndex(int2 c, bool fine) {
  return (fine ? WATER_COARSE_COUNT : 0) + uint(c.y * (fine ? 91 : 61) + c.x);
}
float2 shoreWorld(int2 c, bool fine) {
  return float2(input[0], input[1]) + (fine ? input[8] : 0.0f)
    + float2(c) * (fine ? input[9] : input[3]);
}
float2 terrainAt(float2 world, bool fine) {
  float2 grid = (world - float2(input[0], input[1]) - (fine ? input[8] : 0.0f)) / (fine ? input[9] : input[3]);
  int limit = fine ? 90 : 60;
  if (any(grid < 0.0f) || any(grid > float(limit))) { return float2(0); }
  int2 c = min(int2(grid), int2(limit-1)); float2 f = grid-float2(c);
  return lerp(lerp(terrain[shoreIndex(c,fine)], terrain[shoreIndex(c+int2(1,0),fine)], f.x),
    lerp(terrain[shoreIndex(c+int2(0,1),fine)], terrain[shoreIndex(c+int2(1,1),fine)], f.x),f.y);
}
float2 nearestSeed(int2 c, bool fine) {
  float2 seed = seeds[shoreIndex(c,fine)];
  if (fine) {
    float2 world = shoreWorld(c,true);
    int2 coarse = clamp(int2(round((world-float2(input[0],input[1]))/input[3])),int2(0),int2(60));
    float2 alternative = seeds[shoreIndex(coarse,false)];
    if (dot(world-alternative,world-alternative) < dot(world-seed,world-seed)) { seed=alternative; }
  }
  return seed;
}

float2 traceShoreBed(float2 world, float maxDepth) {
  // WATER_HEIGHT contains interaction displacement, not the datum. Sampling it
  // would make a passing hull move the shoreline phase.
  float seaLevel = input[2];
  float upReach = max(2.0f, args.beachWaveHeight * 2.0f);
  RayDesc ray;
  ray.Origin = float3(world, seaLevel + upReach);
  ray.Direction = float3(0, 0, -1);
  ray.TMin = 0.0f;
  ray.TMax = upReach + maxDepth;
  uint surfaceCount, surfaceStride, materialCount, materialStride;
  surfaces.GetDimensions(surfaceCount, surfaceStride);
  materials.GetDimensions(materialCount, materialStride);
  const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(sceneTlas, flags, OBJECT_MASK_OPAQUE, ray);
  while (query.Proceed()) {
    if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) { continue; }
    uint surfaceId = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CandidateGeometryIndex();
    if (surfaceId >= surfaceCount || surfaceId >= materialCount) { continue; }
    Surface surface = surfaces[surfaceId];
    if (!surface.isStatic || surface.isAnimatedWater || surface.isParticle || surface.isDecal || surface.isPlayerModel) { continue; }
    MemoryPolymorphicSurfaceMaterial material = materials[surfaceId];
    if (!material.isOpaque()) { continue; }
    // The VC vehicle material class also excludes parked hulls, which can have
    // isStatic set on a frame without transform changes.
    OpaqueSurfaceMaterial opaque = reinterpret<OpaqueSurfaceMaterial>(material);
    uint wetClass = surface.wetnessTag >> 24;
    if (wetClass == 1u || wetClass == 3u) { continue; }
    query.CommitNonOpaqueTriangleHit();
  }
  if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    // Positive below the mean water level, negative on the dry beach. Water
    // tiles may cover both: only the rendered terrain determines the shore.
    return float2(query.CommittedRayT() - upReach, 1.0f);
  }
  return float2(0);
}

bool surfShoreSeed(float2 water, float2 land, out float2 seed) {
  seed = float2(1e20f);
  // Reject authored non-beaches before spending rays on the geometric fallback.
  if (beachSeedMask(beachMask, water) <= 0.5f) { return false; }
  float2 seaward = normalize(water - land);
  // A coarse-grid sign change interpolates a vertical quay into a false ramp.
  // Refine on real geometry before applying the original half-metre-map tests.
  [unroll] for (uint i = 0; i < 4; ++i) {
    float2 mid = (water + land) * 0.5f;
    float2 bed = traceShoreBed(mid, 64.0f);
    if (bed.y <= 0.0f) { return false; }
    if (bed.x > 0.0f) { water = mid; } else { land = mid; }
  }
  float2 candidate = (water + land) * 0.5f;
  if (beachSeedMask(beachMask, candidate) <= 0.5f) { return false; }
  float2 nearBed = traceShoreBed(candidate + seaward * 0.25f, 64.0f);
  float2 sea25 = traceShoreBed(candidate + seaward * 2.5f, 64.0f);
  float2 sea5 = traceShoreBed(candidate + seaward * 5.0f, 64.0f);
  if (nearBed.y <= 0.0f || nearBed.x <= 0.0f || nearBed.x >= 0.7f
      || sea25.y <= 0.0f || sea25.x <= 0.0f || sea25.x >= 1.7f
      || sea5.y <= 0.0f || sea5.x <= 0.0f || sea5.x >= 2.6f) { return false; }
  float2 inland = traceShoreBed(candidate - seaward * 4.0f, 64.0f);
  if (inland.y > 0.0f && inland.x > 0.15f) { return false; }
  uint dryNeighbours = 0u;
  [loop] for (uint i = 0; i < 12; ++i) {
    float angle = float(i) * 0.5235987756f;
    float2 bed = traceShoreBed(land + float2(cos(angle), sin(angle)) * 3.0f, 64.0f);
    if (bed.y <= 0.0f || bed.x <= 0.0f) { ++dryNeighbours; }
  }
  if (dryNeighbours < 5u) { return false; }
  // cs_shore0 uses the authored mask in place of this open-sea fallback.
  if (!hasBeachMask(beachMask)) {
    float2 gradient = float2(0);
    const float2 axes[4] = {float2(1, 0), float2(-1, 0), float2(0, 1), float2(0, -1)};
    [unroll] for (uint j = 0; j < 4; ++j) {
      float2 bed = traceShoreBed(water + axes[j], 64.0f);
      gradient += axes[j] * (bed.y > 0.0f ? clamp(bed.x, -1.0f, 3.0f) : -1.0f);
    }
    float2 direction = dot(gradient, gradient) > 1e-6f ? normalize(gradient) : seaward;
    [loop] for (uint j = 0; j < 14; ++j) {
      float2 bed = traceShoreBed(water + direction * (12.0f + 3.0f * float(j)), 64.0f);
      // No hit is unknown/deep water, never evidence of the opposite bank.
      if (bed.y > 0.0f && bed.x <= 0.0f) { return false; }
    }
  }
  seed = candidate;
  return true;
}

// Runs before scene preparation against one completed TLAS/surface/material
// snapshot. No current-frame bindless vertices or CPU terrain queries are used.
[shader("compute")]
[numthreads(WATER_SURFACE_GROUP_SIZE,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if (id >= WATER_VERTEX_COUNT) { return; }
  bool fine = id >= WATER_COARSE_COUNT;
  uint local = fine ? id - WATER_COARSE_COUNT : id;
  uint side = fine ? 91u : 61u;
  float2 world = float2(input[0], input[1]) + (fine ? input[8] : 0.0f)
    + float2(local % side, local / side) * (fine ? input[9] : input[3]);
  int2 cell = int2(local % side, local / side);
  int limit = fine ? 90 : 60;
  if (args.phase == 1) {
    float2 best = float2(1e20f);
    float bestDistance = 1e30f;
    float2 bed = terrain[id];
    if (bed.y > 0.0f && bed.x > 0.0f) {
      const int2 offsets[4] = {int2(-1,0),int2(1,0),int2(0,-1),int2(0,1)};
      [unroll] for (uint k=0;k<4;++k) {
        int2 c=cell+offsets[k];
        if (any(c<0)||any(c>limit)) { continue; }
        float2 other=terrain[shoreIndex(c,fine)];
        if (other.y <= 0.0f || other.x > 0.0f) { continue; }
        float2 candidate;
        if (!surfShoreSeed(world, shoreWorld(c, fine), candidate)) { continue; }
        float distance=dot(candidate-world,candidate-world);
        if(distance<bestDistance) { best=candidate;bestDistance=distance; }
      }
    }
    nextSeeds[id]=best;
    return;
  }
  if (args.phase > 1 && args.phase < 100) {
    int jump=int(args.phase)-2;
    float2 best=float2(1e20f);float bestDistance=1e30f;
    [unroll] for(int y=-1;y<=1;++y) {
      [unroll] for(int x=-1;x<=1;++x) {
        int2 c=cell+int2(x,y)*jump;
        if(any(c<0)||any(c>limit)) { continue; }
        float2 candidate=seeds[shoreIndex(c,fine)];
        float distance=dot(candidate-world,candidate-world);
        if(distance<bestDistance) { best=candidate;bestDistance=distance; }
      }
    }
    nextSeeds[id]=best;
    return;
  }
  if (args.phase == 101) {
    if (!fine) { return; }
    float2 grid=(world-float2(input[0],input[1]))/input[3];
    int2 c=min(int2(grid),int2(59));float2 f=grid-float2(c);
    float4 a=shoreDepth[shoreIndex(c,false)],b=shoreDepth[shoreIndex(c+int2(1,0),false)];
    float4 e=shoreDepth[shoreIndex(c+int2(0,1),false)],d=shoreDepth[shoreIndex(c+int2(1,1),false)];
    float4 bed=lerp(lerp(a,b,f.x),lerp(e,d,f.x),f.y);
    // Both mesh resolutions share one shoreline envelope; only terrain height is sampled finely.
    if (min(min(a.y,b.y),min(e.y,d.y))>0.99f && terrain[id].y>0.99f) {
      shoreDepth[id]=float4(bed.x,1.0f,terrain[id].x,bed.w);
    }
    return;
  }
  if (args.phase == 100) {
    shoreDepth[id]=float4(0);
    float2 bed=terrain[id];
    float2 seed=nearestSeed(cell,fine);
    if(bed.y<=0.0f || any(abs(seed)>1e10f)) { return; }
    float signedDistance=0.0f,weight=0.0f;
    [loop] for(int y=-4;y<=4;++y) {
      [loop] for(int x=-4;x<=4;++x) {
        int2 c=cell+int2(x,y);
        if(any(c<0)||any(c>limit)) { continue; }
        float2 neighbour=terrain[shoreIndex(c,fine)];
        float2 q=nearestSeed(c,fine);
        if(neighbour.y<=0.0f || any(abs(q)>1e10f)) { continue; }
        float w=exp(-0.180337f*float(x*x+y*y));
        signedDistance+=min(length(shoreWorld(c,fine)-q),48.0f)*(neighbour.x>=0.0f?1.0f:-1.0f)*w;
        weight+=w;
      }
    }
    if(weight<=0.0f) { return; }
    signedDistance/=weight;
    float2 seaward=(world-seed)*(bed.x>=0.0f?1.0f:-1.0f);
    float magnitude=length(seaward);
    if(magnitude<1e-4f) { return; }
    seaward/=magnitude;
    float2 probe=terrainAt(seed+seaward*5.0f,fine);
    if(probe.y<0.99f) { probe=terrainAt(seed+seaward*5.0f,false); }
    if(probe.y<0.99f || probe.x<=0.0f) { return; }
    float slope=clamp(probe.x/5.0f,0.1f,0.3f);
    shoreDepth[id]=float4(signedDistance*slope,1.0f,bed.x,slope);
    return;
  }
  terrain[id] = traceShoreBed(world, max(8.0f, args.beachWaveDepth * 2.0f));
}
