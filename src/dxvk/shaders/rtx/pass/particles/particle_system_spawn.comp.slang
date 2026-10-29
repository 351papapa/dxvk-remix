//!feature wetness ENABLE_SURFACE_WETNESS
/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include "rtx/pass/particles/particle_system_bindings.slangh"

#include "rtx/pass/instance_definitions.h"   // OBJECT_MASK_OPAQUE (spawn occlusion trace)

#include "rtx/utility/common.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/procedural_noise.slangh"
#include "rtx/utility/brdf.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/gbuffer_helpers.slangh"
#include "rtx/utility/debug_view_helpers.slangh"
#include "rtx/concept/camera/camera.slangh"
    
layout(binding = PARTICLE_SYSTEM_BINDING_SPAWN_CONTEXT_PARTICLE_MAPPING_INPUT)
StructuredBuffer<uint16_t> SpawnContextParticleMapping;

layout(binding = PARTICLE_SYSTEM_BINDING_SPAWN_CONTEXTS_INPUT)
StructuredBuffer<GpuSpawnContext> SpawnContexts;

// Outputs

layout(binding = PARTICLE_SYSTEM_BINDING_PARTICLES_BUFFER_INPUT_OUTPUT)
RWStructuredBuffer<GpuParticle> Particles;

// Spawn-occlusion counters, cleared every frame and read back on a delay (RtxParticleSystemManager::simulate):
// [0] particles that ran the occlusion trace, [1] particles killed at birth because every candidate was covered,
// [2] particles whose life the landing trace shortened, [3] particles moved to open sky by relocation.
// [4] contact candidates, [5] water contacts, [6] wet-ground contacts, [7] rejected contacts.
layout(binding = PARTICLE_SYSTEM_BINDING_SPAWN_TRACE_STATS_OUTPUT)
RWStructuredBuffer<uint> SpawnTraceStats;


#ifdef PUDDLE_CONTACT
#include "rtx/pass/wetness_helpers.slangh"
bool tracePuddleGround(float3 position, out uint surfaceId, out float3 hit) {
  surfaceId = 0;
  hit = position;
  RayDesc ray;
  ray.Origin = position + wetWorldVector(float3(0,0,15.f * particleCb.sceneScale));
  ray.Direction = wetWorldVector(float3(0,0,-1)); ray.TMin = 0;
  ray.TMax = 80.f * particleCb.sceneScale;
  const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE, ray);
  while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
      uint id = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CandidateGeometryIndex();
      uint materialCount, materialStride;
      surfaceMaterials.GetDimensions(materialCount, materialStride);
      uint surfaceCount, surfaceStride;
      surfaces.GetDimensions(surfaceCount, surfaceStride);
      if (id < materialCount && id < surfaceCount && !surfaces[id].isParticle && surfaceMaterials[id].isOpaque()) { query.CommitNonOpaqueTriangleHit(); }
    }
  }
  if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }
  surfaceId = (query.CommittedInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CommittedGeometryIndex();
  hit = ray.Origin + ray.Direction * query.CommittedRayT();
  return true;
}
float wetRoadAtContact(float3 position, out uint id, out float3 hit) {
  id = 0;
  hit = position;
  if (particleCb.sceneTlasValid == 0 || cb.wetnessArgs.enabled == 0) { return 0; }
  if (!tracePuddleGround(position, id, hit)) { return 0; }
  uint surfaceCount, surfaceStride;
  surfaces.GetDimensions(surfaceCount, surfaceStride);
  if (id >= surfaceCount) { return 0; }
  uint tag = surfaces[id].wetnessTag;
  if ((tag >> 24) != 2u) { return 0; }
  if ((tag & 0x800u) != 0 || (cb.wetnessArgs.roadOnly != 0 && (tag & 0x500u) == 0)) { return 0; }
  bool bypass = (tag & 0x400u) != 0 || cb.wetnessArgs.respectNoRainZones == 0;
  float wet = resolveWeatherWetness(tag, cb.wetnessArgs)
    * (bypass ? 1.f : wetnessExposure(hit));
  return saturate(wet);
}

float puddleAtContact(float3 position, out float3 hit) {
  uint id;
  float wet = wetRoadAtContact(position, id, hit);
  if (wet <= 0) { return 0; }
  // Spawn runs before this frame's AS/material upload. Its TLAS and material
  // buffer agree, but bindless vertex indices already belong to this frame.
  // Recover the local slope from two short neighbouring rays instead of ever
  // dereferencing last-frame surface geometry through the current vertex table.
  float delta = max(particleCb.sceneScale, 1e-4f); // one centimetre
  uint idX, idY;
  float3 hitX, hitY;
  if (!tracePuddleGround(position + float3(delta,0,0), idX, hitX) || idX != id ||
      !tracePuddleGround(position + wetWorldVector(float3(0,delta,0)), idY, hitY) || idY != id) { return 0; }
  float3 normal = normalize(cross(hitX - hit, hitY - hit));
  return wetPuddleCoverage(hit,abs(wetUpNormal(normal)),wet);
}

// Use flags from the SAME completed scene as the TLAS. Never fetch old
// surface vertex indices through the new frame's bindless geometry table.
bool waterAtContact(float3 position, out float3 hit) {
  hit = position;
  if (particleCb.sceneTlasValid == 0) { return false; }
  uint surfaceCount, stride;
  surfaces.GetDimensions(surfaceCount, stride);
  RayDesc ray;
  ray.Origin = position + wetWorldVector(float3(0,0,150.f * particleCb.sceneScale));
  ray.Direction = wetWorldVector(float3(0,0,-1)); ray.TMin = 0;
  ray.TMax = 185.f * particleCb.sceneScale;
  const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, ray);
  while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
      uint id = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CandidateGeometryIndex();
      if (id < surfaceCount && surfaces[id].isAnimatedWater) { query.CommitNonOpaqueTriangleHit(); }
    }
  }
  if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }
  // A solid contact above the water blocks spray, including alpha-tested roads.
  ray.TMax = query.CommittedRayT();
  // Alpha-tested roads still use OBJECT_MASK_OPAQUE but are non-opaque in
  // the BLAS. They must block the ocean underneath just like other roads.
  const uint solidFlags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
    | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<solidFlags> solidQuery;
  solidQuery.TraceRayInline(topLevelAS, solidFlags, OBJECT_MASK_OPAQUE, ray);
  while (solidQuery.Proceed()) {
    if (solidQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
      uint id = (solidQuery.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + solidQuery.CandidateGeometryIndex();
      if (id < surfaceCount && !surfaces[id].isParticle && !surfaces[id].isAnimatedWater) { solidQuery.CommitNonOpaqueTriangleHit(); }
    }
  }
  if (solidQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT) { return false; }
  hit = ray.Origin + ray.Direction * query.CommittedRayT();
  return true;
}

#endif

f16vec3 sampleDirectionInCone(f16vec3 axis, float cosConeHalfAngle, float2 rndUV)
{
  // Sample a direction in a cone about +Z, and get its PDF (we ignore PDF here)
  float solidAnglePdf;
  f16vec3 localDir = calcUniformConeDirectionSample(vec2(rndUV), cosConeHalfAngle, solidAnglePdf);

  f16vec3 tangent, bitangent;
  calcOrthonormalBasis(axis, tangent, bitangent);

  // Rotate the local-Z sample into the world-axis frame
  return localDir.x * tangent + localDir.y * bitangent + localDir.z * axis;
}

// Everything random about one spawn candidate: a point on the emitter mesh, the smear along the emitter's
// motion, and the emission-cone direction. The shelter relocation loop draws an independent one per attempt.
struct SpawnSample
{
  float3 spawnPosition;
  float3 initialVelocity;
  float4 uvMinMax;
  float4 color;
  float  randSeed;
};

SpawnSample sampleSpawnPoint(const GpuSpawnContext spawnCtx, const GpuParticleSystem particleSystem, uint seedBase)
{
  // Get random triangle

  float3 positions[3];
  float3 prevPositions[3];
  float4 color = (spawnCtx.spawnMeshColorsIdx != BINDING_INDEX_INVALID) ? 0.0f : 1.0f;
  float4 uvMinMax = float4(0..xx, 1..xx);

  uint randomSeed = seedBase;

  float3 random01 = float3(randomFloat(uint2(randomSeed++, particleCb.frameIdx)), randomFloat(uint2(randomSeed++, particleCb.frameIdx)), randomFloat(uint2(randomSeed++, particleCb.frameIdx)));
  float3 randomBary = saturate(random01 / dot(random01, 1..xxx));
  
  uint triIdx = randomFloat(uint2(randomSeed++, particleCb.frameIdx)) * spawnCtx.numTriangles;
  
  for (uint i=0 ; i<3 ; i++)
  {
    uint indexIdx = triIdx * 3 + i;
    uint index = spawnCtx.indices32bit ? BUFFER_ARRAY(indices32, (uint)spawnCtx.spawnMeshIndexIdx, indexIdx) : BUFFER_ARRAY(indices, (uint)spawnCtx.spawnMeshIndexIdx, indexIdx);
    
    const uint baseSrcPositionOffset = (spawnCtx.spawnMeshPositionsOffset + index * spawnCtx.spawnMeshPositionsStride) / 4;
    positions[i] = float3(BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPositionsIdx, baseSrcPositionOffset + 0),
                          BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPositionsIdx, baseSrcPositionOffset + 1),
                          BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPositionsIdx, baseSrcPositionOffset + 2));
    
    if (spawnCtx.spawnMeshPrevPositionsIdx != BINDING_INDEX_INVALID)
    {
      prevPositions[i].x = BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPrevPositionsIdx, baseSrcPositionOffset + 0);
      prevPositions[i].y = BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPrevPositionsIdx, baseSrcPositionOffset + 1);
      prevPositions[i].z = BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPrevPositionsIdx, baseSrcPositionOffset + 2);
    }
    else
    {
      prevPositions[i] = positions[i];
    }
    
    if(spawnCtx.spawnMeshColorsIdx != BINDING_INDEX_INVALID)
    {
      const uint baseSrcColorOffset = (spawnCtx.spawnMeshColorsOffset + index * spawnCtx.spawnMeshColorsStride) / 4;
      const uint colorBits = floatBitsToUint(BUFFER_ARRAY(geometries, (uint) spawnCtx.spawnMeshColorsIdx, baseSrcColorOffset));
      // Format is B8G8R8A8_UNORM
      color += unorm4x8ToFloat4x32(colorBits).zyxw * randomBary[i];
    }
    
    // Only fetch emitter UVs when a valid texcoord buffer is bound. For a mesh without UVs,
    // spawnMeshTexcoordsIdx == BINDING_INDEX_INVALID (0xFFFF); reading geometries[0xFFFF] is an
    // out-of-range bindless descriptor access and faults the GPU (device loss). Mirrors the
    // existing guards on the color and prev-position reads above.
    float2 texcoord = float2(0.0f, 0.0f);
    if (spawnCtx.spawnMeshTexcoordsIdx != BINDING_INDEX_INVALID)
    {
      const uint baseSrcTexcoordOffset = (spawnCtx.spawnMeshTexcoordsOffset + index * spawnCtx.spawnMeshTexcoordsStride) / 4;
      texcoord = float2(BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshTexcoordsIdx, baseSrcTexcoordOffset + 0),
                        BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshTexcoordsIdx, baseSrcTexcoordOffset + 1));
    }
    
    if (particleSystem.desc.useSpawnTexcoords && spawnCtx.spawnMeshTexcoordsIdx != BINDING_INDEX_INVALID)
    {
      if (i == 0)
      {
        uvMinMax.xy = texcoord;
        uvMinMax.zw = texcoord;
      }
      else
      {
        uvMinMax.xy = min(uvMinMax.xy, texcoord);
        uvMinMax.zw = max(uvMinMax.zw, texcoord);
      }
    }
  }
  
  // Sample random point on triangle
  const float randSeed = randomFloat(uint2(randomSeed++, particleCb.frameIdx));

  float3 prevWorldPosition = mul(spawnCtx.spawnPrevObjectToWorld, float4(interpolateHitAttribute(prevPositions, randomBary), 1.f)).xyz;
  float3 worldPosition = mul(spawnCtx.spawnObjectToWorld, float4(interpolateHitAttribute(positions, randomBary), 1.f));
  
  float3 worldSpawnVelocity = (worldPosition - prevWorldPosition) * particleCb.invDeltaTimeSecs;
  
  float3 localNormal = cross(normalize(positions[1] - positions[0]), normalize(positions[2] - positions[0]));
  float3 worldNormal = mul(spawnCtx.spawnObjectToWorld, float4(localNormal, 0.f));
  
  float cosCone = cos(radians(particleSystem.desc.initialVelocityConeAngleDegrees));
  f16vec3 dir = sampleDirectionInCone(worldNormal, cosCone, float2(randomFloat(uint2(randomSeed++, particleCb.frameIdx)), randomFloat(uint2(randomSeed++, particleCb.frameIdx))));
  
  SpawnSample result;
  result.initialVelocity = worldSpawnVelocity * particleSystem.desc.initialVelocityFromMotion + dir * particleSystem.desc.initialVelocityFromNormal * particleCb.sceneScale;
  result.spawnPosition = lerp(worldPosition, prevWorldPosition, randSeed);
  result.uvMinMax = uvMinMax;
  result.color = color;
  result.randSeed = randSeed;
  return result;
}

// How many spawn candidates the shelter probe may try before killing a particle.
// With 8, a spawn volume that is 95% covered still lands ~1/3 of its budget in the open 5% (1 - 0.95^8).
static const uint kShelterRelocationAttempts = 8;

// Generate a new particle in-place with random properties
GpuParticle respawn(uint particleIdx, uint spawnCtxIdx, const GpuParticleSystem particleSystem)
{
  const GpuSpawnContext spawnCtx = SpawnContexts[spawnCtxIdx];

  SpawnSample candidate = sampleSpawnPoint(spawnCtx, particleSystem, particleIdx);

  // Occlusion traces against the scene, which unlike the screen-space collision test also see
  // off-screen roofs and overhangs.
  //
  //  1. The shelter probe traces up, against the fall direction, from each candidate. Geometry above
  //     the spawn plane is behind a downward ray's origin, so only an upward probe can see it.
  //     A covered candidate is redrawn, up to kShelterRelocationAttempts times, so the spawn budget
  //     moves to open sky. Only when every candidate is covered is the particle born sleeping
  //     (timeToLive = 0), which is never rendered and retires through the normal counter path.
  //     Relocation keeps the spawn rate, so rain in a doorway can look denser than outside.
  //
  //  2. The landing trace follows the spawn velocity for the whole life and shortens timeToLive so the
  //     particle dies at the surface. It is exact for straight paths and approximate with turbulence or drag.
  //
  // All rays are opaque-only (OBJECT_MASK_OPAQUE and RAY_FLAG_CULL_NON_OPAQUE), so alpha-blended and
  // alpha-tested surfaces do not stop precipitation. Every in-mask candidate is then opaque,
  // so a single Proceed() completes the traversal.
  //
  // topLevelAS is last frame's structure and may not exist early in a scene, hence the sceneTlasValid gate.
  const bool traceOcclusion = particleSystem.desc.traceSpawnOcclusion && particleCb.sceneTlasValid != 0;
  bool sheltered = false;

  if (traceOcclusion)
  {
    InterlockedAdd(SpawnTraceStats[0], 1);

    // The probe reaches 1 km of authored distance (100000 cm times sceneScale), so it still clears
    // high geometry when the scene scale is understated. Sky-bound rays leave the TLAS bounds quickly.
    const float shelterProbeDistance = 100000.0f * particleCb.sceneScale;

    bool relocated = false;
    for (uint attempt = 0; attempt < kShelterRelocationAttempts; ++attempt)
    {
      if (attempt > 0)
      {
        // The seed base strides by maxNumParticles, so a retry never replays another particle's seeds
        // and spawns two drops at the same position.
        candidate = sampleSpawnPoint(spawnCtx, particleSystem,
                                  particleIdx + attempt * particleSystem.desc.maxNumParticles);
      }

      const float speed = length(candidate.initialVelocity);
      if (!(speed > 1e-6f))
      {
        // A motionless particle has no fall direction to probe against, so it is accepted untraced.
        sheltered = false;
        break;
      }

      RayDesc shelterRay;
      shelterRay.Origin = candidate.spawnPosition;
      shelterRay.Direction = -(candidate.initialVelocity / speed);
      shelterRay.TMin = 0.0f;
      shelterRay.TMax = shelterProbeDistance;

      RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE> shelterQuery;
      shelterQuery.TraceRayInline(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE, OBJECT_MASK_OPAQUE, shelterRay);
      shelterQuery.Proceed();

      sheltered = shelterQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
      if (!sheltered)
      {
        relocated = attempt > 0;
        break;
      }
    }

    if (relocated)
    {
      InterlockedAdd(SpawnTraceStats[3], 1);
    }
  }

  // Spawn from the accepted candidate, or from the last covered one, which is killed below.
#ifdef PUDDLE_CONTACT
  float3 contactPosition;
  bool openWaterContact = (particleSystem.desc.spawnOnWaterOrPuddles || particleSystem.desc.spawnOnWetRoads)
    && waterAtContact(candidate.spawnPosition, contactPosition);
  bool contactValid = openWaterContact;
  if (!contactValid) {
    if (particleSystem.desc.spawnOnWetRoads) {
      uint surfaceId;
      float wet = wetRoadAtContact(candidate.spawnPosition, surfaceId, contactPosition);
      contactValid = wet > 0.0f;
      candidate.color.w *= wet;
    } else {
      contactValid = puddleAtContact(candidate.spawnPosition, contactPosition) >= 0.15f;
    }
  }
  if (contactValid) { candidate.spawnPosition = contactPosition + wetWorldVector(float3(0,0,3.f * particleCb.sceneScale)); }
#endif
  GpuParticle newParticle;
  newParticle.reset(particleSystem, candidate.spawnPosition, candidate.initialVelocity, candidate.uvMinMax, candidate.color, candidate.randSeed);

  if (traceOcclusion)
  {
    if (sheltered)
    {
      // Every candidate was covered, so the particle is born sleeping.
      newParticle.timeToLive = 0.0f;
      InterlockedAdd(SpawnTraceStats[1], 1);
    }
    else
    {
      const float speed = length(candidate.initialVelocity);
      if (speed > 1e-6f)
      {
        const float maxDistance = speed * float(newParticle.timeToLive);

        RayDesc ray;
        ray.Origin = candidate.spawnPosition;
        ray.Direction = candidate.initialVelocity / speed;
        ray.TMin = 0.0f;
        ray.TMax = maxDistance;

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE> rayQuery;
        rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE, OBJECT_MASK_OPAQUE, ray);
        rayQuery.Proceed();

        if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
          // Die slightly short of the surface, so the last frame does not intersect it.
          const float hitDistance = rayQuery.CommittedRayT();
          newParticle.timeToLive = min(newParticle.timeToLive, max(hitDistance * 0.98f, 0.0f) / speed);
          InterlockedAdd(SpawnTraceStats[2], 1);
        }
      }
    }
  }

#ifdef PUDDLE_CONTACT
  InterlockedAdd(SpawnTraceStats[4], 1);
  if (contactValid) {
    if (openWaterContact) { InterlockedAdd(SpawnTraceStats[5], 1); }
    else { InterlockedAdd(SpawnTraceStats[6], 1); }
  } else { InterlockedAdd(SpawnTraceStats[7], 1); }
  if (!contactValid) { newParticle.timeToLive = 0.f; }
  if (contactValid && !openWaterContact) {
    newParticle.weatherWetnessAtBirth = half(max(cb.wetnessArgs.contactWetness, 0.0001f));
  }
#endif
  return newParticle;
}


[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint2 threadIdx : SV_DispatchThreadID)
{
  const GpuParticleSystem particleSystem = particleCb.particleSystem;
  
  if(threadIdx.x >= particleSystem.spawnParticleCount)
  {
    return;
  }
  
  const uint particleIdx = (threadIdx.x + particleSystem.spawnParticleOffset) % particleSystem.desc.maxNumParticles;
  const uint spawnCtxIdx = SpawnContextParticleMapping[threadIdx.x];
  const bool shouldRespawn = Particles[particleIdx].isDead() || Particles[particleIdx].isSleeping();
  if (shouldRespawn)
  {
    Particles[particleIdx] = respawn(particleIdx, spawnCtxIdx, particleSystem);
  }
}
