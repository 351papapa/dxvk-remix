//!variant water_foam_update.comp
//!variant water_foam_update_culled.comp
//!>       FOAM_VIEW_CULLING
//!end-variants
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/water_foam_culling.slangh"
#include "rtx/pass/water_foam.h"
#include "rtx/pass/water_surface_args.h"
#include "rtx/pass/water_ocean_wave.slangh"
#include "rtx/pass/water_shore_wave.slangh"
#include "rtx/pass/water_shore_depth.slangh"
#include "rtx/pass/water_beach_mask.slangh"
#include "rtx/concept/camera/camera.slangh"
layout(push_constant) ConstantBuffer<WaterFoamArgs> args;
layout(binding = WATER_FOAM_FIELD) Texture2D<float4> Previous;
layout(binding = WATER_FOAM_NEXT) RWTexture2D<float4> Next;
// xy: water height/validity; zw: static terrain height/validity.
layout(binding = BINDING_WATER_FOAM_SURFACE_HEIGHT) RWTexture2D<float4> NextHeight;
layout(binding = WATER_FOAM_PREVIOUS_HEIGHT) Texture2D<float4> PreviousHeight;
layout(binding = WATER_FOAM_WAVES) StructuredBuffer<float> WaveInput;
layout(binding = WATER_FOAM_BEACH_MASK) StructuredBuffer<uint> BeachMask;
layout(binding = WATER_FOAM_SHORE) StructuredBuffer<float4> ShoreDepth;
#include "rtx/pass/water_foam_common.slangh"
#include "rtx/pass/water_foam_sentinel.slangh"

float4 previousAt(float2 world) {
  float2 p = (world - float2(args.previousX, args.previousY)) / args.spacing;
  if (args.reset != 0 || any(p < 0) || any(p >= float(args.fieldSize - 1))) { return float4(0); }
  int2 c = int2(floor(p)); float2 f = frac(p);
  return lerp(lerp(Previous.Load(int3(c, 0)), Previous.Load(int3(c + int2(1, 0), 0)), f.x),
    lerp(Previous.Load(int3(c + int2(0, 1), 0)), Previous.Load(int3(c + int2(1, 1), 0)), f.x), f.y);
}

float4 previousHeightAt(float2 world) {
  int2 c = int2(round((world - float2(args.previousX, args.previousY)) / args.spacing));
  if (args.reset != 0 || any(c < 0) || any(c >= int(args.fieldSize))) { return float4(0); }
  return PreviousHeight.Load(int3(c, 0));
}

float foamOceanVertexFade(int2 c) {
  uint index = uint(c.y * 61 + c.x);
  if (WaveInput[WATER_WET + index] <= 0.0f) { return 0.0f; }
  float2 p = float2(c) * WaveInput[3];
  float edge = min(min(p.x, p.y), min(WaveInput[12] - p.x, WaveInput[12] - p.y));
  float a = saturate(edge / 9.0f);
  float edgeFade = WaveInput[11] > 0.0f ? WaveInput[WATER_FADE + index] : a*a*(3.0f-2.0f*a);
  float shoreFade = args.shoreHeight > 0.0f ? shoreOceanAttenuation(smoothShoreBed(ShoreDepth,c,false),
    beachSurfaceMask(BeachMask, p + float2(WaveInput[0], WaveInput[1]))) : 1.0f;
  return edgeFade * shoreFade;
}

float foamOceanFade(float2 parameter, out float2 gradient) {
  gradient = float2(0);
  float2 local = parameter - float2(WaveInput[0], WaveInput[1]);
  float spacing = WaveInput[3];
  if (spacing <= 0.0f || any(local < 0.0f) || any(local > 60.0f * spacing)) { return 0.0f; }
  float2 grid = local / spacing;
  int2 cell = min(int2(grid), int2(59));
  float2 f = grid - float2(cell);
  float a = foamOceanVertexFade(cell);
  float b = foamOceanVertexFade(cell + int2(1, 0));
  float c = foamOceanVertexFade(cell + int2(0, 1));
  float d = foamOceanVertexFade(cell + int2(1, 1));
  gradient = float2(lerp(b - a, d - c, f.y), lerp(c - a, d - b, f.x)) / spacing;
  return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

OceanWave foamOceanWave(float2 parameter) {
  float2 local=parameter-float2(WaveInput[0],WaveInput[1])-WaveInput[8];
  float extent=90.0f*WaveInput[9];
  float edge=min(min(local.x,local.y),min(extent-local.x,extent-local.y));
  float fineBlend=smoothstep(0.0f,0.7f,edge);
  float4 k=float4(WaveInput[16],WaveInput[17],WaveInput[18],WaveInput[19]);
  float4 resolved=lerp(oceanGridFilter(k,WaveInput[3]),oceanGridFilter(k,WaveInput[9]),fineBlend);
  return sampleOceanWave(parameter,
    float4(WaveInput[16], WaveInput[17], WaveInput[18], WaveInput[19]),
    float4(WaveInput[20], WaveInput[21], WaveInput[22], WaveInput[23]),
    float4(WaveInput[24], WaveInput[25], WaveInput[26], WaveInput[27]) * args.oceanWaveStrength * resolved, WaveInput[28]);
}

float foamOceanCrest(float2 world) {
  if (args.waveInputAvailable == 0u || args.oceanWaveStrength <= 0.0f) { return 0.0f; }
  float2 parameter = world;
  // The displaced world point is not its Gerstner parameter. Invert the same XY deformation.
  [unroll] for (uint iteration = 0; iteration < 3; ++iteration) {
    OceanWave wave = foamOceanWave(parameter);
    float2 fadeGradient;
    float fade = foamOceanFade(parameter, fadeGradient);
    if (fade <= 0.001f) { return 0.0f; }
    float2 residual = parameter + wave.horizontal * fade - world;
    float j00 = 1.0f - fade * wave.compression.x + wave.horizontal.x * fadeGradient.x;
    float j01 = -fade * wave.compression.y + wave.horizontal.x * fadeGradient.y;
    float j10 = -fade * wave.compression.y + wave.horizontal.y * fadeGradient.x;
    float j11 = 1.0f - fade * wave.compression.z + wave.horizontal.y * fadeGradient.y;
    float determinant = j00 * j11 - j01 * j10;
    if (abs(determinant) > 0.04f) {
      parameter -= float2(j11 * residual.x - j01 * residual.y, j00 * residual.y - j10 * residual.x) / determinant;
    } else {
      parameter = world - wave.horizontal * fade;
    }
  }
  if (!all(isfinite(parameter))) { return 0.0f; }
  OceanWave wave = foamOceanWave(parameter);
  float2 fadeGradient;
  float fade = foamOceanFade(parameter, fadeGradient);
  if (fade <= 0.001f || wave.amplitudeSum <= 1e-5f
      || length(parameter + wave.horizontal * fade - world) > max(0.05f, args.spacing * 0.5f)) { return 0.0f; }
  // Only the fine patch evaluates the continuous ocean; its outer 0.7m still joins coarse triangles.
  float2 fineLocal = parameter - float2(WaveInput[0], WaveInput[1]) - WaveInput[8];
  float fineExtent = 90.0f * WaveInput[9];
  float fineEdge = min(min(fineLocal.x, fineLocal.y), min(fineExtent - fineLocal.x, fineExtent - fineLocal.y));
  float fineSupport = smoothstep(0.7f, 1.4f, fineEdge);
  if (fineSupport <= 0.0f) { return 0.0f; }
  float curvature = max(wave.curvature * fade, 0.0f);
  float crest = smoothstep(args.crestThreshold, args.crestThreshold * 2.5f, curvature);
  float upperWave = smoothstep(0.05f, 0.35f, wave.height / wave.amplitudeSum);
  float crestTop = 1.0f - smoothstep(0.35f, 0.8f, length(wave.gradient) / max(wave.slopeSum, 0.001f));
  float approaching = smoothstep(-0.12f, 0.12f, wave.rise * fade);
  return crest * upperWave * crestTop * approaching * saturate(fade) * fineSupport;
}

bool foamTerrain(float3 p, float3 up, out float terrainHeight, out float3 normal) {
  terrainHeight = 0.0f;
  normal = up;
  RayDesc ray;
  ray.Origin = p + up * 2.0f;
  ray.Direction = -up;
  ray.TMin = 0.0f;
  ray.TMax = 2.0f + max(2.0f, args.runupWidth);
  uint surfaceCount, materialCount, stride;
  surfaces.GetDimensions(surfaceCount, stride);
  surfaceMaterials.GetDimensions(materialCount, stride);
  const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE, ray);
  while (query.Proceed()) {
    if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) { continue; }
    uint id = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CandidateGeometryIndex();
    if (id >= surfaceCount || id >= materialCount) { continue; }
    Surface surface = surfaces[id];
    if (!surface.isStatic || surface.isAnimatedWater || surface.isParticle || surface.isDecal || surface.isPlayerModel) { continue; }
    MemoryPolymorphicSurfaceMaterial material = surfaceMaterials[id];
    if (!material.isOpaque()) { continue; }
    uint wetClass = surface.wetnessTag >> 24;
    if (wetClass == 1u || wetClass == 3u) { continue; }
    query.CommitNonOpaqueTriangleHit();
  }
  if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }
  uint id = (query.CommittedInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CommittedGeometryIndex();
  float3 vertices[3];
  if (!foamTriangleVertices(surfaces[id], query.CommittedPrimitiveIndex(), vertices)) { return false; }
  float3 n = cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
  float nLength = length(n);
  if (nLength < 1e-6f) { return false; }
  n /= nLength;
  if (dot(n, up) < 0.0f) { n = -n; }
  // A height field can carry a beach, but cannot represent a vertical wall.
  if (dot(n, up) < 0.75f) { return false; }
  terrainHeight = dot(ray.Origin, up) - query.CommittedRayT();
  normal = n;
  return isfinite(terrainHeight);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID) {
  if (any(pixel >= args.fieldSize)) { return; }
  Next[pixel] = float4(0);
  NextHeight[pixel] = float4(0);
  if (any(pixel == 0) || any(pixel == args.fieldSize - 1)) { return; }
  float3 up = cb.isZUp != 0 ? float3(0, 0, 1) : float3(0, 1, 0);
  float2 world = float2(args.originX, args.originY) + float2(pixel) * args.spacing;
  float4 current = previousAt(world);
  float4 oldHeight = previousHeightAt(world);
  float waterDecay = exp(-args.dt * 3.0f / max(args.lifetime, 0.1f));
  float wetnessDecay = args.wetnessLifetime > 0.0f ? exp(-args.dt * 3.0f / args.wetnessLifetime) : 0.0f;
#ifdef FOAM_VIEW_CULLING
  const float4x4 worldToClip = mul(cb.camera.viewToProjection, cb.camera.worldToView);
  if (foamTileOutsideView(worldToClip, pixel, float2(args.originX, args.originY),
      args.spacing, dot(cameraGetWorldPosition(cb.camera), up), cb.isZUp != 0)) {
    Next[pixel] = float4(current.x * waterDecay, current.yz * exp(-args.dt * 0.35f), current.w * wetnessDecay);
    // A skipped water sample cannot be used as last frame's height on re-entry.
    oldHeight.y = 0.0f;
    NextHeight[pixel] = oldHeight;
    return;
  }
#endif
  float cameraHeight = dot(cameraGetWorldPosition(cb.camera), up);
  RayDesc ray;
  ray.Origin = cb.isZUp != 0 ? float3(world, cameraHeight + 64.0f) : float3(world.x, cameraHeight + 64.0f, world.y);
  ray.Direction = -up; ray.TMin = 0.0f; ray.TMax = 256.0f;
  uint surfaceCount, surfaceStride;
  surfaces.GetDimensions(surfaceCount, surfaceStride);
  const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, ray);
  while (query.Proceed()) {
    if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) { continue; }
    uint id = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CandidateGeometryIndex();
    if (id < surfaceCount && surfaces[id].isAnimatedWater && (args.oceanOnly == 0 || surfaces[id].isOceanWater)) {
      query.CommitNonOpaqueTriangleHit();
    }
  }
  bool hasWater = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
  bool oceanSurface = false;
  if (hasWater) {
    uint waterId = (query.CommittedInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CommittedGeometryIndex();
    oceanSurface = surfaces[waterId].isOceanWater;
  }
  // Dry wetness remain valid even if the water mesh no longer covers the cell.
  if (!hasWater && (oldHeight.w < 0.5f || current.w < 0.001f)) { return; }
  float waterHeight = hasWater ? cameraHeight + 64.0f - query.CommittedRayT() : oldHeight.x;
  float probeHeight = hasWater ? waterHeight : oldHeight.z;
  float3 p = cb.isZUp != 0 ? float3(world, probeHeight) : float3(world.x, probeHeight, world.y);
  float terrainHeight;
  float3 terrainNormal;
  bool hasTerrain = foamTerrain(p, up, terrainHeight, terrainNormal);
  if (hasTerrain && (terrainHeight < cameraHeight - 192.0f || terrainHeight > cameraHeight + 64.0f)) { hasTerrain = false; }
  NextHeight[pixel] = float4(waterHeight, hasWater ? (oceanSurface ? 2.0f : 1.0f) : 0.0f, terrainHeight, hasTerrain ? 1.0f : 0.0f);
  bool wet = hasWater && (!hasTerrain || waterHeight >= terrainHeight);
  bool sameTerrain = hasTerrain && oldHeight.w > 0.5f && abs(terrainHeight - oldHeight.z) < 0.05f;
  float wetness = sameTerrain ? current.w * wetnessDecay : 0.0f;
  if (!wet) {
    // Land history stays at its world position; it must not advect onto dry land.
    Next[pixel] = float4(0, 0, 0, wetness);
    return;
  }

  float rise = 0.0f;
  if (oldHeight.y > 0.5f && args.dt > 0.0001f && abs(waterHeight - oldHeight.x) < 0.75f) {
    rise = clamp((waterHeight - oldHeight.x) / args.dt, -3.0f, 3.0f);
  }
  float energy = saturate(abs(rise) * args.waveResponse);
  float incoming = saturate(max(rise, 0.0f) * args.waveResponse);
  float slope = length(terrainNormal - up * dot(terrainNormal, up)) / max(dot(terrainNormal, up), 0.001f);
  float depth = hasTerrain ? max(waterHeight - terrainHeight, 0.0f) : 1000.0f;
  float shoreDistance = depth / max(slope, 0.015f);
  float footprint = args.spacing * 0.75f;
  bool beach = hasTerrain && slope > 0.008f && slope < 0.7f && args.runupWidth > 0.0f;
  float beachSupport = beach ? 1.0f - smoothstep(max(args.runupWidth - footprint, 0.0f), args.runupWidth + footprint, shoreDistance) : 0.0f;

  bool probeEnabled = args.sourceEnabled != 0u && args.lineWidth > 0.0f;
  float probeLimit = args.lineWidth + footprint;
  float contact = probeLimit;
  float3 nearest = p, perpendicular, lastPerpendicular = float3(0);
  float3 axisX = float3(1, 0, 0), axisY = cross(up, axisX);
  // The beach already supplies a continuous slope-aware band; walls keep ray contacts.
  if (probeEnabled && beachSupport <= 0.0f) {
    [unroll] for (uint i = 0; i < 8; ++i) {
      float angle = float(i) * 0.7853981634f;
      float3 probe = axisX * cos(angle) + axisY * sin(angle), hit;
      float d = foamContact(p, probe, probeLimit, true, hit, perpendicular);
      if (dot(probe, perpendicular) < 0.999f && dot(lastPerpendicular, perpendicular) < 0.999f) {
        lastPerpendicular = perpendicular;
        float3 refinedHit, unused;
        float refined = foamContact(p, perpendicular, probeLimit, true, refinedHit, unused);
        if (refined < d) { d = refined; hit = refinedHit; }
      }
      if (d < contact) { contact = d; nearest = hit; }
    }
  }
  float contactSupport = contact < probeLimit
    ? 1.0f - smoothstep(max(args.lineWidth - footprint, 0.0f), probeLimit, contact) : 0.0f;
  float2 n = world * 0.22f + float2(args.time * 0.07f, -args.time * 0.05f);
  float2 curl = float2(foamNoise(n + float2(0, 0.1f)).x - foamNoise(n - float2(0, 0.1f)).x,
    -(foamNoise(n + float2(0.1f, 0)).x - foamNoise(n - float2(0.1f, 0)).x)) * 0.5f;
  float4 back = previousAt(world - (current.yz + curl) * args.dt);
  float average = (previousAt(world + float2(args.spacing, 0)).x + previousAt(world - float2(args.spacing, 0)).x
    + previousAt(world + float2(0, args.spacing)).x + previousAt(world - float2(0, args.spacing)).x) * 0.25f;
  float shoreFoam = 0.0f, shoreMask = 0.0f;
  if (oceanSurface && args.waveInputAvailable != 0u && args.crestStrength > 0.0f && args.shoreHeight > 0.0f) {
    float2 parameter = world;
    [unroll] for (uint i = 0; i < 3; ++i) {
      float2 fadeGradient;
      float fade = foamOceanFade(parameter, fadeGradient);
      parameter = world - foamOceanWave(parameter).horizontal * fade;
    }
    float2 local = parameter - float2(WaveInput[0], WaveInput[1]);
    bool fine = all(local >= WaveInput[8]) && all(local <= WaveInput[29]);
    float2 grid = (local - (fine ? WaveInput[8] : 0.0f)) / (fine ? WaveInput[9] : WaveInput[3]);
    if (all(grid >= 0.0f) && all(grid <= (fine ? 90.0f : 60.0f))) {
      int2 cell = min(int2(grid), int2(fine ? 89 : 59));
      float2 f = grid - float2(cell);
      float4 a = smoothShoreBed(ShoreDepth, cell, fine);
      float4 b = smoothShoreBed(ShoreDepth, cell+int2(1,0), fine);
      float4 c = smoothShoreBed(ShoreDepth, cell+int2(0,1), fine);
      float4 d = smoothShoreBed(ShoreDepth, cell+int2(1,1), fine);
      float4 bed = lerp(lerp(a,b,f.x), lerp(c,d,f.x), f.y);
      if (bed.y > 0.99f) {
        ShoreWaveSample shore = sampleShoreWave(bed.x, bed.z, bed.w, parameter, WaveInput[4],
          args.shoreHeight, args.shoreSpacing, args.shorePeriod, args.shoreDepth, beachSurfaceMask(BeachMask, parameter));
        shoreFoam = min(shore.foam * args.crestStrength, 1.6f);
        shoreMask = shore.mask * (1.0f - contactSupport);
      }
    }
  }
  // Open-water whitecaps must fade before a following wave carries their history into a trough.
  float densityLifetime = max(beachSupport, contactSupport) > 0.0f ? args.lifetime : min(args.lifetime, 2.0f);
  float densityDecay = exp(-args.dt * 3.0f / max(densityLifetime, 0.1f));
  float density = max(0.0f, lerp(back.x, average, min(args.dt * 0.4f, 1.0f))) * densityDecay;
  if (probeEnabled) {
    float source = beachSupport * incoming * 3.0f * (1.0f-shoreMask) + contactSupport * (0.02f + energy * 1.8f);
    density = min(density + source * args.dt, 1.6f);
  }
  // Analytic crests share the submitted ocean phase; triangle edges never seed whitecaps.
  if (oceanSurface && args.crestStrength > 0.0f && args.dt > 0.0f) {
    density = min(density + 2.0f * foamOceanCrest(world) * args.crestStrength * args.dt, 1.6f);
  }
  // Phase foam owns eligible sandy surf. Keep contact foam outside that mask,
  // rather than accumulating a permanent white border underneath each breaker.
  density = max(density * (1.0f - shoreMask), shoreFoam * (1.0f - contactSupport));
  // Wetness is coverage history, not foam density. It remains fixed on the bed.
  if (hasTerrain && depth < max(args.runupWidth, 0.5f) && args.wetnessLifetime > 0.0f && args.dt > 0.0f) {
    wetness = 1.0f;
  }
  float3 offset = p - nearest; offset -= up * dot(offset, up);
  float2 outward = cb.isZUp != 0 ? offset.xy : offset.xz;
  outward = length(outward) > 1e-4f ? normalize(outward) : float2(0);
  float3 uphill3 = up * dot(terrainNormal, up) - terrainNormal;
  float2 uphill = cb.isZUp != 0 ? uphill3.xy : uphill3.xz;
  uphill = length(uphill) > 1e-4f ? normalize(uphill) : float2(0);
  float2 targetVelocity = beachSupport > 0.0f
    ? uphill * clamp(rise / max(slope, 0.05f), -1.5f, 1.5f)
    : outward * energy * 0.25f;
  float support = max(beachSupport, contactSupport);
  float2 velocity = lerp(back.yz * exp(-args.dt * 0.7f), targetVelocity, min(args.dt * 3.0f, 1.0f) * support);
  Next[pixel] = float4(density, velocity, wetness);
}
