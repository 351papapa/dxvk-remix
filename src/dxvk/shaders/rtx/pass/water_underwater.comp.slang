//!variant water_underwater.comp
//!variant water_underwater_shafts.comp
//!> WATER_LIGHT_SHAFTS
//!variant water_underwater_shafts_scene.comp
//!> WATER_LIGHT_SHAFTS WATER_SCENE_LIGHTS
//!end-variants

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/water_underwater.h"
#include "rtx/concept/camera/camera.slangh"
#include "rtx/algorithm/sky_provider.slangh"

layout(push_constant) ConstantBuffer<WaterUnderwaterArgs> args;
layout(binding = UNDERWATER_DEPTH) Texture2D<float> LinearDepth;
#ifdef WATER_LIGHT_SHAFTS
layout(binding = UNDERWATER_SHAFT_FIELD) RWTexture2D<float4> ShaftField;
#else
layout(binding = UNDERWATER_COLOR) RWTexture2D<float4> Color;
layout(binding = UNDERWATER_SHAFT_FIELD) Texture2D<float4> ShaftField;

float3 upsampleShafts(float2 uv, float sceneDistance) {
  uint width, height;
  ShaftField.GetDimensions(width, height);
  float2 coordinate = uv * float2(width, height) - 0.5f;
  int2 base = int2(floor(coordinate));
  float2 f = frac(coordinate);
  float3 radiance = float3(0.0f);
  float weightSum = 0.0f;
  [unroll]
  for (int y = 0; y < 2; ++y) {
    [unroll]
    for (int x = 0; x < 2; ++x) {
      int2 cell = clamp(base + int2(x, y), int2(0), int2(width - 1, height - 1));
      float4 sample = ShaftField[cell];
      float weight = (x == 0 ? 1.0f - f.x : f.x) * (y == 0 ? 1.0f - f.y : f.y);
      // Reject foreground/background mixing at silhouettes during upsampling.
      weight *= exp(-abs(sample.w - sceneDistance) / max(0.25f, sceneDistance * 0.025f));
      radiance += sample.xyz * weight;
      weightSum += weight;
    }
  }
  return radiance / max(weightSum, 0.00001f);
}
#endif
groupshared float cameraWaterDepth;

#ifdef WATER_LIGHT_SHAFTS
layout(binding = UNDERWATER_PATTERN) Sampler2D Pattern;

// Trace the refracted light path to the nearest surface, then its air segment.
// Rejecting a non-water first hit prevents shafts shining through pool walls.
bool shaftWaterEntry(float3 position, float3 toWater, float limit, out float3 entry, out float lightDistance) {
  entry = position;
  lightDistance = 0.0f;
  RayDesc ray;
  ray.Origin = position;
  ray.Direction = toWater;
  ray.TMin = 0.01f;
  ray.TMax = limit;
  const uint flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, ray);
  while (query.Proceed()) { }
  if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }
  uint index = (query.CommittedInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CommittedGeometryIndex();
  if (!surfaces[index].isAnimatedWater) { return false; }
  lightDistance = query.CommittedRayT();
  entry = position + toWater * lightDistance;
  return true;
}

bool shaftSunVisible(float3 entry, float3 sun, float3 up) {
  RayDesc shadow;
  shadow.Origin = entry + up * 0.05f;
  shadow.Direction = sun;
  shadow.TMin = 0.01f;
  shadow.TMax = 1000.0f;
  const uint flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
    | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE, shadow);
  while (query.Proceed()) { }
  return query.CommittedStatus() != COMMITTED_TRIANGLE_HIT;
}

float shaftPattern(float3 entry, float distance, float patternMean) {
  if (args.patternAvailable == 0u) { return 1.0f; }
  float2 uv = (cb.isZUp != 0 ? entry.xy : entry.xz) / args.shaftScale;
  float time = cb.timeSinceStartSeconds * 0.6f;
  float2 warp = 0.075f * float2(sin(uv.y * 6.2831853f + time), sin(uv.x * 6.2831853f - time * 0.83f));
  // Use a scalar focusing pattern: water absorption supplies the spectral colour.
  float lod = log2(1.0f + distance * 0.08f);
  float a = Pattern.SampleLevel(uv + warp + float2(time * 0.035f, time * 0.021f), lod).r;
  float b = Pattern.SampleLevel(float2(-uv.y, uv.x) - warp + float2(-time * 0.027f, time * 0.017f), lod).r;
  // The DDS is a contrast mask, not a second loss of incoming sunlight. Its
  // coarsest mip supplies the exposure reference; bound focused peaks to avoid white dots.
  float focusing = min(a, b) / max(patternMean, 0.05f);
  return min(focusing * focusing, 4.0f);
}
#ifdef WATER_SCENE_LIGHTS
#include "rtx/pass/water_underwater_scene_lights.slangh"
#endif
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
  float3 up = cb.isZUp != 0 ? float3(0, 0, 1) : float3(0, 1, 0);
  float3 camera = cameraGetWorldPosition(cb.camera);
  if (lane == 0) {
    cameraWaterDepth = 0.0f;
    RayDesc probe;
    probe.Origin = camera;
    probe.Direction = up;
    probe.TMin = 0.001f;
    probe.TMax = 100.0f;
    const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
    RayQuery<flags> query;
    query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, probe);
    while (query.Proceed()) {
      if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
        uint surface = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CandidateGeometryIndex();
        if (surfaces[surface].isAnimatedWater) {
          query.CommitNonOpaqueTriangleHit();
        }
      }
    }
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
      cameraWaterDepth = query.CommittedRayT();
    }
  }
  GroupMemoryBarrierWithGroupSync();
#ifdef WATER_SCENE_LIGHTS
  if (lane == 0u) {
    shaftSelectTileLights(pixel / 8u, camera, up, cameraWaterDepth);
  }
  GroupMemoryBarrierWithGroupSync();
#endif
  if (pixel.x >= args.width || pixel.y >= args.height) { return; }
#ifdef WATER_LIGHT_SHAFTS
  float3 emptyDebug = args.debugMode == 2 ? float3(1, 0, 0)
    : (args.debugMode == 3 ? float3(0, 0, 1) : float3(0));
  ShaftField[pixel] = float4(emptyDebug, 100.0f);
#endif
  if (cameraWaterDepth <= 0.0f) {
#ifndef WATER_LIGHT_SHAFTS
    if (args.debugMode != 0) {
      float3 diagnostic = args.debugMode <= 2 ? float3(1, 0, 0)
        : (args.debugMode == 3 ? float3(0, 0, 1) : float3(0));
      Color[pixel] = float4(diagnostic, 1.0f);
    }
#endif
    return;
  }
#ifndef WATER_LIGHT_SHAFTS
  if (args.debugMode == 1) {
    Color[pixel] = float4(0.0f, 1.0f - saturate(cameraWaterDepth / 10.0f),
      saturate(cameraWaterDepth / 10.0f), 1.0f);
    return;
  }
#endif

  float2 uv = (float2(pixel) + 0.5f) / float2(args.width, args.height);
  uint2 sourcePixel = min(uint2(uv * float2(cb.camera.resolution)), cb.camera.resolution - 1);
  const CameraDirections direction = cameraScreenUVToDirection(cb.camera, uv, false);
  float sceneDistance = min(abs(LinearDepth[sourcePixel]) / max(abs(direction.viewDirection.z), 0.01f), 100.0f);
  if (sceneDistance <= 0.0f) { sceneDistance = 100.0f; }

  float upCosine = dot(direction.worldDirection, up);
  float waterDistance = upCosine > 0.001f ? min(sceneDistance, cameraWaterDepth / upCosine) : sceneDistance;
  waterDistance = max(waterDistance, 0.0f);
  float immersion = smoothstep(0.0f, args.transition, cameraWaterDepth);

#ifndef WATER_LIGHT_SHAFTS
  if (args.debugMode >= 2) {
    Color[pixel] = float4(args.shaftStrength > 0.0f
      ? upsampleShafts(uv, sceneDistance) : float3(1, 0, 1), 1.0f);
    return;
  }
  float3 transmittance = exp(-args.density * waterDistance * float3(2.4f, 0.85f, 0.5f));
  float daylight = cb.skyMode == 1 ? lerp(0.025f, 1.0f, smoothstep(-0.08f, 0.25f, skyProviderSunElevation())) : 1.0f;
  float4 input = Color[pixel];
  float3 fogged = input.xyz * transmittance + args.fogColor * daylight * (1.0f - transmittance);

  if (args.shaftStrength > 0.0f) { fogged += upsampleShafts(uv, sceneDistance); }
  Color[pixel] = float4(lerp(input.xyz, fogged, immersion), input.w);
#else
  float3 volumeRadiance = float3(0.0f);
  uint entryCount = 0;
  uint visibleCount = 0;
#ifdef WATER_SCENE_LIGHTS
  volumeRadiance = shaftIntegrateSceneLights(pixel, camera, direction.worldDirection, up,
    cameraWaterDepth, waterDistance, entryCount, visibleCount);
#else
  if (args.shaftStrength > 0.0f && waterDistance > 0.05f && cb.skyMode == 1) {
    float3 sun = skyProviderSunDirection();
    float sunHeight = dot(sun, up);
    if (sunHeight > 0.01f) {
      float3 tangent = (sun - sunHeight * up) / 1.333f;
      float3 toWater = tangent + up * sqrt(max(0.0f, 1.0f - dot(tangent, tangent)));
      float3 sigmaT = args.density * float3(2.4f, 0.85f, 0.5f);
      float3 sigmaS = args.density * float3(0.22f, 0.30f, 0.36f);
      float g = args.shaftAnisotropy;
      // Haze uses artist-authored radiance. Normalize isotropic phase to 1 so
      // the shaft strength shares that exposure convention instead of losing another 4*pi.
      float phase = (1.0f - g * g)
        / pow(max(0.01f, 1.0f + g * g - 2.0f * g * dot(direction.worldDirection, toWater)), 1.5f);
      float3 surfacePoint = camera + up * cameraWaterDepth;
      float3 sunRadiance = max(float3(0.0f), skyProviderSunIrradiance(surfacePoint) * cb.preExposure);
      float marchDistance = min(waterDistance, args.shaftMaxDistance);
      float3 shaft = float3(0.0f);
      float patternMean = args.patternAvailable != 0u ? Pattern.SampleLevel(float2(0.5f), 16.0f).r : 1.0f;
      // Stable spatial dither avoids coherent sample planes without temporal flicker.
      float jitter = frac(52.9829189f * frac(dot(float2(pixel), float2(0.06711056f, 0.00583715f))));
      [loop]
      for (uint i = 0; i < args.shaftSamples; ++i) {
        float start = float(i) / float(args.shaftSamples);
        float end = float(i + 1) / float(args.shaftSamples);
        start = start * start * marchDistance;
        end = end * end * marchDistance;
        float segmentLength = end - start;
        float rayDistance = lerp(start, end, lerp(0.25f, 0.75f, jitter));
        float3 position = camera + direction.worldDirection * rayDistance;
        float3 entry;
        float lightDistance;
        if (!shaftWaterEntry(position, toWater, args.shaftMaxDistance, entry, lightDistance)) { continue; }
        ++entryCount;
        if (!shaftSunVisible(entry, sun, up)) { continue; }
        ++visibleCount;
        float focusing = shaftPattern(entry, lightDistance, patternMean);
        float3 viewTransmittance = exp(-sigmaT * start);
        float3 lightTransmittance = exp(-sigmaT * lightDistance);
        float3 integratedScatter = sigmaS * (1.0f - exp(-sigmaT * segmentLength)) / max(sigmaT, float3(0.00001f));
        shaft += focusing * viewTransmittance * lightTransmittance * integratedScatter;
      }
      volumeRadiance = shaft * sunRadiance * (phase * args.shaftStrength * 0.98f);
    }
  }
#endif
  if (args.debugMode == 2) {
    float ratio = saturate(float(entryCount) / float(args.shaftSamples));
    volumeRadiance = float3(1.0f - ratio, ratio, 0.0f);
  } else if (args.debugMode == 3) {
    float ratio = float(visibleCount) / float(max(entryCount, 1u));
    volumeRadiance = entryCount > 0 ? float3(1.0f - ratio, ratio, 0.0f) : float3(0, 0, 1);
  }
  ShaftField[pixel] = float4(volumeRadiance, sceneDistance);
#endif
}
