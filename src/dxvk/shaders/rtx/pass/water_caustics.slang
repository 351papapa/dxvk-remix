//!variant water_caustics.comp
//!variant water_caustics_local.comp
//!> LOCAL_LIGHTS
//!variant water_caustics_composite.comp
//!> COMPOSITE
//!end-variants

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/water_caustics.h"
#include "rtx/utility/geometry_flags.slangh"
#include "rtx/pass/sparse_rendering/sparse_rendering.slangh"
#include "rtx/pass/sparse_rendering/compact_active_pixels.slangh"

layout(push_constant) ConstantBuffer<WaterCausticsArgs> args;
layout(binding = CAUSTICS_ACTIVE_PIXELS) Texture2D<uint16_t> ActivePixels;
layout(binding = CAUSTICS_CACHE) RWTexture2D<float4> CausticsCache;

#ifdef COMPOSITE
layout(binding = CAUSTICS_PRIMARY_RADIANCE) RWTexture2D<float4> PrimaryRadiance;
layout(binding = CAUSTICS_SECONDARY_RADIANCE) RWTexture2D<float4> SecondaryRadiance;
#else
#include "rtx/utility/packing.slangh"
#include "rtx/algorithm/sky_provider.slangh"
#include "rtx/pass/water_caustics_math.slangh"
#ifdef LOCAL_LIGHTS
#include "rtx/concept/light/light.slangh"
#endif

layout(binding = CAUSTICS_PRIMARY_POSITION) Texture2D<float4> PrimaryPosition;
layout(binding = CAUSTICS_SECONDARY_POSITION) Texture2D<float4> SecondaryPosition;
layout(binding = CAUSTICS_PRIMARY_NORMAL) Texture2D<uint> PrimaryNormal;
layout(binding = CAUSTICS_SECONDARY_NORMAL) Texture2D<uint> SecondaryNormal;
layout(binding = CAUSTICS_PRIMARY_ALBEDO) Texture2D<float4> PrimaryAlbedo;
layout(binding = CAUSTICS_SECONDARY_ALBEDO) Texture2D<float4> SecondaryAlbedo;
layout(binding = CAUSTICS_FLAGS) Texture2D<uint16_t> Flags;
layout(binding = CAUSTICS_MATERIAL) Texture2D<uint> Material;
layout(binding = CAUSTICS_PDF) Texture2D<float> SurfacePdf;
layout(binding = CAUSTICS_PRIMARY_CONE) Texture2D<float> PrimaryCone;
layout(binding = CAUSTICS_SECONDARY_CONE) Texture2D<float> SecondaryCone;
layout(binding = CAUSTICS_PATTERN) Sampler2D Pattern;

float2 causticHash(float2 p) {
  float3 p3 = frac(float3(p.x, p.y, p.x) * float3(0.1031f, 0.1030f, 0.0973f));
  p3 += dot(p3, p3.yzx + 33.33f);
  return frac((p3.xx + p3.yz) * p3.zy);
}

float2 causticNoise(float2 p) {
  float2 cell = floor(p);
  float2 f = frac(p);
  f = f * f * (3.0f - 2.0f * f);
  return lerp(lerp(causticHash(cell), causticHash(cell + float2(1, 0)), f.x),
              lerp(causticHash(cell + float2(0, 1)), causticHash(cell + float2(1, 1)), f.x), f.y) * 2.0f - 1.0f;
}

float causticTile(float2 uv, float2 cell, float lod) {
  float2 random = causticHash(cell);
  float angle = random.x * 6.2831853f;
  float s = sin(angle), c = cos(angle);
  float2 rotated = float2(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
  return Pattern.SampleLevel(rotated + random * 13.0f, lod).r;
}

float causticStochastic(float2 uv, float lod) {
  float2 grid = float2(uv.x - uv.y * 0.57735027f, uv.y * 1.15470054f);
  float2 cell = floor(grid), f = frac(grid);
  float2 v0, v1, v2;
  float3 weights;
  if (f.x + f.y < 1.0f) {
    v0 = cell; v1 = cell + float2(1, 0); v2 = cell + float2(0, 1);
    weights = float3(1.0f - f.x - f.y, f.x, f.y);
  } else {
    v0 = cell + 1.0f; v1 = cell + float2(0, 1); v2 = cell + float2(1, 0);
    weights = float3(f.x + f.y - 1.0f, 1.0f - f.x, 1.0f - f.y);
  }
  weights = weights * weights * weights;
  weights /= max(dot(weights, float3(1)), 0.0001f);
  float3 samples = float3(causticTile(uv, v0, lod), causticTile(uv, v1, lod), causticTile(uv, v2, lod));
  // RMS blending retains bright filaments better than a linear crossfade.
  return sqrt(dot(weights, samples * samples));
}

// Trace toward the water. `ocean` reports whether the surface that was hit is the
// WaterLab ocean material, so the caller can pick the right pattern tiling: ocean
// and authored pools are separate art domains and a pool in a sea shot must not
// inherit the ocean scale.
bool causticWaterHit(float3 position, float3 normal, float3 direction, float maxDistance,
                     out float distance, out bool ocean) {
  distance = 0.0f;
  ocean = false;
  if (dot(normal, direction) <= 0.001f) { return false; }
  RayDesc ray;
  ray.Origin = position + normal * 0.02f;
  ray.Direction = direction;
  ray.TMin = 0.01f;
  ray.TMax = maxDistance;
  const uint flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, ray);
  while (query.Proceed()) { }
  if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }
  uint surface = (query.CommittedInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CommittedGeometryIndex();
  if (!surfaces[surface].isAnimatedWater) { return false; }
  ocean = surfaces[surface].isOceanWater;
  distance = query.CommittedRayT();
  return true;
}


// One evaluation of the caustic web at a pattern-space coordinate. `time` and `lod`
// are hoisted to the caller so the three dispersion taps can share them.
float causticPatternAt(float2 uv, float time, float lod) {
  float2 warp = 0.075f * float2(sin(uv.y * 6.2831853f + time), sin(uv.x * 6.2831853f - time * 0.83f));
  float a, b;
  if (args.randomized != 0) {
    warp = 0.28f * causticNoise(uv * 0.7f + float2(time * 0.09f, -time * 0.061f))
      + 0.09f * causticNoise(uv * 1.73f + float2(-time * 0.13f, time * 0.077f));
    a = causticStochastic(uv + warp + float2(time * 0.021f, time * 0.009f), lod);
    b = causticStochastic(float2(-uv.y, uv.x) * 1.137f - warp + float2(-time * 0.017f, time * 0.013f), lod);
  } else {
    a = Pattern.SampleLevel(uv + warp + float2(time * 0.035f, time * 0.021f), lod).r;
    b = Pattern.SampleLevel(float2(-uv.y, uv.x) - warp + float2(-time * 0.027f, time * 0.017f), lod).r;
  }
  return 2.0f * min(a * a, b * b);
}

// Horizontal projection of a receiver normal, in the same plane the pattern is
// sampled in. The dispersion taps slide along this direction.
float2 causticFlowDirection(float3 normal) {
  const float2 horizontal = cb.isZUp != 0 ? normal.xy : float2(normal.x, normal.z);
  const float length = sqrt(dot(horizontal, horizontal));
  return length > 1e-4f ? horizontal / length : float2(0.0f, 1.0f);
}

float3 causticPattern(float3 waterPosition, float cone, bool ocean, float2 flowDirection, float depth) {
  // Pick the tiling from the water surface actually hit. The CPU writes the ocean
  // value into poolScale when the per-domain split is switched off, so this needs
  // no extra branch for the disabled case.
  const float scale = ocean ? args.scale : args.poolScale;
  const float2 uv = (cb.isZUp != 0 ? waterPosition.xy : waterPosition.xz) / scale;
  const float time = cb.timeSinceStartSeconds * args.speed;
  const float lod = max(0.0f, log2(max(cone * 64.0f / scale, 1.0f)));

  const float dispersion = args.dispersion;
  if (dispersion <= 0.0f) {
    return float3(causticPatternAt(uv, time, lod));
  }

  // Dispersion is a surface-transmission effect: the fringes come from the colour
  // splitting as light crosses the water surface. It only exists when the caustics are
  // seen *through* that surface, which means the camera has to be above the water.
  // Underwater the eye never crosses the surface and ImmersiveVC shows no dispersion
  // either - its dispersion code lives in the water-surface shader, not in the
  // underwater post pass.
  const float3 cameraPosition = transpose(cb.camera.viewToWorld)[3].xyz;
  const float3 up = cb.isZUp != 0 ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 1.0f, 0.0f);
  if (dot(cameraPosition - waterPosition, up) <= 0.0f) {
    return float3(causticPatternAt(uv, time, lod));
  }

  // Dispersion. Offset along the receiver normal's horizontal projection, scaled by
  // the exposure and a depth ramp, in caustic-tile units. The three taps drive the
  // output R / G / B channels separately, which is what turns each filament into a
  // colour fringe instead of just a blur.
  //
  // Constants follow ImmersiveVC: offset = normalize(normal.xy) * Disp * 0.08 *
  // saturate(depth * 0.5). At the default 0.35 that is ~0.028 of a tile.
  const float2 delta = flowDirection * (dispersion * 0.08f * saturate(depth * 0.5f));
  const float back = causticPatternAt(uv - delta, time, lod);
  const float mid = causticPatternAt(uv, time, lod);
  const float front = causticPatternAt(uv + delta, time, lod);

  // tint has a mean of exactly one, so it only moves the per-channel balance. The
  // overall level is tied to the middle tap, which is what the single-tap path
  // returns, so enabling dispersion cannot change the exposure.
  const float mean = (back + mid + front) * (1.0f / 3.0f);
  const float3 tint = float3(back, mid, front) / max(mean, 1e-4f);
  return mid * tint;
}

// Transmitted caustics modulate the existing direct-light solution, including its shadows.
bool evaluateTransmittedCaustics(uint2 pixel, bool primary, out float3 modulation) {
  modulation = float3(0);
  if (args.strength <= 0.0f || ((Material[pixel] >> 30) & surfaceMaterialTypeMask) != surfaceMaterialTypeOpaque) { return false; }
  float cone = abs(primary ? PrimaryCone[pixel] : SecondaryCone[pixel]);
  if (cone <= 0.0f) { return false; }
  float3 position = primary ? PrimaryPosition[pixel].xyz : SecondaryPosition[pixel].xyz;
  float3 up = cb.isZUp != 0 ? float3(0, 0, 1) : float3(0, 1, 0);
  RayDesc ray;
  ray.Origin = position + up * 0.02f;
  ray.Direction = up;
  ray.TMin = 0.01f;
  ray.TMax = args.maxDepth;
  const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, ray);
  while (query.Proceed()) {
    if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) { continue; }
    uint surface = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CandidateGeometryIndex();
    if (surfaces[surface].isAnimatedWater) { query.CommitNonOpaqueTriangleHit(); }
  }
  if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }
  uint surface = (query.CommittedInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CommittedGeometryIndex();
  float depth = query.CommittedRayT() + 0.02f;
  // Do not focus lights that never cross this interface (for example pool-floor lamps).
  bool sourceAboveWater = args.lightValid != 0u;
#ifdef LOCAL_LIGHTS
  float3 cameraPosition = transpose(cb.camera.viewToWorld)[3].xyz;
  if (length(position - cameraPosition) < args.localLightRange) {
    for (uint i = 0u; i < min(args.localLightCount, 8u) && !sourceAboveWater; ++i) {
      uint lightIndex = args.localLightIndices[i];
      if (lightIndex >= cb.lightCount) { continue; }
      DecodedPolymorphicLight light = decodePolymorphicLight(lights[lightIndex]);
      float3 center = uintBitsToFloat(light.data0.xyz);
      sourceAboveWater = uint(light.polymorphicType) < lightTypeDistant && any(light.radiance > 0.0f)
        && dot(center - position, up) > depth && length(center - cameraPosition) < args.localLightRange;
    }
  }
#endif
  if (!sourceAboveWater) { return false; }
  float3 normal = signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(primary ? PrimaryNormal[pixel] : SecondaryNormal[pixel]));
  float3 pattern = causticPattern(ray.Origin + up * query.CommittedRayT(), cone,
    surfaces[surface].isOceanWater, causticFlowDirection(normal), depth);
  modulation = pattern * (args.strength * causticDepthFade(depth, args.maxDepth, args.fadeStart));
  return true;
}

float3 evaluateDirectionalCaustics(uint2 pixel, bool primary) {
  if (((Material[pixel] >> 30) & surfaceMaterialTypeMask) != surfaceMaterialTypeOpaque) {
    return float3(0);
  }
  // The sign encodes a serialized PSR view direction, not a GBuffer miss.
  float cone = abs(primary ? PrimaryCone[pixel] : SecondaryCone[pixel]);
  bool useAtmosphere = cb.skyMode == 1;
  if (cone <= 0.0f || (useAtmosphere ? skyProviderSunElevation() <= 0.0f : args.lightValid == 0)) {
    return float3(0);
  }
  float3 position = primary ? PrimaryPosition[pixel].xyz : SecondaryPosition[pixel].xyz;
  float3 normal = signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(primary ? PrimaryNormal[pixel] : SecondaryNormal[pixel]));
  float3 albedo = primary ? PrimaryAlbedo[pixel].xyz : SecondaryAlbedo[pixel].xyz;
  float3 up = cb.isZUp != 0 ? float3(0, 0, 1) : float3(0, 1, 0);
  float3 sun = args.lightDirection;
  if (useAtmosphere) {
    sun = skyProviderSunDirection();
  }
  float sunHeight = dot(sun, up);
  if (sunHeight <= 0.0f || max(albedo.x, max(albedo.y, albedo.z)) <= 0.0f) {
    return float3(0);
  }
  float3 tangent = (sun - sunHeight * up) / 1.333f;
  float3 toWater = tangent + up * sqrt(max(0.0f, 1.0f - dot(tangent, tangent)));
  float distance;
  bool reflected = false;
  bool ocean = false;
  float depthLimit = args.maxDepth;
  float strength = args.strength;
  bool hitWater = false;
  if (!hitWater && args.reflectedStrength > 0.0f) {
    reflected = true;
    toWater = sun - 2.0f * sunHeight * up;
    depthLimit = args.reflectedHeight;
    strength = args.reflectedStrength * (0.08f + 0.92f * pow(1.0f - sunHeight, 5.0f));
    // Fade against the actual trace reach, including the grazing-angle cap.
    float2 reach = causticReflectedReach(depthLimit, sunHeight);
    depthLimit = reach.y;
    hitWater = causticWaterHit(position, normal, toWater, reach.x, distance, ocean);
  }
  if (!hitWater) { return float3(0); }
  float cosine = saturate(dot(normal, toWater));
  float depth = distance * abs(dot(toWater, up));
  float3 waterPosition = position + normal * 0.02f + distance * toWater;
  const uint rayFlags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;

  RayDesc shadow;
  shadow.Origin = waterPosition + up * 0.05f;
  shadow.Direction = sun;
  shadow.TMin = 0.02f;
  shadow.TMax = 10000.0f;
  const uint shadowFlags = rayFlags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
  RayQuery<shadowFlags> shadowQuery;
  shadowQuery.TraceRayInline(topLevelAS, shadowFlags, OBJECT_MASK_OPAQUE, shadow);
  while (shadowQuery.Proceed()) { }
  if (shadowQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT) { return float3(0); }

  float3 pattern = causticPattern(waterPosition, cone, ocean, causticFlowDirection(normal), depth);
  float fade = causticDepthFade(depth, depthLimit, args.fadeStart);
  float3 attenuation = reflected ? float3(1) : pow(clamp(args.transmittance, float3(0.001f), float3(1.0f)), float3(distance / args.transmittanceDistance));
  float3 sunlight = args.lightIrradiance;
  if (useAtmosphere) {
    sunlight = skyProviderSunIrradiance(waterPosition) * cb.preExposure;
  }
  float3 radiance = albedo * sunlight * attenuation * (cosine * pattern * fade * strength / 3.14159265f);
  radiance *= smoothstep(0.0f, 0.087f, sunHeight) / max(SurfacePdf[pixel], 0.001f);
  return max(radiance, float3(0));
}
#ifdef LOCAL_LIGHTS
float3 evaluateLocalCaustics(uint2 pixel, bool primary) {
  if (((Material[pixel] >> 30) & surfaceMaterialTypeMask) != surfaceMaterialTypeOpaque) { return float3(0); }
  float cone = abs(primary ? PrimaryCone[pixel] : SecondaryCone[pixel]);
  if (cone <= 0.0f) { return float3(0); }
  float3 position = primary ? PrimaryPosition[pixel].xyz : SecondaryPosition[pixel].xyz;
  float3 cameraPosition = transpose(cb.camera.viewToWorld)[3].xyz;
  float range = args.localLightRange;
  float receiverFade = 1.0f - smoothstep(range * 0.75f, range, length(position - cameraPosition));
  if (receiverFade <= 0.0f) { return float3(0); }
  float3 normal = signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(primary ? PrimaryNormal[pixel] : SecondaryNormal[pixel]));
  float3 albedo = primary ? PrimaryAlbedo[pixel].xyz : SecondaryAlbedo[pixel].xyz;
  if (max(albedo.x, max(albedo.y, albedo.z)) <= 0.0f) { return float3(0); }
  float3 up = cb.isZUp != 0 ? float3(0, 0, 1) : float3(0, 1, 0);
  float probeDistance;
  bool reflected = false;
  bool ocean = false;
  bool hit = false;
  if (!hit && args.reflectedStrength > 0.0f) {
    reflected = true;
    hit = causticWaterHit(position, -up, -up, args.reflectedHeight, probeDistance, ocean);
  }
  if (!hit) { return float3(0); }
  float waterHeight = dot(position, up) + (reflected ? -1.0f : 1.0f) * (probeDistance + 0.02f);
  float depthLimit = reflected ? args.reflectedHeight : args.maxDepth;
  float3 origin = position + normal * 0.02f;
  float3 result = float3(0);
  for (uint i = 0; i < min(args.localLightCount, 8u); ++i) {
    uint index = args.localLightIndices[i];
    if (index >= cb.lightCount) { continue; }
    DecodedPolymorphicLight light = decodePolymorphicLight(lights[index]);
    if (light.polymorphicType >= lightTypeDistant) { continue; }
    float3 center = uintBitsToFloat(light.data0.xyz);
    if (dot(center, up) <= waterHeight + 0.02f) { continue; }
    float sourceFade = 1.0f - smoothstep(range * 0.75f, range, length(center - cameraPosition));
    if (sourceFade <= 0.0f) { continue; }
    // Two fixed emitter samples keep this undenoised pass temporally stable.
    for (uint sampleIndex = 0; sampleIndex < 2; ++sampleIndex) {
      float2 uv = float2(sampleIndex == 0 ? 0.25f : 0.75f, 0.5f);
      float3 interfacePoint = causticInterfacePoint(origin, center, up, waterHeight, reflected);
      if (light.polymorphicType == lightTypeCylinder) {
        CylinderLight cylinder = cylinderLightCreate(light);
        float3 tangent, bitangent;
        calcOrthonormalBasis(cylinder.axis, tangent, bitangent);
        float3 outward = interfacePoint - center;
        float angle = atan2(dot(outward, bitangent), dot(outward, tangent));
        uv.x = frac(angle / 6.2831853f + (sampleIndex == 0 ? -0.125f : 0.125f));
      }
      VolumeInteraction interaction = (VolumeInteraction)0;
      LightSample lightSample = (LightSample)0;
      for (uint iteration = 0; iteration < 2; ++iteration) {
        interaction.position = interfacePoint;
        lightSample = decodedPolymorphicLightSampleArea(light, uv, interaction);
        interfacePoint = causticInterfacePoint(origin, lightSample.position, up, waterHeight, reflected);
      }
      if (dot(lightSample.position, up) <= waterHeight + 0.02f) { continue; }
      float3 toInterface = interfacePoint - origin;
      float interfaceDistance = length(toInterface);
      if (interfaceDistance < 0.03f) { continue; }
      float3 toWater = toInterface / interfaceDistance;
      float waterDistance;
      // Allow wave displacement around the planar solve, then fade using the actual hit.
      if (!causticWaterHit(position, normal, toWater, interfaceDistance + 1.0f, waterDistance, ocean)) { continue; }
      float3 waterPosition = origin + toWater * waterDistance;
      float depth = abs(dot(waterPosition - origin, up));
      float fade = causticDepthFade(depth, depthLimit, args.fadeStart);
      if (fade <= 0.0f) { continue; }
      interaction.position = waterPosition;
      lightSample = decodedPolymorphicLightSampleArea(light, uv, interaction);
      if (lightSample.solidAnglePdf <= 0.0f || !isfinite(lightSample.solidAnglePdf)) { continue; }
      float3 toLight = lightSample.position - waterPosition;
      float lightDistance = length(toLight);
      if (lightDistance < 0.1f) { continue; }
      toLight /= lightDistance;
      float lightHeight = dot(toLight, up);
      if (lightHeight <= 0.0f) { continue; }
      float3 irradiance = lightSample.radiance / lightSample.solidAnglePdf;
      if (!all(isfinite(irradiance)) || max(irradiance.x, max(irradiance.y, irradiance.z)) <= 0.0f) { continue; }
      RayDesc shadow;
      shadow.Origin = waterPosition + up * 0.02f;
      float3 shadowDelta = lightSample.position - shadow.Origin;
      float shadowLength = length(shadowDelta);
      shadow.Direction = shadowDelta / shadowLength;
      shadow.TMin = 0.01f;
      shadow.TMax = shadowLength - 0.03f;
      const uint shadowFlags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
      RayQuery<shadowFlags> query;
      query.TraceRayInline(topLevelAS, shadowFlags, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, shadow);
      while (query.Proceed()) { }
      if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) { continue; }
      float strength = reflected ? args.reflectedStrength * (0.08f + 0.92f * pow(1.0f - lightHeight, 5.0f)) : args.strength;
      float3 attenuation = reflected ? float3(1) : pow(clamp(args.transmittance, float3(0.001f), float3(1)), float3(waterDistance / args.transmittanceDistance));
      float3 contribution = albedo * irradiance * attenuation
        * (saturate(dot(normal, toWater)) * causticPattern(waterPosition, cone, ocean, causticFlowDirection(normal), depth) * fade * strength / 3.14159265f);
      contribution *= receiverFade * sourceFade * smoothstep(0.0f, 0.087f, lightHeight)
        / max(SurfacePdf[pixel], 0.001f);
      if (all(isfinite(contribution))) { result += max(contribution, float3(0)) * 0.5f; }
    }
  }
  return result;
}
#endif
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 dispatchThreadId : SV_DispatchThreadID) {
  if (any(dispatchThreadId >= cb.camera.resolution)) { return; }
  uint2 pixel = dispatchThreadId;
  if (!loadPixelCoordinate(dispatchThreadId, ActivePixels, cb.camera.resolution, cb.sparseRenderingArgs, pixel)) { return; }
#ifdef COMPOSITE
  float4 contribution = CausticsCache[pixel];
  if (contribution.w == 1.0f || contribution.w == -1.0f) {
    float4 value = PrimaryRadiance[pixel];
    PrimaryRadiance[pixel] = float4(contribution.w < 0.0f ? value.xyz * (float3(1) + contribution.xyz) : value.xyz + contribution.xyz, value.w);
  } else {
    float4 value = SecondaryRadiance[pixel];
    SecondaryRadiance[pixel] = float4(contribution.w < 0.0f ? value.xyz * (float3(1) + contribution.xyz) : value.xyz + contribution.xyz, value.w);
  }
#else
  bool primary = geometryFlagsReadFromGBuffer(pixel, Flags).primarySelectedIntegrationSurface;
  float3 modulation;
  if (evaluateTransmittedCaustics(pixel, primary, modulation)) {
    CausticsCache[pixel] = float4(modulation, primary ? -1.0f : -2.0f);
    return;
  }
  CausticsCache[pixel] = float4(evaluateDirectionalCaustics(pixel, primary)
#ifdef LOCAL_LIGHTS
    + evaluateLocalCaustics(pixel, primary)
#endif
    , primary ? 1.0f : 0.0f);
#endif
}
