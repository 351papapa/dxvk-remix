//!variant water_caustics.comp
//!variant water_caustics_local.comp
//!> LOCAL_LIGHTS
//!variant water_caustics_solar.comp
//!> WATER_CAUSTICS_SOLAR_PROJECTION
//!variant water_caustics_local_solar.comp
//!> LOCAL_LIGHTS WATER_CAUSTICS_SOLAR_PROJECTION
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
#include "rtx/algorithm/water_caustic_pattern.slangh"
#ifdef WATER_CAUSTICS_SOLAR_PROJECTION
#include "rtx/algorithm/water_caustic_coverage.slangh"
#endif
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
#ifdef WATER_CAUSTICS_SOLAR_PROJECTION
layout(binding = CAUSTICS_SURFACE_INPUT) StructuredBuffer<float> WaterSurfaceInput;
layout(binding = CAUSTICS_ENVELOPE) Sampler2D FocusPattern;
#endif

// Trace toward the water. `ocean` reports whether the surface that was hit is the
// WaterLab ocean material, so the caller can pick the right pattern tiling: ocean
// and authored pools are separate art domains and a pool in a sea shot must not
// inherit the ocean scale.
bool causticWaterHit(float3 position, float3 normal, float3 direction, float maxDistance,
                     out float distance, out bool ocean, out float normalReflectance) {
  distance = 0.0f;
  ocean = false;
  normalReflectance = 0.0f;
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
  if (surface >= cb.surfaceCount || !surfaces[surface].isAnimatedWater) { return false; }
  const MemoryPolymorphicSurfaceMaterial material = surfaceMaterials[surface];
  if ((material.data[0].x & surfaceMaterialTypeMask) != surfaceMaterialTypeTranslucent) { return false; }
  // The packed F0 comes from this water material's IOR, as in native surface shading.
  normalReflectance = float(reinterpret<TranslucentSurfaceMaterial>(material).baseReflectivity);
  if (!isfinite(normalReflectance)) { return false; }
  normalReflectance = saturate(normalReflectance);
  ocean = surfaces[surface].isOceanWater;
  distance = query.CommittedRayT();
  return true;
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
    return float3(waterCausticPatternAt(Pattern, uv, time, lod, args.randomized));
  }

  // Dispersion. Offset along the receiver normal's horizontal projection, scaled by
  // the exposure and a depth ramp, in caustic-tile units. The three taps drive the
  // output R / G / B channels separately, which is what turns each filament into a
  // colour fringe instead of just a blur.
  //
  // Constants follow ImmersiveVC: offset = normalize(normal.xy) * Disp * 0.08 *
  // saturate(depth * 0.5). At the default 0.35 that is ~0.028 of a tile.
  const float2 delta = flowDirection * (dispersion * 0.08f * saturate(depth * 0.5f));
  const float back = waterCausticPatternAt(Pattern, uv - delta, time, lod, args.randomized);
  const float mid = waterCausticPatternAt(Pattern, uv, time, lod, args.randomized);
  const float front = waterCausticPatternAt(Pattern, uv + delta, time, lod, args.randomized);

  // tint has a mean of exactly one, so it only moves the per-channel balance. The
  // overall level is tied to the middle tap, which is what the single-tap path
  // returns, so enabling dispersion cannot change the exposure.
  const float mean = (back + mid + front) * (1.0f / 3.0f);
  const float3 tint = float3(back, mid, front) / max(mean, 1e-4f);
  return mid * tint;
}

#ifdef WATER_CAUSTICS_SOLAR_PROJECTION
float3 causticFocusPattern(float3 waterPosition, float cone, float2 flowDirection, float depth) {
  const float sourceMean = FocusPattern.SampleLevel(float2(0.5f), args.solarPadding.x).r;
  if (!isfinite(sourceMean) || sourceMean <= 0.00001f) {
    return float3(0.0f);
  }
  const float receiverGain = 2.0f * sourceMean;
  const float scale = max(args.scale, 0.0001f);
  const float2 uv = waterPosition.xy / scale;
  const float time = cb.timeSinceStartSeconds * args.speed;
  const float lod = min(max(0.0f, log2(max(cone * args.solarPadding.y / scale, 1.0f))), args.solarPadding.x);
  const float mid = waterCausticFocusAt(FocusPattern, uv, time, lod, args.solarPadding.x, args.randomized);
  if (args.dispersion <= 0.0f) {
    return float3(mid * receiverGain);
  }
  const float2 delta = flowDirection * (args.dispersion * 0.08f * saturate(depth * 0.5f));
  const float back = waterCausticFocusAt(FocusPattern, uv - delta, time, lod, args.solarPadding.x, args.randomized);
  const float front = waterCausticFocusAt(FocusPattern, uv + delta, time, lod, args.solarPadding.x, args.randomized);
  return waterCausticFocusDispersion(back, mid, front) * receiverGain;
}
#endif

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
  const bool ocean = surfaces[surface].isOceanWater;
  float3 waterPosition = ray.Origin + up * query.CommittedRayT();
  const float2 flowDirection = causticFlowDirection(normal);
  float3 pattern;
#ifdef WATER_CAUSTICS_SOLAR_PROJECTION
  if (args.solarProjectionEnabled && ocean && cb.isZUp && args.lightValid && args.solarContrast > 0.0f) {
    float3 projectedPattern = float3(0.0f);
    float projectedDepth;
    float projectionWeight;
    // Both paths share the solar coordinate, while zero contrast retains the vertical web.
    if (waterCausticProject(position, args.lightDirection, args.solarSurfaceHeight,
        waterPosition, projectedDepth, projectionWeight)) {
      projectionWeight *= waterCausticCoverage(WaterSurfaceInput, waterPosition.xy, args.solarSurfaceHeight);
      if (projectionWeight > 0.0f) {
        projectedPattern = causticFocusPattern(waterPosition, cone, flowDirection, depth) * projectionWeight;
      }
    }
    pattern = projectedPattern;
  } else
#endif
  {
    pattern = causticPattern(waterPosition, cone, ocean, flowDirection, depth);
  }
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
  if (args.reflectedStrength <= 0.0f) { return float3(0); }
  float3 toWater = sun - 2.0f * sunHeight * up;
  float distance, normalReflectance;
  bool ocean;
  // Fade against the actual trace reach, including the grazing-angle cap.
  float2 reach = causticReflectedReach(args.reflectedHeight, sunHeight);
  float depthLimit = reach.y;
  if (!causticWaterHit(position, normal, toWater, reach.x, distance, ocean, normalReflectance)) { return float3(0); }
  float strength = args.reflectedStrength * (normalReflectance + (1.0f - normalReflectance) * pow(1.0f - sunHeight, 5.0f));
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
  float3 sunlight = args.lightIrradiance;
  if (useAtmosphere) {
    sunlight = skyProviderSunIrradiance(waterPosition) * cb.preExposure;
  }
  float3 radiance = albedo * sunlight * (cosine * pattern * fade * strength / 3.14159265f);
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
  if (args.reflectedStrength <= 0.0f) { return float3(0); }
  float probeDistance, normalReflectance;
  bool ocean;
  if (!causticWaterHit(position, -up, -up, args.reflectedHeight, probeDistance, ocean, normalReflectance)) { return float3(0); }
  float waterHeight = dot(position, up) - (probeDistance + 0.02f);
  float depthLimit = args.reflectedHeight;
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
      float3 interfacePoint = causticReflectedInterfacePoint(origin, center, up, waterHeight);
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
        interfacePoint = causticReflectedInterfacePoint(origin, lightSample.position, up, waterHeight);
      }
      if (dot(lightSample.position, up) <= waterHeight + 0.02f) { continue; }
      float3 toInterface = interfacePoint - origin;
      float interfaceDistance = length(toInterface);
      if (interfaceDistance < 0.03f) { continue; }
      float3 toWater = toInterface / interfaceDistance;
      float waterDistance;
      // Allow wave displacement around the planar solve, then fade using the actual hit.
      if (!causticWaterHit(position, normal, toWater, interfaceDistance + 1.0f, waterDistance, ocean, normalReflectance)) { continue; }
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
      float strength = args.reflectedStrength * (normalReflectance + (1.0f - normalReflectance) * pow(1.0f - lightHeight, 5.0f));
      float3 contribution = albedo * irradiance
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
