//!variant sun_flare_state.comp
//!end-variants

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/sun_flare.h"
#include "rtx/concept/camera/camera.slangh"
#include "rtx/algorithm/visibility.slangh"
#include "rtx/algorithm/sky_provider.slangh"

layout(push_constant) ConstantBuffer<SunFlareStateArgs> args;
layout(binding = SUN_FLARE_STATE) RWTexture2D<float4> FlareState;

float probeCameraWaterDepth(float3 camera, float3 up) {
  RayDesc probe;
  probe.Origin = camera;
  probe.Direction = up;
  probe.TMin = 0.001f;
  probe.TMax = max(args.waterProbeDistance, 0.01f);
  const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, probe);
  while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
      const uint surface = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK)
        + query.CandidateGeometryIndex();
      if (surfaces[surface].isAnimatedWater) {
        query.CommitNonOpaqueTriangleHit();
      }
    }
  }
  return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? query.CommittedRayT() : 0.0f;
}

float3 sunVisibility(float3 eye, float3 sun, float angularRadius) {
  const uint count = clamp(args.visibilitySamples, 1u, 16u);
  const float3 reference = abs(sun.y) < 0.95f ? float3(0, 1, 0) : float3(1, 0, 0);
  const float3 tangent = normalize(cross(reference, sun));
  const float3 bitangent = cross(sun, tangent);
  float3 attenuation = float3(0.0f);
  for (uint i = 0; i < count; ++i) {
    const float radius = count > 1u ? sqrt((float(i) + 0.5f) / float(count)) : 0.0f;
    const float angle = float(i) * 2.39996323f;
    const float3 direction = normalize(sun + tan(angularRadius) * radius
      * (cos(angle) * tangent + sin(angle) * bitangent));
    Ray ray;
    ray.origin = eye + direction * 0.005f;
    ray.direction = direction;
    ray.tMax = max(args.maxRayDistance, 1.0f);
    ray.coneRadius = 0.0f;
    ray.spreadAngle = 0.0f;
    const uint8_t mask = uint8_t(OBJECT_MASK_OPAQUE | OBJECT_MASK_ALPHA_BLEND | OBJECT_MASK_TRANSLUCENT);
    const VisibilityResult result = traceVisibilityRay<visibilityModeDisableCulling
      | visibilityModeEnableTranslucentMaterials>(ray, mask, invalidRayPortalIndex, PORTAL_SPACE_NONE);
    attenuation += saturate(result.attenuation);
  }
  return attenuation / float(count);
}

float sunCloudTransmittance(float3 eye, float3 sun, float3 sunYUp) {
  if (!args.cloudOcclusion || skyArgs.cloudEnabled <= 0.0f) {
    return 1.0f;
  }
  if (skyArgs.cloudSecondaryLutEnable) {
    return saturate(sampleCloudDomeLut(AtmosphereCloudSecondaryLut, AtmosphereSkyViewSampler,
      cloudDomeDirToUv(sunYUp), 0.0f).w);
  }
  return saturate(skyProviderCloudShadow(eye, sun));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId != uint3(0, 0, 0))) {
    return;
  }
  const float4 previous = args.resetHistory ? float4(0.5f, 0.5f, 0.0f, 0.0f)
    : FlareState[uint2(0, 0)];
  const float previousRecovery = args.resetHistory ? 0.0f
    : max(FlareState[uint2(1, 0)].w - 1.0f, 0.0f);
  const float3 eye = cameraGetWorldPosition(cb.camera);
  const float3 up = cb.isZUp ? float3(0, 0, 1) : float3(0, 1, 0);
  // This is independent of the water color/haze dispatch and of its master toggle.
  float waterDepth = args.suppressUnderwater && args.cameraMediumState == 0u
    ? probeCameraWaterDepth(eye, up) : 0.0f;
  if (args.suppressUnderwater && (args.cameraMediumState == 2u || cb.cameraWaterVolumeActive)) {
    const float configuredDepth = cb.cameraWaterSurfaceValid && cb.isZUp
      ? cb.cameraWaterSurfaceHeight - eye.z : 0.001f;
    waterDepth = max(waterDepth, max(configuredDepth, 0.001f));
  }
  const float3 atmosphericSun = skyArgs.sunDirection;
  float3 sun = skyProviderSunDirection();
  const float sunLength = length(sun);
  if (!skyProviderActive() || !isfinite(sunLength) || sunLength < 0.5f) {
    FlareState[uint2(0, 0)] = float4(previous.xy, 0.0f, waterDepth);
    FlareState[uint2(1, 0)] = float4(0.0f);
    return;
  }
  sun /= sunLength;
  const float3 viewDirection = mul(mat3(cb.camera.worldToView), sun);
  const float forward = (cb.camera.flags & rightHandedFlag) != 0u ? -viewDirection.z : viewDirection.z;
  const float4 clip = mul(cb.camera.viewToProjection, float4(viewDirection, 0.0f));
  if (forward <= 0.001f || abs(clip.w) <= 0.0001f || !all(isfinite(clip))) {
    FlareState[uint2(0, 0)] = float4(previous.xy, 0.0f, waterDepth);
    FlareState[uint2(1, 0)] = float4(0.0f);
    return;
  }
  const float2 sunUV = cameraNDCToScreenUV(cb.camera, clip.xy / clip.w);
  const float outside = max(max(-sunUV.x, sunUV.x - 1.0f), max(-sunUV.y, sunUV.y - 1.0f));
  const float edgeGate = 1.0f - smoothstep(0.0f, max(args.edgeFade, 0.0001f), max(outside, 0.0f));
  const float horizonGate = smoothstep(0.0f, 0.02f, dot(sun, up));
  // At any positive underwater depth the optical source is suppressed immediately.
  // waterTransition is the minimum resurfacing fade-in duration, not a water-fog dependency.
  const bool submerged = waterDepth > 0.0f;
  float3 attenuation = float3(0.0f);
  float cloud = 0.0f;
  if (!submerged && edgeGate * horizonGate > 0.0f) {
    attenuation = sunVisibility(eye, sun, clamp(skyArgs.sunAngularRadius, 0.0001f, 0.1f));
    cloud = sunCloudTransmittance(eye, sun, normalize(atmosphericSun));
  }
  const float geometryVisibility = max(attenuation.x, max(attenuation.y, attenuation.z));
  const float target = saturate(geometryVisibility * cloud * edgeGate * horizonGate);
  const float deltaTime = clamp(args.deltaTime, 0.0f, 0.25f);
  const float recovery = previous.w > 0.0f && !submerged ? max(args.waterTransition, 0.0f)
    : max(previousRecovery - deltaTime, 0.0f);
  float seconds = target > previous.z ? args.fadeInSeconds : args.fadeOutSeconds;
  if (recovery > 0.0f && target > previous.z) {
    seconds = max(seconds, args.waterTransition);
  }
  const float blend = 1.0f - exp(-deltaTime / max(seconds, 0.0001f));
  float visibility = lerp(saturate(previous.z), target, blend);
  if (submerged || horizonGate <= 0.0f || !isfinite(visibility)) {
    visibility = 0.0f;
  }
  const float3 illumination = max(skyArgs.sunIlluminance, float3(0.0f));
  const float peak = max(illumination.x, max(illumination.y, illumination.z));
  const float3 atmosphere = getAtmosphericTransmittanceForDir(normalize(atmosphericSun), skyArgs);
  const float3 sunColor = peak > 0.0f ? saturate(illumination / peak * atmosphere) : float3(0.0f);
  FlareState[uint2(0, 0)] = float4(sunUV, visibility, waterDepth);
  FlareState[uint2(1, 0)] = float4(sunColor, peak > 0.0f ? 1.0f + recovery : 0.0f);
}
