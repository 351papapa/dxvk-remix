//!variant local_starburst_prepare.comp
//!end-variants
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/local_starbursts.h"
#include "rtx/concept/camera/camera.slangh"
#include "rtx/algorithm/visibility.slangh"

layout(push_constant) ConstantBuffer<LocalStarburstArgs> args;
layout(binding = STARBURST_INPUT) StructuredBuffer<LocalStarburstInput> sources;
layout(binding = STARBURST_SPRITES) RWStructuredBuffer<LocalStarburstSprite> sprites;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint i : SV_DispatchThreadID) {
  if (i >= args.count) {
    return;
  }
  sprites[i] = (LocalStarburstSprite)0;
  const LocalStarburstInput source = sources[i];
  const float3 eye = cameraGetWorldPosition(cb.camera);
  const float3 toLight = source.positionSize.xyz - eye;
  const float distance = length(toLight);
  const float4 view = mul(cb.camera.worldToView, float4(source.positionSize.xyz, 1));
  const float4 clip = mul(cb.camera.viewToProjection, view);
  if (!all(isfinite(clip)) || !isfinite(distance) || distance <= 0.0001f || clip.w <= 0.0001f ||
      ((cb.camera.flags & rightHandedFlag) ? -view.z : view.z) <= 0 || source.colorOpacity.w <= 0) {
    return;
  }
  const float2 screen = float2(args.width, args.height);
  const float2 uv = cameraNDCToScreenUV(cb.camera, clip.xy / clip.w);
  if (any(uv < 0) || any(uv > 1)) {
    return;
  }
  const float3 direction = toLight / distance;
  const float bias = min(args.endpointBias, distance * 0.25f);
  Ray ray;
  ray.origin = eye;
  ray.direction = direction;
  ray.tMax = distance - max(bias, distance * 1e-5f);
  ray.coneRadius = 0;
  ray.spreadAngle = 0;
  const uint8_t mask = uint8_t(OBJECT_MASK_OPAQUE | OBJECT_MASK_ALPHA_BLEND | OBJECT_MASK_TRANSLUCENT);
  const VisibilityResult visibility = traceVisibilityRay<visibilityModeDisableCulling
    | visibilityModeEnableTranslucentMaterials>(ray, mask, invalidRayPortalIndex, PORTAL_SPACE_NONE);
  const float3 tint = saturate(visibility.attenuation) * source.colorOpacity.xyz * source.colorOpacity.w;
  if (!all(isfinite(tint)) || !any(tint > 0)) {
    return;
  }
  const float halfWidth = 0.5f * screen.y * source.positionSize.w * args.sizeScale;
  LocalStarburstSprite sprite;
  sprite.centerHalfSize = float4(uv * screen, halfWidth, halfWidth / source.aspect);
  sprite.tint = float4(tint * args.brightness, 1);
  sprites[i] = sprite;
}
