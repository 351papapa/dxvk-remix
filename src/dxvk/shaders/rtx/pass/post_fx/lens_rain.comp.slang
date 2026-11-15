// SPDX-License-Identifier: MIT
// Local runtime feature source. See CREDITS.md.
#include "rtx/pass/post_fx/lens_rain.h"

layout(binding = LENS_RAIN_INPUT) Sampler2D SceneColor;
layout(binding = LENS_RAIN_OUTPUT) RWTexture2D<float4> OutColor;
layout(binding = LENS_RAIN_DROPS) StructuredBuffer<LensRainDrop> Drops;
layout(binding = LENS_RAIN_TILES) StructuredBuffer<uint> DropTiles;
layout(push_constant) ConstantBuffer<LensRainArgs> args;

// The tile index accelerates lookup only; positions and trajectories are independent droplets.
float4 evaluateDrops(float2 screen, uint2 tile, out float3 absorption) {
  absorption = float3(0);
  uint header = (tile.y * args.tileCount.x + tile.x) * 2;
  uint offset = DropTiles[header];
  uint count = DropTiles[header + 1];
  float4 result = float4(0);
  uint selected = 0xffffffffu;
  for (uint i = 0; i < count; ++i) {
    uint dropId = DropTiles[offset + i];
    LensRainDrop drop = Drops[dropId];
    float2 axis = drop.axisOpacityStrength.xy;
    float2 delta = screen - drop.centerRadii.xy;
    float2 q = float2(dot(delta, float2(axis.y, -axis.x)), dot(delta, axis)) / drop.centerRadii.zw;
    // Game droplets have a gently tapered profile along the native sliding axis.
    if (drop.absorptionKind.w > 0) { q.x *= 1.0f + 0.12f * clamp(q.y, -1.0f, 1.0f); }
    float r = length(q);
    float aa = min(0.25f, 1.5f / (float(args.imageSize.y) * drop.centerRadii.z));
    float mask = (1.0f - smoothstep(1.0f - aa, 1.0f + aa, r)) * drop.axisOpacityStrength.z * (drop.absorptionKind.w > 0 ? 1.0f : args.amount);
    float2 normal = (q.x * float2(axis.y, -axis.x) + q.y * axis)
      * sqrt(saturate(1.0f - r * r)) * 2.0f * drop.axisOpacityStrength.w;
    float rim = exp(-abs(r - 0.83f) * 24.0f) * mask;
    if (mask > result.z || (mask == result.z && dropId < selected)) {
      result = float4(normal, mask, rim);
      selected = dropId;
      absorption = drop.absorptionKind.xyz * (saturate(1.0f - r * r) * saturate(1.0f - r * r));
    }
  }
  return result;
}

// This flag describes sampled values, not raw bytes. Hardware sRGB sampling and
// the final blit's sRGB encoding already cancel each other and need no manual conversion.
// Remix's HDR float backbuffer can still contain gamma 2.2 values above one.
float3 screenDropDecode(float3 value) {
  if (args.colorFlags.x != 0) { value = pow(max(value, float3(0)), float3(2.2f)); }
  return value;
}
float3 screenDropEncode(float3 value) {
  if (args.colorFlags.x != 0) { value = pow(max(value, float3(0)), float3(1.0f / 2.2f)); }
  return value;
}

float3 sampleScene(float2 uv) {
  return screenDropDecode(SceneColor.SampleLevel(clamp(uv, args.invImageSize * 0.5f, 1.0f - args.invImageSize * 0.5f), 0.0f).rgb);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID) {
  if (any(pixel >= args.imageSize)) { return; }
  float2 uv = (float2(pixel) + 0.5f) * args.invImageSize;
  float4 original = SceneColor.SampleLevel(uv, 0.0f);
  float2 screen = float2(uv.x * float(args.imageSize.x) / float(args.imageSize.y), uv.y);
  uint2 tile = min(uint2(uv * float2(args.tileCount)), args.tileCount - 1);
  float3 absorption;
  float4 drop = evaluateDrops(screen, tile, absorption);
  if (drop.z <= 0.001f) { OutColor[pixel] = original; return; }
  float resolutionScale = float(args.imageSize.y) / 1080.0f;
  float2 offset = drop.xy * args.refractPixels * resolutionScale * args.invImageSize;
  float2 refracted = uv - offset;
  float2 blur = args.blurPixels * resolutionScale * args.invImageSize;
  float3 color = sampleScene(refracted) * 0.5f;
  color += (sampleScene(refracted + float2(blur.x, 0)) + sampleScene(refracted - float2(blur.x, 0))
    + sampleScene(refracted + float2(0, blur.y)) + sampleScene(refracted - float2(0, blur.y))) * 0.125f;
  color.r = lerp(color.r, sampleScene(refracted - offset * args.dispersion).r, 0.6f);
  color.b = lerp(color.b, sampleScene(refracted + offset * args.dispersion).b, 0.6f);
  // Rim contrast follows the sampled scene, so droplets do not emit light in darkness.
  color *= (1.0f - 0.16f * drop.w) * exp(-absorption);
  OutColor[pixel] = float4(screenDropEncode(lerp(screenDropDecode(original.rgb), color, saturate(drop.z))), original.a);
}
