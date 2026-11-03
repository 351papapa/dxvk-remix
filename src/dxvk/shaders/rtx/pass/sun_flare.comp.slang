//!variant sun_flare.comp
//!end-variants

#include "rtx/pass/sun_flare.h"

layout(push_constant) ConstantBuffer<SunFlareArgs> args;
layout(binding = SUN_FLARE_STATE) Texture2D<float4> FlareState;
layout(binding = SUN_FLARE_COLOR) RWTexture2D<float4> Color;
layout(binding = SUN_FLARE_VC_STAR) Sampler2D VcStar;
layout(binding = SUN_FLARE_VC_HEX) Sampler2D VcHex;
layout(binding = SUN_FLARE_VC_CIRCLE) Sampler2D VcCircle;
layout(binding = SUN_FLARE_VC_RING) Sampler2D VcRing;
layout(binding = SUN_FLARE_SA_STAR) Sampler2D SaStar;

struct Flare {
  float position;
  float size;
  float3 color;
  uint texture;
};

// Rockstar data, transcribed by gta-reversed (Coronas.cpp:61) and reVC (Coronas.cpp:32).
// Colors are the raw /256 coefficients, sizes are original pixel half-sizes /4.
static const Flare saFlares[26] = {
  { 4.00f, 8.00f, float3(36,30,24), 0 },
  { 3.00f,11.20f, float3(24,18,15), 0 },
  { 2.00f, 8.00f, float3(24,12,12), 0 },
  { 1.50f, 9.60f, float3(54,54,48), 0 },
  { 1.25f, 8.00f, float3(24,24,18), 0 },
  { 0.80f,22.40f, float3(36,30,24), 0 },
  { 0.60f, 6.40f, float3(24,15,12), 0 },
  { 0.25f,16.00f, float3(36,30,30), 0 },
  { 0.10f, 9.60f, float3(18,18,18), 0 },
  { 0.05f,22.40f, float3(36,30,24), 0 },
  {-0.03f, 4.80f, float3(18,18,18), 0 },
  {-0.10f, 9.60f, float3(42,42,42), 0 },
  {-0.30f, 8.00f, float3(18, 6, 6), 0 },
  {-0.40f,96.00f, float3(18,12, 9), 0 },
  {-0.55f, 6.40f, float3(18,18,12), 0 },
  {-0.75f,22.40f, float3(42,24,18), 0 },
  {-0.90f, 8.32f, float3(21,12,18), 0 },
  {-1.00f,17.60f, float3(42,13,18), 0 },
  {-1.20f, 5.60f, float3(21,12,12), 0 },
  {-1.35f,14.40f, float3(42,42,24), 0 },
  {-1.70f,86.80f, float3(21,15,15), 0 },
  {-2.00f, 8.00f, float3(48,30,30), 0 },
  {-2.50f, 7.20f, float3(21,15,12), 0 },
  {-3.00f,22.40f, float3(42,30,24), 0 },
  {-6.00f,38.40f, float3(42,42,30), 0 },
  {-9.00f,22.40f, float3(42,30,36), 0 }
};
static const Flare vcFlares[7] = {
  {-0.50f,15.00f, float3(50,50, 0), 1},
  {-1.00f,10.00f, float3(50,20, 0), 2},
  {-1.50f,15.00f, float3(50, 0, 0), 3},
  {-2.50f,25.00f, float3(50, 0, 0), 1},
  { 0.50f,12.50f, float3(40,40,25), 1},
  { 0.05f,20.00f, float3(30,22, 9), 2},
  { 1.30f, 7.50f, float3(50,30, 9), 3}
};

float3 sampleFlare(uint texture, float2 uv) {
  if (args.textureSet == 1u) {
    return SaStar.SampleLevel(uv, 0).xyz;
  }
  if (texture == 1u) {
    return VcHex.SampleLevel(uv, 0).xyz;
  }
  if (texture == 2u) {
    return VcCircle.SampleLevel(uv, 0).xyz;
  }
  if (texture == 3u) {
    return VcRing.SampleLevel(uv, 0).xyz;
  }
  return VcStar.SampleLevel(uv, 0).xyz;
}

float sharedFlicker() {
  if (!args.saFlicker || args.style != 0u) {
    return 1.0f;
  }
  uint value = args.frameIndex * 747796405u + 2891336453u;
  value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
  value = (value >> 22u) ^ value;
  return 0.7f + float(value & 65535u) * (0.3f / 65535.0f);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID) {
  if (pixel.x >= args.width || pixel.y >= args.height) {
    return;
  }
  const float4 state = FlareState[uint2(0, 0)];
  const float4 tint = FlareState[uint2(1, 0)];
  if (state.z <= 0.0001f || tint.w <= 0.0f || args.brightness <= 0.0f) {
    return;
  }
  const float2 screen = float2(args.width, args.height);
  const float2 center = screen * 0.5f;
  const float2 sunOffset = state.xy * screen - center;
  const float2 coordinate = float2(pixel) + 0.5f;
  const float resolutionScale = args.scaleWithResolution ? screen.y / 480.0f : 1.0f;
  const float aspect = args.style == 1u ? max(args.vcAspectRatio, 0.01f) : 1.0f;
  const float sourceIntensity = state.z * sharedFlicker() * args.brightness;
  const float intensity = args.style == 0u ? 255.0f / 256.0f : 200.0f / 256.0f;
  float3 contribution = float3(0.0f);
  const uint count = args.style == 0u ? 26u : 7u;
  for (uint i = 0; i < count; ++i) {
    Flare flare;
    if (args.style == 0u) {
      flare = saFlares[i];
    } else {
      flare = vcFlares[i];
    }
    const float halfHeight = max(4.0f * flare.size * resolutionScale * args.sizeScale, 0.001f);
    const float2 halfSize = float2(halfHeight * aspect, halfHeight);
    const float2 uv = (coordinate - (center + sunOffset * flare.position)) / (2.0f * halfSize) + 0.5f;
    if (all(uv >= 0.0f) && all(uv <= 1.0f)) {
      // The original ONE/ONE blend uses texture RGB; multiplying texture alpha again darkens the mask.
      contribution += max(sampleFlare(flare.texture, uv), float3(0.0f)) * (flare.color / 256.0f);
    }
  }
  const float4 input = Color[pixel];
  Color[pixel] = float4(input.xyz + contribution * tint.xyz * (sourceIntensity * intensity), input.w);
}
