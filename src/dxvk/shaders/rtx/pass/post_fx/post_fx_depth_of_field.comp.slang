/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include "rtx/pass/post_fx/post_fx.h"

// Half resolution bokeh gather. Taps read one half resolution image with the signed CoC in alpha,
// which is monotonic in view depth, so no depth is read here.
//
// The pass emits two layers:
//   FAR  - the defocused base layer, gathered over a pixel's own disc with a scatter-as-gather test.
//          Near and far defocus both land here, so a blurred foreground has a blurred base under it.
//   NEAR - foreground that bleeds outward across silhouettes, premultiplied for a single "over".

[[vk::binding(POST_FX_DOF_COLOR_COC_INPUT)]]
Texture2D<float4> ColorCoCTexture;

[[vk::binding(POST_FX_DOF_TILE_INPUT)]]
Texture2D<float4> TileTexture;

[[vk::binding(POST_FX_DOF_NEAR_OUTPUT)]]
RWTexture2D<float4> OutNearTexture;

[[vk::binding(POST_FX_DOF_FAR_OUTPUT)]]
RWTexture2D<float4> OutFarTexture;

[[vk::binding(POST_FX_DOF_LINEAR_SAMPLER)]]
SamplerState LinearSampler;

layout(push_constant)
ConstantBuffer<PostFxDepthOfFieldArgs> cb;

#include "rtx/pass/post_fx/post_fx_dof_common.slangh"

// Dilated tile classification, resolved once per thread group.
groupshared float2 s_tileSamples[POST_FX_DOF_MAX_TILE_SAMPLES];
groupshared float2 s_tileRadii;

[shader("compute")]
[numthreads(POST_FX_DOF_TILE_SIZE, POST_FX_DOF_TILE_SIZE, 1)]
void main(
  in uint2 pixelPos : SV_DispatchThreadID,
  in uint2 groupId : SV_GroupID,
  in uint groupIndex : SV_GroupIndex)
{
  // One thread group owns one tile. A near-field disc can reach cb.tileReach tiles outwards,
  // which the host derives from the same clamped radius.
  const uint reach = min(cb.tileReach, uint(POST_FX_DOF_MAX_TILE_REACH));
  const uint span = 2u * reach + 1u;
  const uint tileSampleCount = span * span;
  const uint2 tileCount = max(
    (cb.halfImageSize + uint2(POST_FX_DOF_TILE_SIZE - 1u, POST_FX_DOF_TILE_SIZE - 1u)) / uint(POST_FX_DOF_TILE_SIZE),
    uint2(1u, 1u));

  if (groupIndex < tileSampleCount)
  {
    const int2 tileOffset = int2(groupIndex % span, groupIndex / span) - int2(reach, reach);
    const int2 tileCoord = clamp(int2(groupId) + tileOffset, int2(0, 0), int2(tileCount) - int2(1, 1));
    s_tileSamples[groupIndex] = TileTexture.Load(int3(tileCoord, 0)).xy;
  }
  GroupMemoryBarrierWithGroupSync();

  if (groupIndex == 0u)
  {
    float2 dilated = 0.0f;
    for (uint i = 0; i < tileSampleCount; ++i)
    {
      dilated = max(dilated, s_tileSamples[i]);
    }
    // The base layer gathers only within a pixel's own disc, so only the near field is dilated.
    s_tileRadii = float2(s_tileSamples[tileSampleCount / 2u].x, dilated.y);
  }
  GroupMemoryBarrierWithGroupSync();

  if (any(pixelPos >= cb.halfImageSize))
  {
    return;
  }

  const float tileBaseRadius = s_tileRadii.x;
  const float tileNearRadius = s_tileRadii.y;
  const float4 centerSample = ColorCoCTexture.Load(int3(int2(pixelPos), 0));
  const float3 centerColor = centerSample.rgb;
  const float centerCoC = centerSample.a;

  // Most tiles are entirely in focus and exit here after one fetch.
  if (tileBaseRadius < POST_FX_DOF_MIN_RADIUS && tileNearRadius < POST_FX_DOF_MIN_RADIUS)
  {
    OutNearTexture[pixelPos] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    OutFarTexture[pixelPos] = float4(centerColor, 0.0f);
    return;
  }

  const float2 pixelCenter = float2(pixelPos) + 0.5f;
  const float phase = dofInterleavedGradientNoise(pixelCenter, cb.frameIdx) * kDofTwoPi;
  // A complex multiply advances the golden angle spiral without a sin/cos pair per tap.
  const float2 rotation = float2(cos(kDofGoldenAngle), sin(kDofGoldenAngle));
  const uint maxTaps = min(cb.sampleCount, uint(POST_FX_DOF_MAX_SAMPLES));
  // The smallest CoC difference worth acting on, half a half resolution pixel of radius.
  const float cocEpsilon = 0.5f / max(cb.maxGatherRadiusHalf, 1e-4f);

  // -------------------------------------------------------------------------
  // Base layer
  // -------------------------------------------------------------------------
  float3 baseColor = centerColor;
  const float centerRadius = min(abs(centerCoC) * cb.maxGatherRadiusHalf, cb.maxGatherRadiusHalf);

  if (centerRadius >= POST_FX_DOF_MIN_RADIUS && maxTaps > 0u)
  {
    const uint tapCount = dofTapCountForRadius(centerRadius, maxTaps);

    // The destination pixel is tap zero, weighted by the same radial profile as any other tap.
    float weightSum = dofDiscCoverage(centerRadius, 0.0f) * dofBokehProfile(0.0f);
    float3 colorSum = centerColor * weightSum;
    float2 tapDirection = float2(cos(phase), sin(phase));
    const float invCenterRadius = 1.0f / centerRadius;

    [loop]
    for (uint i = 0; i < tapCount; ++i)
    {
      // The sqrt spreads taps uniformly by area. Only the starting angle is dithered,
      // which keeps frame-to-frame variance low.
      const float fraction = (float(i) + 0.5f) / float(tapCount);
      const float tapRadius = sqrt(fraction) * centerRadius;
      const float2 tapUv = (pixelCenter + tapDirection * tapRadius) * cb.invHalfImageSize;
      tapDirection = float2(
        tapDirection.x * rotation.x - tapDirection.y * rotation.y,
        tapDirection.x * rotation.y + tapDirection.y * rotation.x);

      const float4 tap = ColorCoCTexture.SampleLevel(LinearSampler, tapUv, 0.0f);
      const float tapCoC = tap.a;
      const float tapDiscRadius = abs(tapCoC) * cb.maxGatherRadiusHalf;

      // A tap contributes only if its own disc reaches this pixel,
      // so a defocused background cannot bleed over an in-focus foreground.
      float coverage = dofDiscCoverage(tapDiscRadius, tapRadius);

      // A source nearer than this pixel and in front of the focus plane belongs to the near layer.
      // Taking it here would count it twice.
      if (tapCoC < 0.0f && tapCoC < centerCoC - cocEpsilon)
      {
        coverage = 0.0f;
      }

      const float weight = coverage * dofBokehProfile(tapRadius * invCenterRadius);
      colorSum += tap.rgb * weight;
      weightSum += weight;
    }

    // bokehMinIntensity 0 zeroes the centre tap's weight, so all weights can be zero.
    baseColor = weightSum > 1e-5f ? colorSum / weightSum : centerColor;
  }

  // -------------------------------------------------------------------------
  // Near field layer
  // -------------------------------------------------------------------------
  float4 nearResult = 0.0f;
  const float nearGatherRadius = min(tileNearRadius, cb.maxGatherRadiusHalf);

  if (nearGatherRadius >= POST_FX_DOF_MIN_RADIUS && maxTaps > 0u)
  {
    const uint tapCount = dofTapCountForRadius(nearGatherRadius, maxTaps);
    const float invNearRadius = 1.0f / nearGatherRadius;

    // This pixel is its own first near-field source.
    const float centerNearRadius = max(-centerCoC, 0.0f) * cb.maxGatherRadiusHalf;
    float coverageSum = dofDiscCoverage(centerNearRadius, 0.0f);
    float weightSum = coverageSum * dofBokehProfile(0.0f);
    float3 colorSum = centerColor * weightSum;

    // Offset by half a golden angle so the near and base kernels do not share texels.
    const float nearPhase = phase + 0.5f * kDofGoldenAngle;
    float2 tapDirection = float2(cos(nearPhase), sin(nearPhase));

    [loop]
    for (uint i = 0; i < tapCount; ++i)
    {
      const float fraction = (float(i) + 0.5f) / float(tapCount);
      const float tapRadius = sqrt(fraction) * nearGatherRadius;
      const float2 tapUv = (pixelCenter + tapDirection * tapRadius) * cb.invHalfImageSize;
      tapDirection = float2(
        tapDirection.x * rotation.x - tapDirection.y * rotation.y,
        tapDirection.x * rotation.y + tapDirection.y * rotation.x);

      const float4 tap = ColorCoCTexture.SampleLevel(LinearSampler, tapUv, 0.0f);
      // Acceptance uses the source's near CoC, so a foreground disc can cover a focused pixel.
      const float tapNearRadius = max(-tap.a, 0.0f) * cb.maxGatherRadiusHalf;
      const float coverage = dofDiscCoverage(tapNearRadius, tapRadius);
      const float weight = coverage * dofBokehProfile(tapRadius * invNearRadius);

      coverageSum += coverage;
      weightSum += weight;
      colorSum += tap.rgb * weight;
    }

    if (weightSum > 1e-5f)
    {
      // Alpha is the fraction of the gather disc that near-field sources cover.
      // It is bounded and varies smoothly as the foreground moves across the pixel.
      const float alpha = saturate(coverageSum / float(tapCount + 1u));
      // Premultiplied, so the resolve's bilinear upsample of a partly covered edge stays correct.
      nearResult = float4((colorSum / weightSum) * alpha, alpha);
    }
  }

  OutNearTexture[pixelPos] = nearResult;
  OutFarTexture[pixelPos] = float4(baseColor, centerRadius);
}
