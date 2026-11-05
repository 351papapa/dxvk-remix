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

// Full resolution composite. The circle of confusion is recomputed per output pixel, so an
// in-focus pixel keeps its original texels exactly, even against a heavily defocused background.
// Depth exists only at render resolution, so dofResolveFarBlend() upsamples the blend with a colour guide.

[[vk::binding(POST_FX_DOF_RESOLVE_INPUT)]]
Texture2D<float4> InColorTexture;

[[vk::binding(POST_FX_DOF_RESOLVE_PRIMARY_LINEAR_VIEW_Z_INPUT)]]
Texture2D<float> PrimaryLinearViewZ;

[[vk::binding(POST_FX_DOF_RESOLVE_NEAR_INPUT)]]
Texture2D<float4> NearTexture;

[[vk::binding(POST_FX_DOF_RESOLVE_FAR_INPUT)]]
Texture2D<float4> FarTexture;

[[vk::binding(POST_FX_DOF_RESOLVE_OUTPUT)]]
RWTexture2D<float4> OutColorTexture;

[[vk::binding(POST_FX_DOF_RESOLVE_LINEAR_SAMPLER)]]
SamplerState LinearSampler;

[[vk::binding(POST_FX_DOF_RESOLVE_FOCUS_STATE_INPUT)]]
RWTexture1D<float> FocusState;

[[vk::binding(POST_FX_DOF_RESOLVE_PRIMARY_SURFACE_FLAGS_INPUT)]]
Texture2D<uint> PrimarySurfaceFlags;

layout(push_constant)
ConstantBuffer<PostFxDepthOfFieldArgs> cb;

#define POST_FX_DOF_USE_LINEAR_VIEW_Z 1
#define POST_FX_DOF_USE_FOCUS_STATE   1
#define POST_FX_DOF_USE_SURFACE_FLAGS 1
#include "rtx/pass/post_fx/post_fx_dof_common.slangh"

// Blur radii in full resolution pixels over which the sharp image hands over to the half resolution
// base layer. Below the lower edge the original texels are kept. The ramp hides the half resolution grid's seam.
static const float kSharpRadius = 1.0f;
static const float kBlurRadius = 3.0f;

// Sharpness of the guide's range term on the squared relative colour distance. A distance of 0.25,
// a clear silhouette, leaves about an eighth of the spatial weight, and 0.5 leaves almost none.
// Smooth shading sits below 0.05, where the upsample is close to plain bilinear.
static const float kDofGuideSharpness = 32.0f;

// A texel fraction this close to a boundary is snapped to it. At native resolution
// cb.inputOverOutputViewSize is 1 only to within float rounding, so this keeps a 1:1 ratio an exact point sample.
static const float kDofTexelSnap = 1e-3f;
static const float kDofTexelSnapScale = 1.0f / (1.0f - 2.0f * kDofTexelSnap);

// Base layer blend for one render resolution depth texel. The texel is point
// sampled, so its CoC belongs to exactly one surface and so does the blend.
float dofFarBlendForTexel(const int2 texel, const float focusDistance)
{
  const float coc = dofGetCoCFromViewDepth(dofLoadViewDepthTexel(texel), focusDistance);
  return smoothstep(kSharpRadius, kBlurRadius, abs(coc) * cb.maxGatherRadius);
}

// What the upsample resolves for one output pixel. Both use the same texels and weights,
// so they are computed together to fetch the guide once.
struct DofResolveBlend
{
  // How much of the half resolution base layer this pixel takes, in 0..1.
  float farBlend;
  // How much of this pixel is held-out view model, in 0..1. 1 passes the original texel through.
  float viewModel;
};

// Scale-free distance between two linear HDR colours, in 0..1.
// The luminance of the difference, rather than the difference of luminances, also catches hue edges.
// The denominator makes the distance independent of brightness, since this runs before tone mapping.
float dofGuideDistance(const float3 a, const float3 b)
{
  const float3 lumaWeights = float3(0.2126f, 0.7152f, 0.0722f);
  const float difference = dot(abs(a - b), lumaWeights);
  const float magnitude = dot(abs(a) + abs(b), lumaWeights);
  return saturate(difference / max(magnitude, 1e-4f));
}

// Joint bilateral upsample of the base layer blend, guided by the output resolution colour.
// A point sampled render resolution CoC would stair-step the blur boundary along silhouettes.
// Interpolated depth across a silhouette belongs to no surface, so each of the 2x2 depth texels
// is point sampled into its own blend and only the weights are interpolated. The weight is the
// bilinear spatial term times a range term on the colour, so the boundary lands on the colour edge.
DofResolveBlend dofResolveFarBlend(const uint2 pixelPos, const float3 centerColor, const float focusDistance)
{
  // This pixel's centre in render resolution texel space, as dofLoadViewDepth() maps it.
  const float2 srcPos = (float2(pixelPos) + 0.5f) * cb.inputOverOutputViewSize;

  DofResolveBlend result;

  if (cb.edgeAwareUpsample == 0u)
  {
    // A single point sample, with a hard render-grid view model mask.
    const int2 texel = int2(srcPos);
    result.viewModel = dofIsExcludedViewModelTexel(texel) ? 1.0f : 0.0f;
    result.farBlend = result.viewModel > 0.0f ? 0.0f : dofFarBlendForTexel(texel, focusDistance);
    return result;
  }

  const float2 basePos = srcPos - 0.5f;
  const int2 baseTexel = int2(floor(basePos));
  const float2 texelFrac = saturate((frac(basePos) - kDofTexelSnap) * kDofTexelSnapScale);

  float spatial[4];
  spatial[0] = (1.0f - texelFrac.x) * (1.0f - texelFrac.y);
  spatial[1] = texelFrac.x * (1.0f - texelFrac.y);
  spatial[2] = (1.0f - texelFrac.x) * texelFrac.y;
  spatial[3] = texelFrac.x * texelFrac.y;

  float blends[4];
  float viewModels[4];
  float minBlend = 1.0f;
  float maxBlend = 0.0f;
  float minViewModel = 1.0f;
  float maxViewModel = 0.0f;
  float bestSpatial = 0.0f;
  float nearestBlend = 0.0f;
  float nearestViewModel = 0.0f;

  [unroll]
  for (uint i = 0u; i < 4u; ++i)
  {
    const int2 offset = int2(int(i & 1u), int(i >> 1u));
    const int2 texel = baseTexel + offset;

    // A held-out view model texel is sharp, so its blend is zero. The flag goes through the same weights
    // as the blend, so the weapon's outline does not stair-step on the render grid.
    viewModels[i] = dofIsExcludedViewModelTexel(texel) ? 1.0f : 0.0f;
    blends[i] = viewModels[i] > 0.0f ? 0.0f : dofFarBlendForTexel(texel, focusDistance);

    if (spatial[i] > 0.0f)
    {
      minBlend = min(minBlend, blends[i]);
      maxBlend = max(maxBlend, blends[i]);
      minViewModel = min(minViewModel, viewModels[i]);
      maxViewModel = max(maxViewModel, viewModels[i]);
      if (spatial[i] > bestSpatial)
      {
        bestSpatial = spatial[i];
        nearestBlend = blends[i];
        nearestViewModel = viewModels[i];
      }
    }
  }

  // When all contributing neighbours agree, the guide cannot change the answer and is skipped.
  // The in-focus case returns exactly 0 here, which keeps the bit-identical early-out in main().
  if (maxBlend <= minBlend && maxViewModel <= minViewModel)
  {
    result.farBlend = nearestBlend;
    result.viewModel = nearestViewModel;
    return result;
  }

  // Render resolution texel index -> output resolution pixel index.
  const float2 outputPerSource = 1.0f / max(cb.inputOverOutputViewSize, float2(1e-6f, 1e-6f));
  const int2 maxOutputPixel = int2(cb.imageSize) - int2(1, 1);

  float weightedBlend = 0.0f;
  float weightedViewModel = 0.0f;
  float totalWeight = 0.0f;

  [unroll]
  for (uint j = 0u; j < 4u; ++j)
  {
    if (spatial[j] <= 0.0f)
    {
      continue;
    }

    const int2 offset = int2(int(j & 1u), int(j >> 1u));
    // The output pixel under this depth texel's centre, whose colour shows the true silhouette.
    const int2 guidePixel = clamp(
      int2((float2(baseTexel + offset) + 0.5f) * outputPerSource),
      int2(0, 0),
      maxOutputPixel);
    const float3 guideColor = InColorTexture.Load(int3(guidePixel, 0)).rgb;

    const float guideDistance = dofGuideDistance(centerColor, guideColor);
    const float weight = spatial[j] * exp(-kDofGuideSharpness * guideDistance * guideDistance);

    weightedBlend += weight * blends[j];
    weightedViewModel += weight * viewModels[j];
    totalWeight += weight;
  }

  // A colour that matches no neighbour collapses every weight. Fall back to the nearest texel,
  // never to zero, which would unblur a defocused pixel.
  if (totalWeight > 1e-6f)
  {
    const float invTotalWeight = 1.0f / totalWeight;
    result.farBlend = weightedBlend * invTotalWeight;
    result.viewModel = weightedViewModel * invTotalWeight;
  }
  else
  {
    result.farBlend = nearestBlend;
    result.viewModel = nearestViewModel;
  }

  return result;
}

// Four bilinear taps arranged as a tent over the half resolution grid.
// rtx.dof.bokehFilterStrength scales the tent from a plain bilinear fetch (0)
// to a full half-resolution-texel tent (1).
float4 sampleNearTent(const float2 uv, const float2 offset)
{
  float4 result = NearTexture.SampleLevel(LinearSampler, uv + float2(-offset.x, -offset.y), 0.0f);
  result += NearTexture.SampleLevel(LinearSampler, uv + float2( offset.x, -offset.y), 0.0f);
  result += NearTexture.SampleLevel(LinearSampler, uv + float2(-offset.x,  offset.y), 0.0f);
  result += NearTexture.SampleLevel(LinearSampler, uv + float2( offset.x,  offset.y), 0.0f);
  return result * 0.25f;
}

float3 sampleFarTent(const float2 uv, const float2 offset)
{
  float3 result = FarTexture.SampleLevel(LinearSampler, uv + float2(-offset.x, -offset.y), 0.0f).rgb;
  result += FarTexture.SampleLevel(LinearSampler, uv + float2( offset.x, -offset.y), 0.0f).rgb;
  result += FarTexture.SampleLevel(LinearSampler, uv + float2(-offset.x,  offset.y), 0.0f).rgb;
  result += FarTexture.SampleLevel(LinearSampler, uv + float2( offset.x,  offset.y), 0.0f).rgb;
  return result * 0.25f;
}

[shader("compute")]
[numthreads(POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1)]
void main(
  in uint2 threadId : SV_GroupThreadID,
  in uint2 pixelPos : SV_DispatchThreadID)
{
  if (any(pixelPos >= cb.imageSize))
  {
    return;
  }

  const float4 centerColor = InColorTexture.Load(int3(int2(pixelPos), 0));

  if (cb.maxGatherRadius <= 0.0f || cb.sampleCount == 0u)
  {
    OutColorTexture[pixelPos] = centerColor;
    return;
  }

  // The half resolution images are ceil(imageSize / 2), so full and half resolution texel centres
  // share normalised coordinates, to within half a texel on an odd dimension.
  const float2 uv = (float2(pixelPos) + 0.5f) * cb.invImageSize;
  const float focusDistance = dofGetFocusDistance();
  const DofResolveBlend blend = dofResolveFarBlend(pixelPos, centerColor.rgb, focusDistance);
  const float farBlend = blend.farBlend;
  const float4 nearProbe = NearTexture.SampleLevel(LinearSampler, uv, 0.0f);

  // The held-out view model is the nearest thing in the scene, so the near field must not cover it either.
  // The upsample already zeroed the base layer over it.
  const float viewModelSharpness = saturate(blend.viewModel);
  const float nearAlpha = saturate(nearProbe.a) * (1.0f - viewModelSharpness);

  // An in-focus pixel with no foreground over it, or a fully held-out view model pixel, passes through.
  // Most of a frame takes this path without touching the guide.
  if (farBlend <= 0.0f && nearAlpha <= 1e-3f)
  {
    OutColorTexture[pixelPos] = centerColor;
    return;
  }

  const float tentScale = saturate(cb.bokehFilterStrength);
  const float2 tentOffset = tentScale * 0.5f * cb.invHalfImageSize;

  float3 resolvedColor = centerColor.rgb;
  if (farBlend > 0.0f)
  {
    const float3 farColor = tentScale > 0.0f
      ? sampleFarTent(uv, tentOffset)
      : FarTexture.SampleLevel(LinearSampler, uv, 0.0f).rgb;
    resolvedColor = lerp(resolvedColor, farColor, farBlend);
  }

  // The near field is premultiplied, so this is a plain "over", and one scale holds the view model out.
  float4 nearField = nearProbe;
  if (nearField.a > 1e-3f && tentScale > 0.0f)
  {
    nearField = sampleNearTent(uv, tentOffset);
  }
  nearField *= 1.0f - viewModelSharpness;
  resolvedColor = resolvedColor * (1.0f - saturate(nearField.a)) + nearField.rgb;

  OutColorTexture[pixelPos] = float4(resolvedColor, centerColor.a);
}
