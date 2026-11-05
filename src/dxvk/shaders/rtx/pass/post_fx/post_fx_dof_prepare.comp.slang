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

[[vk::binding(POST_FX_DOF_PREPARE_INPUT)]]
Texture2D<float4> InColorTexture;

[[vk::binding(POST_FX_DOF_PREPARE_PRIMARY_LINEAR_VIEW_Z_INPUT)]]
Texture2D<float> PrimaryLinearViewZ;

[[vk::binding(POST_FX_DOF_PREPARE_COLOR_COC_OUTPUT)]]
RWTexture2D<float4> OutColorCoC;

[[vk::binding(POST_FX_DOF_PREPARE_TILE_OUTPUT)]]
RWTexture2D<float4> OutTile;

[[vk::binding(POST_FX_DOF_PREPARE_FOCUS_STATE_INPUT)]]
RWTexture1D<float> FocusState;

[[vk::binding(POST_FX_DOF_PREPARE_PRIMARY_SURFACE_FLAGS_INPUT)]]
Texture2D<uint> PrimarySurfaceFlags;

layout(push_constant)
ConstantBuffer<PostFxDepthOfFieldArgs> cb;

#define POST_FX_DOF_USE_LINEAR_VIEW_Z 1
#define POST_FX_DOF_USE_FOCUS_STATE   1
#define POST_FX_DOF_USE_SURFACE_FLAGS 1
#include "rtx/pass/post_fx/post_fx_dof_common.slangh"

// Per-tile reduction of the two blur radii. One thread group owns one tile, so it stays in the group.
groupshared float2 s_tileReduce[POST_FX_DOF_TILE_THREADS];

[shader("compute")]
[numthreads(POST_FX_DOF_TILE_SIZE, POST_FX_DOF_TILE_SIZE, 1)]
void main(
  in uint2 pixelPos : SV_DispatchThreadID,
  in uint2 groupId : SV_GroupID,
  in uint groupIndex : SV_GroupIndex)
{
  const float focusDistance = dofGetFocusDistance();
  const uint2 maxFullPixel = max(cb.imageSize, uint2(1u, 1u)) - uint2(1u, 1u);
  const uint2 fullResBase = pixelPos * 2u;

  // A plain average of the 2x2 block preserves the energy of HDR highlights for the bokeh gather.
  // Firefly weighting such as 1/(1+luma) would throw it away and flatten the highlights.
  float3 colorSum = 0.0f;
  float3 keptColorSum = 0.0f;
  float keptCount = 0.0f;
  float cocSum = 0.0f;
  float nearestCoC = 0.0f;
  float maxAbsCoC = 0.0f;

  [unroll]
  for (uint y = 0; y < 2u; ++y)
  {
    [unroll]
    for (uint x = 0; x < 2u; ++x)
    {
      const uint2 subPixel = min(fullResBase + uint2(x, y), maxFullPixel);
      const float3 subColor = InColorTexture.Load(int3(int2(subPixel), 0)).rgb;
      colorSum += subColor;

      // A held-out view model texel has no blur radius, so it also drops out of the tile statistics.
      const bool excluded = dofIsExcludedViewModel(subPixel);
      if (!excluded)
      {
        keptColorSum += subColor;
        keptCount += 1.0f;
      }

      const float subCoC = excluded
        ? 0.0f
        : dofGetCoCFromViewDepth(dofLoadViewDepth(subPixel), focusDistance);
      cocSum += subCoC;
      // The minimum signed CoC is the nearest sub-pixel. The tile uses extremes, so classification is conservative.
      nearestCoC = min(nearestCoC, subCoC);
      maxAbsCoC = max(maxAbsCoC, abs(subCoC));
    }
  }

  // Averaging only the kept sub-pixels stops weapon colour from fringing the background's blur.
  // A boundary texel can end up up to 4x too bright, which reads better than a weapon-coloured ghost.
  // A fully held-out texel is never read, so the plain average only avoids a divide by zero.
  const float3 color = keptCount > 0.0f ? keptColorSum / keptCount : colorSum * 0.25f;
  const float coc = cocSum * 0.25f;

  // x: the largest |CoC| radius in the tile, which bounds the base layer and needs no dilation.
  // y: the largest near-field radius, which the gather dilates over neighbouring tiles.
  float2 tileRadii = float2(maxAbsCoC, max(-nearestCoC, 0.0f)) * cb.maxGatherRadiusHalf;
  const bool inBounds = all(pixelPos < cb.halfImageSize);
  if (!inBounds)
  {
    tileRadii = 0.0f;
  }

  s_tileReduce[groupIndex] = tileRadii;
  GroupMemoryBarrierWithGroupSync();

  // Tree reduction. Only the steps need ordering, since no slot is read and written in one step.
  [unroll]
  for (uint stride = POST_FX_DOF_TILE_THREADS / 2u; stride > 0u; stride >>= 1u)
  {
    if (groupIndex < stride)
    {
      s_tileReduce[groupIndex] = max(s_tileReduce[groupIndex], s_tileReduce[groupIndex + stride]);
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (groupIndex == 0u)
  {
    OutTile[groupId] = float4(s_tileReduce[0], 0.0f, 0.0f);
  }

  if (!inBounds)
  {
    return;
  }

  OutColorCoC[pixelPos] = float4(color, coc);
}
