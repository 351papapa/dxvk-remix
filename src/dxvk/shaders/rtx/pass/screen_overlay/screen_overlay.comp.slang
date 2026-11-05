/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "screen_overlay.h"

layout(binding = SCREEN_OVERLAY_INPUT_OUTPUT)
RWTexture2D<float4> InOutImage;

layout(binding = SCREEN_OVERLAY_TEXTURE)
Sampler2D<float4> OverlayTexture;

layout(push_constant)
ConstantBuffer<ScreenOverlayArgs> cb;

[shader("compute")]
[numthreads(SCREEN_OVERLAY_TILE_SIZE, SCREEN_OVERLAY_TILE_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID) {
  if (any(threadId >= cb.imageSize))
    return;

  // Map the pixel into the layer's destination rectangle.
  const float2 screenUv = (float2(threadId) + 0.5f) / float2(cb.imageSize);
  const float2 uv = (screenUv - float2(cb.rectX, cb.rectY)) / float2(cb.rectW, cb.rectH);
  if (any(uv < 0.0f) || any(uv > 1.0f))
    return;

  float4 overlay = OverlayTexture.SampleLevel(uv, 0);
  if (cb.alphaCutoff > 0.0f && overlay.a <= cb.alphaCutoff)
    return;
  float4 target = InOutImage[threadId];

  // Overlay over target. Premultiplied sources already carry alpha in RGB,
  // so only the opacity scales them; straight-alpha sources are weighted here.
  const float alpha = saturate(overlay.a * cb.opacity);
  const float3 source = (cb.flags & SCREEN_OVERLAY_FLAG_PREMULTIPLIED) != 0
    ? overlay.rgb * cb.opacity
    : overlay.rgb * alpha;
  float3 blended = source + target.rgb * (1.0f - alpha);

  InOutImage[threadId] = float4(blended, 1.0f);
}
