/*
* Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx/pass/tonemap/tonemapping.slangh"
#include "rtx/pass/tonemap/tonemapping_noise.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/color.slangh"

[[vk::binding(TONEMAPPING_APPLY_BLUE_NOISE_TEXTURE_INPUT)]]
Texture2DArray BlueNoise;

[[vk::binding(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT)]]
RWTexture2D<float4> InColorBuffer;

[[vk::binding(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT)]]
RWTexture1D<float> InExposure;

[[vk::binding(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT)]]
RWTexture2D<float4> OutColorBuffer;

layout(push_constant)
ConstantBuffer<ToneMappingApplyToneMappingArgs> cb;

float3 applyToneMapping(float3 color, uint2 pixelPosition)
{
  color *= getExposure(InExposure, cb.enableAutoExposure, cb.exposureFactor);

  if (cb.toneMappingEnabled != 0)
  {
    color = tonemap(color, cb.tonemapPeak, cb.tonemapPrimaries, cb.grade);
  }

  if (cb.performSRGBConversion > 0)
  {
    color = linearToGamma(color);
  }

  // Dither only when this pass encodes to gamma. Linear-space dither would be
  // amplified in the near-black toe by the later sRGB encode in srgb_dither.
  return ditherTo8Bit(
    color,
    pixelPosition,
    cb.frameIndex,
    cb.performSRGBConversion > 0 && cb.ditherMode != ditherModeNone,
    cb.ditherMode == ditherModeSpatialTemporal
  );
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  uint width;
  uint height;
  OutColorBuffer.GetDimensions(width, height);
  if (threadId.x >= width || threadId.y >= height)
  {
    return;
  }

  float3 inputColor = imageLoad(InColorBuffer, threadId).xyz;

  float3 outputColor = applyToneMapping(inputColor, threadId);

  // Clamp outputColor to avoid problems with NIS. HDR output keeps everything up to the peak
  // (relative to paper white) for the encode at present.
  outputColor = cb.hdrOutput != 0
    ? clamp(outputColor, 0.0f, max(cb.tonemapPeak, 1.0f))
    : saturate(outputColor);

  OutColorBuffer[threadId] = vec4(outputColor, 1.0f);
}
