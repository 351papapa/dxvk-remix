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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// The copy passes of the cone maps (POM_CONE_* modes in pom_cone_args.h). Copies read through the source
// view with Load(), so its format, sRGB decode and swizzle give the same value a POM march samples. A 2-channel output
// written to a 1-channel image keeps only its first channel (the height pyramid). Cone values are k / 255 or k / 65535
// already (rounded down by the passes that made them), so writing them to the UNORM G channel keeps them exactly.

#include "rtx/pass/pom_cone/pom_cone_args.h"

layout(binding = POM_CONE_BINDING_SOURCE)
Texture2D<float4> Source;

layout(binding = POM_CONE_BINDING_CONE)
Texture2D<float4> Cone;

layout(binding = POM_CONE_BINDING_OUTPUT)
RWTexture2D<float2> Output;

layout(push_constant)
ConstantBuffer<PomConeCopyArgs> cb;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  const uint2 position = cb.origin + threadId;
  if (any(position >= cb.extent))
  {
    return;
  }

  if (cb.mode == POM_CONE_REDUCE_MAX)
  {
    const int2 last = int2(cb.sourceExtent) - int2(1, 1);
    const int2 base = int2(position) * 2;
    float height = 0.0f;
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
      const int2 texel = min(base + int2(i & 1, i >> 1), last);
      height = max(height, Source.Load(int3(texel, 0)).r);
    }
    Output[position] = float2(height, 0.0f);
    return;
  }

  const int2 texel = int2(position);
  const float height = Source.Load(int3(texel, int(cb.sourceMip))).r;
  float cone = 0.0f;
  if (cb.mode == POM_CONE_COPY_HEIGHT_CONE)
  {
    cone = Cone.Load(int3(texel, 0)).r;
  }
  else if (cb.mode == POM_CONE_COPY_HEIGHT_COARSE)
  {
    const int shift = int(cb.coarseShift);
    cone = Cone.Load(int3(texel.x >> shift, texel.y >> shift, 0)).r;
  }
  Output[position] = float2(height, cone);
}
