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

// rtx.alphaBake.census. One workgroup per triangle of a baked opacity micromap array.
// The AND and OR of the triangle's words decide whether every micro-triangle is transparent (all words 0),
// every one is opaque (all words == opaqueWord), or the triangle is mixed.

#include "omm_class_census_args.h"

layout(push_constant)
ConstantBuffer<OmmCensusArgs> cb;

layout(binding = OMM_CENSUS_BINDING_ARRAY_INPUT)
StructuredBuffer<uint> OmmArray;

layout(binding = OMM_CENSUS_BINDING_COUNTERS)
RWStructuredBuffer<uint> Counters;

groupshared uint g_allTransparent;
groupshared uint g_allOpaque;

[shader("compute")]
[numthreads(OMM_CENSUS_THREADS_PER_GROUP, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint threadId : SV_GroupThreadID)
{
  const uint triangleIndex = groupId.y * cb.groupsX + groupId.x;

  if (threadId == 0)
  {
    g_allTransparent = 1;
    g_allOpaque = 1;
  }
  GroupMemoryBarrierWithGroupSync();

  if (triangleIndex < cb.numTriangles)
  {
    const uint base = triangleIndex * cb.wordsPerTriangle;
    bool transparent = true;
    bool opaque = true;
    for (uint w = threadId; w < cb.wordsPerTriangle; w += OMM_CENSUS_THREADS_PER_GROUP)
    {
      const uint word = OmmArray[base + w];
      transparent = transparent && word == 0u;
      opaque = opaque && word == cb.opaqueWord;
    }
    if (!transparent)
    {
      g_allTransparent = 0;
    }
    if (!opaque)
    {
      g_allOpaque = 0;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if (threadId == 0 && triangleIndex < cb.numTriangles)
  {
    const uint cls = g_allTransparent != 0 ? 0u : (g_allOpaque != 0 ? 1u : 2u);
    uint unused;
    InterlockedAdd(Counters[cls], 1u, unused);
    InterlockedAdd(Counters[cb.slotOffset + cls], 1u, unused);
  }
}
