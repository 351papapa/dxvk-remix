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

// rtx.alphaBake.compactMicromaps. One workgroup per triangle of a baked opacity micromap array.
// A uniformly transparent or opaque triangle gets the matching special index. A mixed one takes the next compact
// slot, whose triangle record points at the triangle's data in the baked array. The order does not matter,
// because the index buffer names the slot.

#include "omm_compact_args.h"

layout(push_constant)
ConstantBuffer<OmmCompactArgs> cb;

layout(binding = OMM_COMPACT_BINDING_ARRAY_INPUT)
StructuredBuffer<uint> OmmArray;

layout(binding = OMM_COMPACT_BINDING_INDEX_OUTPUT)
RWStructuredBuffer<uint> IndexOut;

layout(binding = OMM_COMPACT_BINDING_TRIANGLE_OUTPUT)
RWStructuredBuffer<uint> TriangleOut;

layout(binding = OMM_COMPACT_BINDING_COUNTER)
RWStructuredBuffer<uint> Counter;

groupshared uint g_allTransparent;
groupshared uint g_allOpaque;

[shader("compute")]
[numthreads(OMM_COMPACT_THREADS_PER_GROUP, 1, 1)]
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
    for (uint w = threadId; w < cb.wordsPerTriangle; w += OMM_COMPACT_THREADS_PER_GROUP)
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
    if (g_allTransparent != 0)
    {
      IndexOut[triangleIndex] = OMM_SPECIAL_INDEX_FULLY_TRANSPARENT;
    }
    else if (g_allOpaque != 0)
    {
      IndexOut[triangleIndex] = OMM_SPECIAL_INDEX_FULLY_OPAQUE;
    }
    else
    {
      uint slot;
      InterlockedAdd(Counter[0], 1u, slot);
      IndexOut[triangleIndex] = slot;
      TriangleOut[slot * 2 + 0] = triangleIndex * cb.wordsPerTriangle * 4u; // dataOffset in bytes
      TriangleOut[slot * 2 + 1] = cb.triangleWord1;
    }
  }
}
