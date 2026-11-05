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

// rtx.alphaBake.softwareOmm. One thread per word of the software store (see omm_sw_pack_args.h).

#include "omm_sw_pack_args.h"

layout(push_constant)
ConstantBuffer<OmmSwPackArgs> cb;

layout(binding = OMM_SW_PACK_BINDING_ARRAY_INPUT)
StructuredBuffer<uint> OmmArray;

layout(binding = OMM_SW_PACK_BINDING_INDEX_INPUT)
StructuredBuffer<uint> IndexIn;

layout(binding = OMM_SW_PACK_BINDING_TRIANGLE_INPUT)
StructuredBuffer<uint> TriangleIn;

layout(binding = OMM_SW_PACK_BINDING_STORE_OUTPUT)
RWStructuredBuffer<uint> StoreOut;

[shader("compute")]
[numthreads(OMM_SW_PACK_THREADS_PER_GROUP, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint threadId : SV_GroupThreadID)
{
  const uint word = (groupId.y * cb.groupsX + groupId.x) * OMM_SW_PACK_THREADS_PER_GROUP + threadId;
  if (word >= cb.totalWords)
  {
    return;
  }

  if (word < cb.numTriangles)
  {
    StoreOut[word] = IndexIn[word];
    return;
  }

  const uint dataWord = word - cb.numTriangles;
  const uint slot = dataWord / cb.wordsPerTriangle;
  const uint wordInTriangle = dataWord - slot * cb.wordsPerTriangle;
  const uint sourceWord = TriangleIn[slot * 2u] / 4u + wordInTriangle;
  StoreOut[word] = OmmArray[sourceWord];
}
