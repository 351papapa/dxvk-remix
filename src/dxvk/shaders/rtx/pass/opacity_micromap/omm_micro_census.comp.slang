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

// rtx.alphaBake.censusMicroStates. One workgroup per triangle of a baked opacity micromap array.
// Each thread popcounts its words. In 4-state the even bits are the low state bit and the odd bits the high one,
// so T = 00, O = 01, UT = 10, UO = 11. The group sums the four counts, and thread 0 adds them, the triangle and its
// per-state shares to the bin of the triangle's class (uniform when all T or all O, as omm_compact.comp.slang decides,
// and mixed otherwise).

#include "omm_micro_census_args.h"

layout(push_constant)
ConstantBuffer<OmmMicroCensusArgs> cb;

layout(binding = OMM_MICRO_CENSUS_BINDING_ARRAY_INPUT)
StructuredBuffer<uint> OmmArray;

layout(binding = OMM_MICRO_CENSUS_BINDING_COUNTERS)
RWStructuredBuffer<uint> Counters;

groupshared uint g_stateCount[4];

// A 64-bit add on a (low, high) pair, where the high word takes the carry of the low one.
void add64(uint index, uint value)
{
  if (value == 0u)
  {
    return;
  }
  uint previous;
  InterlockedAdd(Counters[index], value, previous);
  if (previous + value < previous)
  {
    uint unused;
    InterlockedAdd(Counters[index + 1u], 1u, unused);
  }
}

[shader("compute")]
[numthreads(OMM_MICRO_CENSUS_THREADS_PER_GROUP, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint threadId : SV_GroupThreadID)
{
  const uint triangleIndex = groupId.y * cb.groupsX + groupId.x;

  if (threadId < 4u)
  {
    g_stateCount[threadId] = 0u;
  }
  GroupMemoryBarrierWithGroupSync();

  if (triangleIndex < cb.numTriangles)
  {
    const uint base = triangleIndex * cb.wordsPerTriangle;
    uint count[4] = { 0u, 0u, 0u, 0u };
    for (uint w = threadId; w < cb.wordsPerTriangle; w += OMM_MICRO_CENSUS_THREADS_PER_GROUP)
    {
      const uint word = OmmArray[base + w];
      if (cb.twoState != 0u)
      {
        const uint opaque = countbits(word);
        count[1] += opaque;
        count[0] += 32u - opaque;
      }
      else
      {
        const uint lo = word & 0x55555555u;
        const uint hi = (word >> 1u) & 0x55555555u;
        const uint opaque = countbits(lo & ~hi);
        const uint unknownTransparent = countbits(hi & ~lo);
        const uint unknownOpaque = countbits(lo & hi);
        count[1] += opaque;
        count[2] += unknownTransparent;
        count[3] += unknownOpaque;
        count[0] += 16u - opaque - unknownTransparent - unknownOpaque;
      }
    }
    for (uint s = 0u; s < 4u; ++s)
    {
      if (count[s] != 0u)
      {
        InterlockedAdd(g_stateCount[s], count[s]);
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if (threadId == 0u && triangleIndex < cb.numTriangles)
  {
    const uint total = g_stateCount[0] + g_stateCount[1] + g_stateCount[2] + g_stateCount[3];
    const uint classIndex = (g_stateCount[0] == total || g_stateCount[1] == total) ? 0u : 1u;
    const uint level = min(cb.level, OMM_MICRO_CENSUS_LEVELS - 1u);
    const uint bin = (classIndex * 2u + (cb.twoState != 0u ? 1u : 0u)) * OMM_MICRO_CENSUS_LEVELS + level;
    const uint binBase = bin * OMM_MICRO_CENSUS_BIN_WORDS;
    const float shareScale = float(OMM_MICRO_CENSUS_SHARE_ONE) / float(max(total, 1u));
    for (uint s = 0u; s < 4u; ++s)
    {
      add64(binBase + 2u * s, g_stateCount[s]);
      add64(binBase + OMM_MICRO_CENSUS_BIN_SHARES + 2u * s, uint(float(g_stateCount[s]) * shareScale + 0.5f));
    }
    add64(binBase + OMM_MICRO_CENSUS_BIN_TRIANGLES, 1u);
  }
}
