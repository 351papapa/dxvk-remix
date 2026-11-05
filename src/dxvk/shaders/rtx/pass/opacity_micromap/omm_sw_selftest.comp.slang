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

// rtx.alphaBake.softwareOmmSelfTest. One thread per micro-triangle d of every subdivision level 0..12 checks that
// the centroid of micro-triangle d, as the bake places it (micromeshDistToBary +
// calculateMicrotriangleTexcoords), must map back to d through the lookup's softOmmBarycentricsToIndex. Results[2 * level]
// counts mismatches. Thread 0 of each level also checks the clamping edges (the three corners map to the corner
// micro-triangles, and points outside the triangle to a valid index). Results[2 * level + 1] counts failed edge checks.

#include "rtx/pass/opacity_micromap/software_omm_shared.h"
#include "rtx/pass/opacity_micromap/bake_opacity_micromap_utils.slangh"
#include "rtx/algorithm/software_omm_index.slangh"

layout(push_constant)
ConstantBuffer<SoftOmmSelfTestArgs> cb;

layout(binding = SOFT_OMM_SELFTEST_BINDING_RESULTS)
RWStructuredBuffer<uint> Results;

// The micro-triangle at the given discrete barycentrics (upright or flipped) is the one micromeshDistToBary(d) names.
bool softOmmSelfTestIs(uint d, uint level, uint expectedU, uint expectedV, bool expectedUpright)
{
  const uint3 b = micromeshDistToBary(d, level);
  const bool upright = ((b.x ^ b.y ^ b.z) & 1u) != 0u;
  return b.x == expectedU && b.y == expectedV && upright == expectedUpright;
}

[shader("compute")]
[numthreads(SOFT_OMM_SELFTEST_THREADS_PER_GROUP, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint threadId : SV_GroupThreadID)
{
  const uint thread = (groupId.y * cb.groupsX + groupId.x) * SOFT_OMM_SELFTEST_THREADS_PER_GROUP + threadId;
  if (thread >= cb.totalThreads)
  {
    return;
  }

  // The level of this thread. Levels are laid out one after another, with 4^level threads each.
  uint level = 0u;
  uint levelStart = 0u;
  while (level < SOFT_OMM_SELFTEST_MAX_LEVEL && thread >= levelStart + (1u << (2u * level)))
  {
    levelStart += 1u << (2u * level);
    ++level;
  }
  const uint d = thread - levelStart;
  const uint numSteps = 1u << level;
  const float rcpNumSteps = 1.0f / float(numSteps);

  // The centroid of micro-triangle d in (u, v) = (weight of vertex 1, weight of vertex 2), as the bake computes it.
  const uint3 b = micromeshDistToBary(d, level);
  const bool flipped = ((b.x ^ b.y ^ b.z) & 1u) == 0u;
  const float2 centroid = flipped
    ? float2(float(3u * b.x + 2u), float(3u * b.y + 2u)) * (rcpNumSteps / 3.0f)
    : float2(float(3u * b.x + 1u), float(3u * b.y + 1u)) * (rcpNumSteps / 3.0f);

  if (softOmmBarycentricsToIndex(centroid, level) != d)
  {
    InterlockedAdd(Results[2u * level], 1u);
  }

  if (d == 0u)
  {
    uint edgeFailures = 0u;
    const uint count = numSteps * numSteps;
    // The corners vertex 0 (u = v = 0), vertex 1 (u = 1) and vertex 2 (v = 1) are in the upright corner micro-triangles.
    edgeFailures += softOmmSelfTestIs(softOmmBarycentricsToIndex(float2(0.0f, 0.0f), level), level, 0u, 0u, true) ? 0u : 1u;
    edgeFailures += softOmmSelfTestIs(softOmmBarycentricsToIndex(float2(1.0f, 0.0f), level), level, numSteps - 1u, 0u, true) ? 0u : 1u;
    edgeFailures += softOmmSelfTestIs(softOmmBarycentricsToIndex(float2(0.0f, 1.0f), level), level, 0u, numSteps - 1u, true) ? 0u : 1u;
    // Points outside the triangle (clamped) and on the far edge give a valid index.
    const float2 outside[6] = { float2(1.0f, 1.0f), float2(-1.0f, 0.5f), float2(2.0f, 2.0f), float2(0.5f, 0.5f),
                                float2(1e-7f, 0.9999999f), float2(0.9999999f, 1e-7f) };
    [unroll]
    for (uint i = 0u; i < 6u; ++i)
    {
      edgeFailures += softOmmBarycentricsToIndex(outside[i], level) < count ? 0u : 1u;
    }
    if (edgeFailures != 0u)
    {
      InterlockedAdd(Results[2u * level + 1u], edgeFailures);
    }
  }
}
