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

// Cone ratios (GPU Gems 3, ch. 18) for one tile of a height map. Depth d = 1 - height. The cone of texel p
// has its apex at (p, d(p)) and opens upwards. Its ratio is in UV per unit of depth, capped at 1, because a texel one
// texture width or more away cannot bring it below 1. A texel q above p (d(q) < d(p)) bounds the ratio by:
// - its conservative bound |q - p| / (d(p) - d(q)), with which the cone holds no point of the field.
// - its relaxed bound. The ray from (p, 0) through (q, d(q)) enters the field at q, walking on while it stays inside
//   gives its exit E, and the cone must not contain E: |E - p| / (d(p) - E.depth) wherever E is above d(p). E lies on
//   the ray beyond q (parameter t >= 1, depth t d(q)), and t |q - p| / (d(p) - t d(q)) grows with t, so a texel's
//   relaxed bound is never below its conservative one.
// 1. Near field: the texels within nearRadius of p (Chebyshev) get their relaxed bound, the walk one texel per step and
//    at most rayStepsPerCandidate steps (0 steps give the conservative bound). A walk cut short keeps its last point
//    inside the field, which only lowers the bound. It is skipped on square power-of-two maps when rayStepsPerCandidate
//    is 0, because the search below finds those conservative bounds itself with fewer reads.
// 2. Square power-of-two maps (farLevels = log2(width): the max-height pyramid reaches 1x1): every other texel gets
//    its conservative bound, found by a branch-and-bound search of the pyramid instead of reading every texel. A cell N
//    holds no texel nearer to p than its rectangle and none shallower than its least depth m(N), so
//    b(N) = dist(p, N) / (d(p) - m(N)) is at most the bound of any texel in N, relaxed or conservative. Cells are
//    visited depth first from the top level down, the child nearest p first. A cell is refined only while b(N) is below
//    T = (floor(sqrt(ratio) Q) / Q)^2 (times 1.0001 against rounding): a bound of T or more cannot change the stored,
//    rounded-down cone. A texel (level 0) gives its exact bound. So the stored cone is exactly the brute-force minimum
//    of the near field's bounds and every other texel's conservative bound (for the conservative variant, the exact
//    conservative cone). After POM_CONE_MAX_VISITS reads, cells are no longer refined and their b(N) is taken as is.
//    It is lower than the bound of any texel in them, so the cone can only get narrower, which only very smooth maps reach.
//    The window, when every bound is conservative, is the nearest image of each texel (the W consecutive texels per axis
//    from p - W/2), since a farther image has the same depth farther away. With a near field it is every image within
//    one texture width, since a near texel's other images are not in the near field and still need their bound.
// 3. Other maps (no exact cell-to-texel mapping): every texel outside the near field is bounded by its distance alone,
//    (nearRadius + 1) texels / d(p).
// The conservative variant (PomConeMapManager::ConeVariant::Conservative, terrain layers) has rayStepsPerCandidate = 0,
// so every bound is conservative and the cone holds no point of the field. Lookups wrap, as textures tile.
// tests/rtx/unit/test_pom_cone_map.cpp holds a CPU copy of this kernel and checks it against brute-force references,
// so the two must stay in step.

#include "rtx/pass/pom_cone/pom_cone_args.h"

// The max-height pyramid, whose mip 0 is the height map.
layout(binding = POM_CONE_BINDING_SOURCE)
Texture2D<float4> Heights;

layout(binding = POM_CONE_BINDING_OUTPUT)
RWTexture2D<float> Cones;

layout(push_constant)
ConstantBuffer<PomConeGenerateArgs> cb;

float depthAt(int2 texel, int level, int2 levelExtent)
{
  // Power-of-two sizes (most height maps) wrap with a mask: the same texel as the modulo, without two integer
  // divisions per read.
  int2 wrapped;
  if (all((levelExtent & (levelExtent - 1)) == int2(0, 0)))
  {
    wrapped = texel & (levelExtent - 1);
  }
  else
  {
    wrapped = ((texel % levelExtent) + levelExtent) % levelExtent;
  }
  return 1.0f - Heights.Load(int3(wrapped, level)).r;
}

// A cell whose bound is at least this cannot change the cone stored for `ratio` (rounded down to the cone texture's
// quantisation), so it is not refined. Raised by a margin, so rounding never prunes a cell that could lower it.
float pruneThreshold(float ratio)
{
  const float stored = floor(sqrt(saturate(ratio)) * cb.coneQuantization) / cb.coneQuantization;
  return stored * stored * 1.0001f;
}

// 1. The near field: relaxed bounds (conservative ones without a walk) of the texels within radius of p.
float nearField(int2 p, float depthP, int radius, int2 extent, float2 texelUv, float ratio)
{
  for (int dy = -radius; dy <= radius; ++dy)
  {
    for (int dx = -radius; dx <= radius; ++dx)
    {
      if (dx == 0 && dy == 0)
      {
        continue;
      }
      const float2 offset = float2(float(dx), float(dy));
      const float depthQ = depthAt(p + int2(dx, dy), 0, extent);
      const float span = float(max(abs(dx), abs(dy)));
      // The ray is at p + t * offset and depth t * depthQ, with t = 1 at q.
      float t = 1.0f;
      for (uint i = 1; i <= cb.rayStepsPerCandidate; ++i)
      {
        const float tNext = 1.0f + float(i) / span;
        const float depthNext = tNext * depthQ;
        // Below the deepest the field goes, the ray is inside for good.
        if (depthNext > 1.0f)
        {
          break;
        }
        if (depthAt(p + int2(floor(offset * tNext + 0.5f)), 0, extent) > depthNext)
        {
          break;
        }
        t = tNext;
      }
      const float exitDepth = t * depthQ;
      if (exitDepth < depthP)
      {
        ratio = min(ratio, length(offset * t * texelUv) / (depthP - exitDepth));
      }
    }
  }
  return ratio;
}

// 2. Square power-of-two maps: the conservative bounds of every texel outside the near field (Chebyshev distance from p
// above nearRadius, or every texel when it is 0), by branch and bound over the max-height pyramid, levels 0 to topLevel (1x1).
float pyramidSearch(int2 p, float depthP, int nearRadius, int width, int topLevel, float ratio)
{
  const float texelUv = 1.0f / float(width);
  int2 windowFirst;
  int2 windowLast;
  if (nearRadius > 0)
  {
    windowFirst = p - int2(width - 1, width - 1);
    windowLast = p + int2(width - 1, width - 1);
  }
  else
  {
    windowFirst = p - int2(width / 2, width / 2);
    windowLast = windowFirst + int2(width - 1, width - 1);
  }
  // The roots: the top-level cells (one texture each) that meet the window, offsets -1 to 1 per axis. p lies in cell 0,
  // which goes first, so the search tightens near p before it reaches the other images.
  const int2 rootFirst = int2(windowFirst.x >> topLevel, windowFirst.y >> topLevel);
  const int2 rootLast = int2(windowLast.x >> topLevel, windowLast.y >> topLevel);
  float threshold = pruneThreshold(ratio);
  int visits = 0;
  for (int iy = 0; iy < 3; ++iy)
  {
    const int rootY = iy == 0 ? 0 : (iy == 1 ? -1 : 1);
    if (rootY < rootFirst.y || rootY > rootLast.y)
    {
      continue;
    }
    for (int ix = 0; ix < 3; ++ix)
    {
      const int rootX = ix == 0 ? 0 : (ix == 1 ? -1 : 1);
      if (rootX < rootFirst.x || rootX > rootLast.x)
      {
        continue;
      }
      // A depth-first walk of the root's quadtree without a stack: a cell's parent and siblings follow from its
      // coordinates, and the order of siblings from where p lies.
      int2 cell = int2(rootX, rootY);
      int level = topLevel;
      while (true)
      {
        const int size = 1 << level;
        const int2 first = cell * size;
        const int2 last = first + int2(size - 1, size - 1);
        bool descend = false;
        // A texel of the near field has its relaxed bound already (with no near field: p itself, which bounds nothing).
        const bool nearTexel = level == 0 && all(abs(cell - p) <= int2(nearRadius, nearRadius));
        if (all(last >= windowFirst) && all(first <= windowLast) && !nearTexel)
        {
          ++visits;
          const int levelMask = (width >> level) - 1;
          const float leastDepth = 1.0f - Heights.Load(int3(cell & int2(levelMask, levelMask), level)).r;
          if (leastDepth < depthP)
          {
            const int2 nearest = clamp(p, first, last);
            const float bound = length(float2(nearest - p) * texelUv) / (depthP - leastDepth);
            if (bound < threshold)
            {
              if (level > 0 && visits < POM_CONE_MAX_VISITS)
              {
                descend = true;
              }
              else
              {
                ratio = min(ratio, bound);
                threshold = pruneThreshold(ratio);
              }
            }
          }
        }
        if (descend)
        {
          // The child nearest p.
          --level;
          const int halfSize = 1 << level;
          cell = cell * 2 + int2(p.x >= first.x + halfSize ? 1 : 0, p.y >= first.y + halfSize ? 1 : 0);
          continue;
        }
        // The next sibling in nearest-first order, climbing out of every parent whose last child this was.
        while (level < topLevel)
        {
          const int halfSize = 1 << level;
          const int2 parentFirst = int2((cell.x >> 1) * (2 * halfSize), (cell.y >> 1) * (2 * halfSize));
          const int2 nearSide = int2(p.x >= parentFirst.x + halfSize ? 1 : 0, p.y >= parentFirst.y + halfSize ? 1 : 0);
          const int2 sideBits = (cell & int2(1, 1)) ^ nearSide;
          const int order = sideBits.x | (sideBits.y << 1);
          if (order < 3)
          {
            const int next = order + 1;
            cell = (cell & int2(~1, ~1)) | (int2(next & 1, next >> 1) ^ nearSide);
            break;
          }
          cell = int2(cell.x >> 1, cell.y >> 1);
          ++level;
        }
        if (level == topLevel)
        {
          break;
        }
      }
    }
  }
  return ratio;
}

[shader("compute")]
[numthreads(POM_CONE_GENERATE_GROUP_SIZE, POM_CONE_GENERATE_GROUP_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  const uint2 texel = cb.tileOrigin + threadId;
  if (any(texel >= cb.extent))
  {
    return;
  }

  const int2 extent = int2(cb.extent);
  const int2 p = int2(texel);
  const float2 texelUv = float2(1.0f, 1.0f) / float2(cb.extent);
  const int radius = int(cb.nearRadius);
  const float depthP = depthAt(p, 0, extent);
  float ratio = 1.0f;

  // A texel at the top has nothing above it.
  if (depthP > 0.0f)
  {
    const bool hasPyramid = cb.farLevels > 0;
    const int nearRadius = (hasPyramid && cb.rayStepsPerCandidate == 0) ? 0 : radius;
    ratio = nearField(p, depthP, nearRadius, extent, texelUv, ratio);
    if (hasPyramid)
    {
      ratio = pyramidSearch(p, depthP, nearRadius, extent.x, int(cb.farLevels), ratio);
    }
    else
    {
      ratio = min(ratio, float(radius + 1) * min(texelUv.x, texelUv.y) / depthP);
    }
  }

  Cones[texel] = floor(sqrt(saturate(ratio)) * cb.coneQuantization) / cb.coneQuantization;
}
