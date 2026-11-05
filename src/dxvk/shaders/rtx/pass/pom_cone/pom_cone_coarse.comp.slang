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

// Coarse conservative cones, one per cell of S = 2^coarseShift texels, written to every texel of a map
// right after its height copy, so the cone march applies while pom_cone_generate refines the map tile by tile. Depth
// d = 1 - height, ratios in UV per unit of depth, capped at 1, as in pom_cone_generate.comp.slang.
// A cell's cone must hold for every texel p of the cell: no texel q with d(q) < d(p) may lie inside it, i.e.
// ratio <= |q - p| / (d(p) - d(q)). With D = the largest depth in the cell, d(p) - d(q) <= D - d(q) for every p, so:
// - Near (q in the 3x3 cells around the cell, B): the path of 8-neighbour steps from p to q that goes diagonally first
//   stays in B and takes cheb(q - p) steps, each changing the depth by at most g = the largest depth difference
//   between 8-neighbours in B. Also d(q) >= m = the least depth in B. So d(p) - d(q) <= min(g cheb, D - m) and
//   |q - p| >= cheb >= 1 texel: ratio <= texel * max(1 / g, 1 / (D - m)) bounds every such q.
// - Far: a cell N of the max-height pyramid holds no texel shallower than its least depth and no texel nearer to the
//   cell than their rectangles' gap, so gap / (D - least depth) bounds every q in N. Rings of cells 2..R around the
//   cell (level coarseShift) and around its ancestor (every coarser level) cover every q outside B: a level-k cell at
//   offset <= (R - 1) / 2 from the ancestor has its children within offset R one level down. The levels go up to the
//   first where R cells reach a texture width; farther texels are past the cap of 1 UV.
// A conservative cone is also a sound relaxed cone: it contains no point of the field, so a ray in it crosses the
// field zero times, within the relaxed march's limit of one; and a cone step shorter than the relaxed one ends on the
// same ray before it, where at most one crossing can lie. So coarse cones serve both variants, and a texel's coarse
// value may sit next to refined ones of either variant. It handles square power-of-two maps only, because the pyramid
// must map cells to texels exactly. tests/rtx/unit/test_pom_cone_map.cpp holds a CPU copy, so the two must stay in step.

#include "rtx/pass/pom_cone/pom_cone_args.h"

// The max-height pyramid, whose mip 0 is the height map.
layout(binding = POM_CONE_BINDING_SOURCE)
Texture2D<float4> Heights;

// POM_CONE_COARSE_CONES: the cells written by POM_CONE_COARSE_CELLS.
layout(binding = POM_CONE_BINDING_CONE)
Texture2D<float4> Cells;

layout(binding = POM_CONE_BINDING_OUTPUT)
RWTexture2D<float2> Output;

layout(push_constant)
ConstantBuffer<PomConeCoarseArgs> cb;

groupshared float2 cellReduce[POM_CONE_COARSE_GROUP_SIZE * POM_CONE_COARSE_GROUP_SIZE];

// Power-of-two extents only.
int2 wrapTexel(int2 texel, int2 levelExtent)
{
  return texel & (levelExtent - 1);
}

float depthAt(int2 texel, int level, int2 levelExtent)
{
  return 1.0f - Heights.Load(int3(wrapTexel(texel, levelExtent), level)).r;
}

// One group per cell: the largest 8-neighbour depth difference over the pairs whose first texel is in the cell (each
// unordered pair once: right, down, down-right, down-left), and the cell's largest depth.
void writeCell(uint2 cell, uint2 groupThreadId)
{
  const int2 extent = int2(cb.extent);
  const int cellSize = 1 << int(cb.coarseShift);
  const int2 cellFirst = int2(cell) * cellSize;
  float slope = 0.0f;
  float deepest = 0.0f;
  for (int y = int(groupThreadId.y); y < cellSize; y += POM_CONE_COARSE_GROUP_SIZE)
  {
    for (int x = int(groupThreadId.x); x < cellSize; x += POM_CONE_COARSE_GROUP_SIZE)
    {
      const int2 texel = cellFirst + int2(x, y);
      const float depth = depthAt(texel, 0, extent);
      deepest = max(deepest, depth);
      slope = max(slope, abs(depth - depthAt(texel + int2(1, 0), 0, extent)));
      slope = max(slope, abs(depth - depthAt(texel + int2(0, 1), 0, extent)));
      slope = max(slope, abs(depth - depthAt(texel + int2(1, 1), 0, extent)));
      slope = max(slope, abs(depth - depthAt(texel + int2(-1, 1), 0, extent)));
    }
  }

  const uint index = groupThreadId.y * POM_CONE_COARSE_GROUP_SIZE + groupThreadId.x;
  cellReduce[index] = float2(slope, deepest);
  GroupMemoryBarrierWithGroupSync();
  for (uint stride = (POM_CONE_COARSE_GROUP_SIZE * POM_CONE_COARSE_GROUP_SIZE) / 2; stride > 0; stride >>= 1)
  {
    if (index < stride)
    {
      cellReduce[index] = max(cellReduce[index], cellReduce[index + stride]);
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (index == 0)
  {
    Output[cell] = cellReduce[0];
  }
}

// One thread per cell: the cell's cone (see the top of the file).
void writeCone(uint2 cell)
{
  const int shift = int(cb.coarseShift);
  const int2 extent = int2(cb.extent);
  const int2 cellsExtent = int2(extent.x >> shift, extent.y >> shift);
  if (any(int2(cell) >= cellsExtent))
  {
    return;
  }

  const int2 c = int2(cell);
  const float texelUv = 1.0f / float(extent.x);
  const float deepest = Cells.Load(int3(c, 0)).y;
  float ratio = 1.0f;

  // A cell at the top everywhere has nothing above it.
  if (deepest > 0.0f)
  {
    float slope = 0.0f;
    float least = 1.0f;
    for (int dy = -1; dy <= 1; ++dy)
    {
      for (int dx = -1; dx <= 1; ++dx)
      {
        slope = max(slope, Cells.Load(int3(wrapTexel(c + int2(dx, dy), cellsExtent), 0)).x);
        least = min(least, depthAt(c + int2(dx, dy), shift, cellsExtent));
      }
    }
    if (least < deepest && slope > 0.0f)
    {
      ratio = min(ratio, texelUv * max(1.0f / slope, 1.0f / (deepest - least)));
    }

    const int cellSize = 1 << shift;
    const int2 cellFirst = c * cellSize;
    const int2 cellLast = cellFirst + int2(cellSize - 1, cellSize - 1);
    const int ring = POM_CONE_COARSE_RING;
    for (int level = shift; level <= int(cb.topLevel); ++level)
    {
      const int up = level - shift;
      const int size = 1 << level;
      const int2 levelExtent = max(int2(extent.x >> level, extent.y >> level), int2(1, 1));
      const int2 ancestor = int2(c.x >> up, c.y >> up);
      // Offsets up to (ring - 1) / 2 are covered one level down (the block B at the first level).
      const int inner = max(2, (ring - 1) / 2 + 1);
      for (int cy = -ring; cy <= ring; ++cy)
      {
        for (int cx = -ring; cx <= ring; ++cx)
        {
          if (max(abs(cx), abs(cy)) < inner)
          {
            continue;
          }
          const int2 other = ancestor + int2(cx, cy);
          const float leastDepth = depthAt(other, level, levelExtent);
          if (leastDepth >= deepest)
          {
            continue;
          }
          const int2 otherFirst = other * size;
          const int2 otherLast = otherFirst + int2(size - 1, size - 1);
          const int2 gap = max(max(otherFirst - cellLast, cellFirst - otherLast), int2(0, 0));
          ratio = min(ratio, length(float2(gap)) * texelUv / (deepest - leastDepth));
        }
      }
    }
  }

  const float cone = floor(sqrt(saturate(ratio)) * cb.coneQuantization) / cb.coneQuantization;
  Output[cell] = float2(cone, 0.0f);
}

[shader("compute")]
[numthreads(POM_CONE_COARSE_GROUP_SIZE, POM_CONE_COARSE_GROUP_SIZE, 1)]
void main(uint3 threadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
  // Uniform over the dispatch, so the group barriers of writeCell are reached by every thread of a group.
  if (cb.mode == POM_CONE_COARSE_CELLS)
  {
    writeCell(groupId.xy, groupThreadId.xy);
  }
  else
  {
    writeCone(threadId.xy);
  }
}
