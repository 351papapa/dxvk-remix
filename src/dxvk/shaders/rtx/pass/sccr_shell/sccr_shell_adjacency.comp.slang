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

// Welded adjacency and inner-shell directions of one geometry (SccrShellManager, rtx_fork_sccr_shell.cpp).
// The layout and phases are in sccr_shell_args.h. As in smooth_normals.comp.slang, positions weld by their exact float
// bits (-0 as +0), or after rounding to a grid (rtx.displacement.shell.weldEpsilon), through a linearly probed hash
// table claimed with atomics. A slot holds the whole position (all three coordinates' bits) rather than a 32-bit tag,
// because tags of regular coordinates (0.25, 0.5, ...) collide and would weld unrelated corners together.
// Unlike smooth normals, the direction sum is weighted by each corner's angle, which is bounded per triangle so the
// fixed-point sum cannot overflow, and does not depend on how a surface is tessellated. Every welded position also gets
// a canonical id (its smallest vertex index), which the edge table keys on, and the fold guard's depth limits (phase 5).
// tests/rtx/unit/test_sccr_shell.cpp runs a CPU copy of these phases, so the two must stay in step.

#include "rtx/pass/sccr_shell/sccr_shell_args.h"

layout(binding = SCCR_SHELL_BINDING_POSITION_RO)
StructuredBuffer<float> PositionData;

layout(binding = SCCR_SHELL_BINDING_INDEX_RO)
StructuredBuffer<uint> IndexData;

layout(binding = SCCR_SHELL_BINDING_VERTEX_TABLE)
RWStructuredBuffer<uint> VertexTable;

layout(binding = SCCR_SHELL_BINDING_EDGE_TABLE)
RWStructuredBuffer<uint> EdgeTable;

layout(binding = SCCR_SHELL_BINDING_OUTPUT)
RWStructuredBuffer<uint> Output;

layout(push_constant)
ConstantBuffer<SccrShellBuildArgs> cb;

static const uint kSccrShellMaxProbes = 128u;
static const uint kSccrShellInvalidSlot = 0xFFFFFFFFu;
// The fixed-point scale of the direction sum. A corner adds at most this per component (its angle / pi <= 1), so a
// position can gather over 200000 corners before an int32 overflows.
static const float kSccrShellFixedPointScale = 10000.0f;
// A welded sum shorter than this (in units of one corner's full weight) is degenerate. A flat interior vertex sums to
// about 2, and the two faces of a fin cancel.
static const float kSccrShellDegenerateSum = 0.05f;
static const float kSccrShellPi = 3.14159265358979f;
// The fold phase skips a triangle whose welded directions already lie within this cosine of its plane (or behind it)
// at depth 0, because its prism is degenerate or inverted at any depth. It matches the march's kSccrShellMinCosine.
static const float kSccrShellFoldMinCosine = 1e-3f;

uint sccrShellLoadIndex(uint indexIndex)
{
  if (cb.indexStride == 0u)
  {
    return indexIndex;
  }
  if (cb.indexStride == 4u)
  {
    return IndexData[(cb.indexOffset >> 2u) + indexIndex];
  }
  const uint byteOffset = cb.indexOffset + indexIndex * 2u;
  return (IndexData[byteOffset >> 2u] >> ((byteOffset & 2u) * 8u)) & 0xFFFFu;
}

float3 sccrShellLoadPosition(uint vertexIndex)
{
  const uint base = (cb.positionOffset + vertexIndex * cb.positionStride) >> 2u;
  return float3(PositionData[base + 0u], PositionData[base + 1u], PositionData[base + 2u]);
}

// -0 hashes as +0, so copies of a position always produce the same bits (as smooth_normals.h does).
uint sccrShellFloatBits(float value)
{
  return value == 0.0f ? 0u : asuint(value);
}

// The words a slot stores for a position are 1 + its coordinates' bits, and 0 marks an empty word. The all-ones NaN
// keeps its bits and shares a word value with the next NaN, which only welds NaN corners together.
uint3 sccrShellPositionKey(float3 position)
{
  const uint3 bits = uint3(sccrShellFloatBits(position.x), sccrShellFloatBits(position.y), sccrShellFloatBits(position.z));
  return uint3(bits.x == 0xFFFFFFFFu ? bits.x : bits.x + 1u, bits.y == 0xFFFFFFFFu ? bits.y : bits.y + 1u,
               bits.z == 0xFFFFFFFFu ? bits.z : bits.z + 1u);
}

// Mixed well, so regular coordinates (whose low mantissa bits are all 0) still spread over the table.
uint sccrShellPositionSlot(uint3 key, uint tableSize)
{
  uint h = key.x * 0x9E3779B1u;
  h ^= key.y + 0x7F4A7C15u + (h << 6u) + (h >> 2u);
  h ^= key.z + 0x7F4A7C15u + (h << 6u) + (h >> 2u);
  h ^= h >> 16u;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13u;
  h *= 0xC2B2AE35u;
  h ^= h >> 16u;
  return h & (tableSize - 1u);
}

// The point a position welds by is itself (exact bits) or the nearest point of the weld grid (SCCR_SHELL_WELD_GRID).
// Two positions in one grid cell give the same bits. Two that straddle a cell boundary do not weld, which is rare and
// only leaves an open edge.
float3 sccrShellWeldKey(float3 position)
{
  if (cb.weldCell > 0.0f)
  {
    return round(position / cb.weldCell) * cb.weldCell;
  }
  return position;
}

uint sccrShellEdgeSlot(uint smallerId, uint largerId, uint tableSize)
{
  uint h = smallerId * 0x9E3779B1u;
  h ^= largerId + 0x7F4A7C15u + (h << 6u) + (h >> 2u);
  h ^= h >> 15u;
  h *= 0x2C1B3C6Du;
  h ^= h >> 13u;
  return h & (tableSize - 1u);
}

// Returns the slot of a welded position, which triangle claimingTriangle claims if it is new.
// Returns kSccrShellInvalidSlot on overflow.
uint sccrShellVertexInsert(float3 position, uint claimingTriangle)
{
  const uint3 key = sccrShellPositionKey(sccrShellWeldKey(position));
  const uint start = sccrShellPositionSlot(key, cb.vertexTableSize);
  const uint probes = min(kSccrShellMaxProbes, cb.vertexTableSize);
  for (uint p = 0u; p < probes; ++p)
  {
    const uint slot = (start + p) & (cb.vertexTableSize - 1u);
    const uint base = slot * SCCR_SHELL_VERTEX_SLOT_DWORDS;
    // As in the edge table, a thread writes a word only after the words before it matched its own, so the three words
    // always end up holding one position, that of the thread that wrote the last. A thread that finds a different
    // coordinate in a word moves on, and that slot belongs to the other position.
    uint previous;
    InterlockedCompareExchange(VertexTable[base + 0u], 0u, key.x, previous);
    if (previous != 0u && previous != key.x)
    {
      continue;
    }
    InterlockedCompareExchange(VertexTable[base + 8u], 0u, key.y, previous);
    if (previous != 0u && previous != key.y)
    {
      continue;
    }
    InterlockedCompareExchange(VertexTable[base + 9u], 0u, key.z, previous);
    if (previous == 0u)
    {
      // Only the thread that completed the position writes this word, and only phase 4 reads it.
      VertexTable[base + 6u] = claimingTriangle + 1u;
      return slot;
    }
    if (previous == key.z)
    {
      return slot;
    }
  }
  return kSccrShellInvalidSlot;
}

uint sccrShellVertexLookup(float3 position)
{
  const uint3 key = sccrShellPositionKey(sccrShellWeldKey(position));
  const uint start = sccrShellPositionSlot(key, cb.vertexTableSize);
  const uint probes = min(kSccrShellMaxProbes, cb.vertexTableSize);
  for (uint p = 0u; p < probes; ++p)
  {
    const uint slot = (start + p) & (cb.vertexTableSize - 1u);
    const uint base = slot * SCCR_SHELL_VERTEX_SLOT_DWORDS;
    const uint stored = VertexTable[base + 0u];
    if (stored == 0u)
    {
      break;
    }
    if (stored == key.x && VertexTable[base + 8u] == key.y && VertexTable[base + 9u] == key.z)
    {
      return slot;
    }
  }
  return kSccrShellInvalidSlot;
}

// The welded id of a position (its smallest vertex index) and its slot. False when the position did not weld.
bool sccrShellWeldedId(float3 position, out uint weldedId, out uint slot)
{
  slot = sccrShellVertexLookup(position);
  weldedId = 0u;
  if (slot == kSccrShellInvalidSlot)
  {
    return false;
  }
  weldedId = ~VertexTable[slot * SCCR_SHELL_VERTEX_SLOT_DWORDS + 1u];
  return true;
}

// The welded direction of a slot after phase 1. It is the normalised angle-weighted sum, or the face normal of the
// triangle that claimed the position where the sum is degenerate, as at a fin or a two-sided sheet.
float3 sccrShellSlotDirection(uint slot, out bool degenerate)
{
  const uint base = slot * SCCR_SHELL_VERTEX_SLOT_DWORDS;
  const float3 sum = float3(int3(int(VertexTable[base + 2u]), int(VertexTable[base + 3u]), int(VertexTable[base + 4u])));
  const float sumLength = length(sum);
  degenerate = !(sumLength > kSccrShellDegenerateSum * kSccrShellFixedPointScale);
  if (!degenerate)
  {
    return sum / sumLength;
  }
  float3 direction = float3(0.0f, 0.0f, 1.0f);
  const uint claimingTriangle = VertexTable[base + 6u];
  if (claimingTriangle != 0u)
  {
    const uint triIndex = claimingTriangle - 1u;
    const float3 p0 = sccrShellLoadPosition(sccrShellLoadIndex(triIndex * 3u + 0u));
    const float3 p1 = sccrShellLoadPosition(sccrShellLoadIndex(triIndex * 3u + 1u));
    const float3 p2 = sccrShellLoadPosition(sccrShellLoadIndex(triIndex * 3u + 2u));
    const float3 faceCross = cross(p1 - p0, p2 - p0);
    const float faceLength = length(faceCross);
    if (faceLength > 0.0f)
    {
      direction = faceCross / faceLength;
    }
  }
  return direction;
}

void sccrShellFlag(uint flags)
{
  InterlockedOr(Output[SCCR_SHELL_HEADER_FLAGS], flags);
}

float sccrShellCornerAngle(float3 a, float3 b)
{
  const float lengths = sqrt(dot(a, a) * dot(b, b));
  if (!(lengths > 0.0f))
  {
    return 0.0f;
  }
  return acos(clamp(dot(a, b) / lengths, -1.0f, 1.0f));
}

// Phase 1.
void sccrShellWeld(uint triIndex)
{
  uint vertexIndices[3];
  float3 positions[3];
  [unroll]
  for (uint k = 0u; k < 3u; ++k)
  {
    vertexIndices[k] = sccrShellLoadIndex(triIndex * 3u + k);
    positions[k] = sccrShellLoadPosition(vertexIndices[k]);
  }
  const float3 faceCross = cross(positions[1] - positions[0], positions[2] - positions[0]);
  const float faceLength = length(faceCross);
  const float3 faceNormal = faceLength > 0.0f ? faceCross / faceLength : float3(0.0f, 0.0f, 0.0f);

  [unroll]
  for (uint k = 0u; k < 3u; ++k)
  {
    const uint slot = sccrShellVertexInsert(positions[k], triIndex);
    if (slot == kSccrShellInvalidSlot)
    {
      sccrShellFlag(SCCR_SHELL_FLAG_WELD_OVERFLOW);
      continue;
    }
    const uint base = slot * SCCR_SHELL_VERTEX_SLOT_DWORDS;
    InterlockedMax(VertexTable[base + 1u], ~vertexIndices[k]);

    const float3 toNext = positions[(k + 1u) % 3u] - positions[k];
    const float3 toPrevious = positions[(k + 2u) % 3u] - positions[k];
    if (faceLength > 0.0f)
    {
      const float weight = sccrShellCornerAngle(toNext, toPrevious) / kSccrShellPi;
      const int3 contribution = int3(faceNormal * (weight * kSccrShellFixedPointScale));
      InterlockedAdd(VertexTable[base + 2u], uint(contribution.x));
      InterlockedAdd(VertexTable[base + 3u], uint(contribution.y));
      InterlockedAdd(VertexTable[base + 4u], uint(contribution.z));
    }
  }
}

// The welded ids of edge slot edgeSlot of a triangle (from its vertex (edgeSlot + 1) % 3 to (edgeSlot + 2) % 3), the
// direction it runs (0: from the smaller id to the larger) and the slots of its two ends. False when either end did not
// weld or the edge collapses to one welded position.
bool sccrShellEdgeKey(
  uint triIndex, uint edgeSlot, out uint smallerId, out uint largerId, out uint edgeDirection, out uint fromSlot, out uint toSlot)
{
  const uint fromIndex = sccrShellLoadIndex(triIndex * 3u + (edgeSlot + 1u) % 3u);
  const uint toIndex = sccrShellLoadIndex(triIndex * 3u + (edgeSlot + 2u) % 3u);
  uint fromId;
  uint toId;
  smallerId = 0u;
  largerId = 0u;
  edgeDirection = 0u;
  if (!sccrShellWeldedId(sccrShellLoadPosition(fromIndex), fromId, fromSlot) ||
      !sccrShellWeldedId(sccrShellLoadPosition(toIndex), toId, toSlot) ||
      fromId == toId)
  {
    return false;
  }
  smallerId = min(fromId, toId);
  largerId = max(fromId, toId);
  edgeDirection = fromId < toId ? 0u : 1u;
  return true;
}

// Phase 2.
void sccrShellInsertEdge(uint edgeIndex)
{
  const uint triIndex = edgeIndex / 3u;
  uint smallerId;
  uint largerId;
  uint edgeDirection;
  uint fromSlot;
  uint toSlot;
  if (!sccrShellEdgeKey(triIndex, edgeIndex % 3u, smallerId, largerId, edgeDirection, fromSlot, toSlot))
  {
    return;
  }
  const uint side = ((edgeIndex << 1u) | edgeDirection) + 1u;
  const uint start = sccrShellEdgeSlot(smallerId, largerId, cb.edgeTableSize);
  const uint probes = min(kSccrShellMaxProbes, cb.edgeTableSize);
  for (uint p = 0u; p < probes; ++p)
  {
    const uint base = ((start + p) & (cb.edgeTableSize - 1u)) * SCCR_SHELL_EDGE_SLOT_DWORDS;
    // A slot belongs to the key whose two words it holds. A thread may claim the first word and then lose the second
    // to another key with the same smaller id. It moves on, and the slot belongs to that other key.
    uint previous;
    InterlockedCompareExchange(EdgeTable[base + 0u], 0u, smallerId + 1u, previous);
    if (previous != 0u && previous != smallerId + 1u)
    {
      continue;
    }
    InterlockedCompareExchange(EdgeTable[base + 1u], 0u, largerId + 1u, previous);
    if (previous != 0u && previous != largerId + 1u)
    {
      continue;
    }
    InterlockedCompareExchange(EdgeTable[base + 2u], 0u, side, previous);
    if (previous == 0u)
    {
      return;
    }
    InterlockedCompareExchange(EdgeTable[base + 3u], 0u, side, previous);
    if (previous == 0u)
    {
      return;
    }
    // A third triangle on this edge.
    InterlockedOr(EdgeTable[base + 4u], 1u);
    return;
  }
  sccrShellFlag(SCCR_SHELL_FLAG_WELD_OVERFLOW);
}

// Phase 3. Writes the neighbour word, or the boundary's kind (SCCR_SHELL_NEIGHBOUR_*): open, non-manifold, flipped
// winding, or a welded neighbour across an edge with a degenerate direction at an end, whose prisms would point their
// shells at each other.
void sccrShellResolveEdge(uint edgeIndex)
{
  const uint triIndex = edgeIndex / 3u;
  uint smallerId;
  uint largerId;
  uint edgeDirection;
  uint fromSlot;
  uint toSlot;
  uint neighbour = SCCR_SHELL_NEIGHBOUR_NONE;
  uint boundaryFlags = SCCR_SHELL_FLAG_HAS_BOUNDARY;
  if (sccrShellEdgeKey(triIndex, edgeIndex % 3u, smallerId, largerId, edgeDirection, fromSlot, toSlot))
  {
    const uint side = ((edgeIndex << 1u) | edgeDirection) + 1u;
    const uint start = sccrShellEdgeSlot(smallerId, largerId, cb.edgeTableSize);
    const uint probes = min(kSccrShellMaxProbes, cb.edgeTableSize);
    for (uint p = 0u; p < probes; ++p)
    {
      const uint base = ((start + p) & (cb.edgeTableSize - 1u)) * SCCR_SHELL_EDGE_SLOT_DWORDS;
      const uint storedSmaller = EdgeTable[base + 0u];
      if (storedSmaller == 0u)
      {
        break;
      }
      if (storedSmaller != smallerId + 1u || EdgeTable[base + 1u] != largerId + 1u)
      {
        continue;
      }
      if ((EdgeTable[base + 4u] & 1u) != 0u)
      {
        boundaryFlags |= SCCR_SHELL_FLAG_NON_MANIFOLD;
        neighbour = SCCR_SHELL_NEIGHBOUR_NON_MANIFOLD;
        break;
      }
      const uint firstSide = EdgeTable[base + 2u];
      const uint otherSide = firstSide == side ? EdgeTable[base + 3u] : firstSide;
      if (otherSide == 0u || otherSide == side)
      {
        // An open edge has no second triangle.
        break;
      }
      const uint otherPacked = otherSide - 1u;
      if ((otherPacked & 1u) == edgeDirection)
      {
        // Both triangles run along the edge the same way, so the winding is inconsistent.
        boundaryFlags |= SCCR_SHELL_FLAG_FLIPPED_WINDING;
        neighbour = SCCR_SHELL_NEIGHBOUR_FLIPPED;
        break;
      }
      bool fromDegenerate;
      bool toDegenerate;
      sccrShellSlotDirection(fromSlot, fromDegenerate);
      sccrShellSlotDirection(toSlot, toDegenerate);
      if (fromDegenerate || toDegenerate)
      {
        // Both sides see the same two slots, so both get the boundary.
        neighbour = SCCR_SHELL_NEIGHBOUR_DEGENERATE;
        break;
      }
      const uint otherEdge = otherPacked >> 1u;
      neighbour = (otherEdge / 3u) | ((otherEdge % 3u) << SCCR_SHELL_NEIGHBOUR_SLOT_SHIFT);
      break;
    }
  }
  if (SCCR_SHELL_NEIGHBOUR_IS_BOUNDARY(neighbour))
  {
    sccrShellFlag(boundaryFlags);
  }
  Output[SCCR_SHELL_HEADER_DWORDS + edgeIndex] = neighbour;
}

// Returns the smallest d > 0 with a d^2 + b d + c = 0, for c > 0 (positive at 0), or +inf if it never reaches 0.
// The roots are q / a and c / q with q = -(b + sign(b) sqrt(b^2 - 4ac)) / 2, which keeps the small root accurate when a is tiny.
float sccrShellSmallestPositiveRoot(float a, float b, float c)
{
  const float infinity = asfloat(0x7F800000u);
  if (a == 0.0f)
  {
    return b < 0.0f ? -c / b : infinity;
  }
  const float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f)
  {
    return infinity;
  }
  const float root = sqrt(discriminant);
  const float q = -0.5f * (b + (b >= 0.0f ? root : -root));
  float smallest = infinity;
  const float first = q / a;
  if (first > 0.0f)
  {
    smallest = first;
  }
  if (q != 0.0f)
  {
    const float second = c / q;
    if (second > 0.0f && second < smallest)
    {
      smallest = second;
    }
  }
  return smallest;
}

// Phase 5 (after phase 1): the fold guard's limits. With the corners moved d along their unit welded
// directions n_k (the inner triangle at depth d: p_k - d n_k against them, p_k + d n_k along them), the march needs
// every corner's direction on the positive side of the inner triangle's normal N(d) = cross(e1 - d g1, e2 - d g2),
// g_k = n_k - n_0: q_k(d) = dot(n_k, N(d)) = A_k - B_k d + C_k d^2 (d -> -d along them), linear in the barycentrics, so
// the corners bound the whole triangle. q_k can fall to 0 and turn negative (the triangle tips over: a corner runs past
// its neighbours, as along a convex edge) or only touch 0 (a triangle that shrinks evenly to a point, as a cube's corner
// faces or a sphere's do: N(d) scales by the square of the shrink and comes back positive, point-mirrored, past it). The
// limit is therefore the first depth at which some q_k has lost SCCR_SHELL_FOLD_MARGIN of its value at depth 0, which
// catches both and keeps the prism well clear of folding. The smallest over the position's triangles is its limit.
// This assumes equal depths at the three corners, and the margin covers corners clamped to different limits. A triangle
// already inverted or degenerate at depth 0 adds nothing, because the march fails on it whatever the depth.
void sccrShellFold(uint triIndex)
{
  float3 positions[3];
  float3 directions[3];
  uint slots[3];
  [unroll]
  for (uint k = 0u; k < 3u; ++k)
  {
    positions[k] = sccrShellLoadPosition(sccrShellLoadIndex(triIndex * 3u + k));
    slots[k] = sccrShellVertexLookup(positions[k]);
    if (slots[k] == kSccrShellInvalidSlot)
    {
      return;
    }
    bool degenerate;
    directions[k] = sccrShellSlotDirection(slots[k], degenerate);
  }
  const float3 e1 = positions[1] - positions[0];
  const float3 e2 = positions[2] - positions[0];
  const float3 normal = cross(e1, e2);
  const float normalLength = length(normal);
  if (!(normalLength > 0.0f))
  {
    return;
  }
  const float3 g1 = directions[1] - directions[0];
  const float3 g2 = directions[2] - directions[0];
  const float3 linear = cross(e1, g2) + cross(g1, e2);
  const float3 quadratic = cross(g1, g2);
  float againstRoot = asfloat(0x7F800000u);
  float alongRoot = asfloat(0x7F800000u);
  [unroll]
  for (uint k = 0u; k < 3u; ++k)
  {
    const float a = dot(directions[k], normal);
    if (!(a > kSccrShellFoldMinCosine * normalLength))
    {
      return;
    }
    const float b = dot(directions[k], linear);
    const float c = dot(directions[k], quadratic);
    // q_k(d) = (1 - margin) A_k, i.e. c d^2 -/+ b d + margin a = 0 (a > 0 here).
    againstRoot = min(againstRoot, sccrShellSmallestPositiveRoot(c, -b, SCCR_SHELL_FOLD_MARGIN * a));
    alongRoot = min(alongRoot, sccrShellSmallestPositiveRoot(c, b, SCCR_SHELL_FOLD_MARGIN * a));
  }
  const bool againstFinite = againstRoot < asfloat(0x7F800000u);
  const bool alongFinite = alongRoot < asfloat(0x7F800000u);
  if (!againstFinite && !alongFinite)
  {
    return;
  }
  sccrShellFlag(SCCR_SHELL_FLAG_FOLDS_CLAMPED);
  [unroll]
  for (uint k = 0u; k < 3u; ++k)
  {
    const uint base = slots[k] * SCCR_SHELL_VERTEX_SLOT_DWORDS;
    // Positive floats order like their bits; the complement turns the minimum into a maximum over a cleared table.
    if (againstFinite)
    {
      InterlockedMax(VertexTable[base + 5u], ~asuint(againstRoot));
    }
    if (alongFinite)
    {
      InterlockedMax(VertexTable[base + 7u], ~asuint(alongRoot));
    }
  }
}

// Signed octahedral, 16-bit snorm per axis.
uint sccrShellEncodeDirection(float3 direction)
{
  const float3 n = direction / max(abs(direction.x) + abs(direction.y) + abs(direction.z), 1e-20f);
  float2 octahedral = n.xy;
  if (n.z < 0.0f)
  {
    octahedral = (1.0f - abs(n.yx)) * float2(n.x >= 0.0f ? 1.0f : -1.0f, n.y >= 0.0f ? 1.0f : -1.0f);
  }
  const int2 quantized = int2(round(clamp(octahedral, -1.0f, 1.0f) * 32767.0f));
  return (uint(quantized.x) & 0xFFFFu) | ((uint(quantized.y) & 0xFFFFu) << 16u);
}

// Phase 4.
void sccrShellWriteVertex(uint vertexIndex)
{
  const uint outputBase = SCCR_SHELL_HEADER_DWORDS + 3u * cb.numTriangles + 2u * vertexIndex;
  const uint weldedIdOutput = SCCR_SHELL_HEADER_DWORDS + 3u * cb.numTriangles + 2u * cb.numVertices + vertexIndex;
  const uint slot = sccrShellVertexLookup(sccrShellLoadPosition(vertexIndex));
  if (slot == kSccrShellInvalidSlot)
  {
    // Every corner of every triangle was inserted in phase 1, so this is an overflow or a vertex no triangle uses.
    Output[outputBase + 0u] = 0u;
    Output[outputBase + 1u] = SCCR_SHELL_VERTEX_FLAG_UNWELDED;
    Output[weldedIdOutput] = vertexIndex;
    return;
  }
  const uint base = slot * SCCR_SHELL_VERTEX_SLOT_DWORDS;
  const uint weldedId = ~VertexTable[base + 1u];
  Output[weldedIdOutput] = weldedId;
  uint flags = (weldedId & 1u) != 0u ? SCCR_SHELL_VERTEX_FLAG_WELD_PARITY : 0u;
  flags |= weldedId == vertexIndex ? SCCR_SHELL_VERTEX_FLAG_CANONICAL : 0u;

  // A fin or a two-sided sheet falls back to a face normal (sccrShellSlotDirection); phase 3 made the edges at such
  // positions boundaries.
  bool degenerate;
  const float3 direction = sccrShellSlotDirection(slot, degenerate);
  if (degenerate)
  {
    flags |= SCCR_SHELL_VERTEX_FLAG_DEGENERATE;
    sccrShellFlag(SCCR_SHELL_FLAG_DEGENERATE_DIRECTIONS);
  }

  // The fold guard's limits from phase 5, in object units. They are SCCR_SHELL_VERTEX_NO_LIMIT where no triangle at the
  // position folds, clamped before the conversion, which leaves larger values undefined.
  const uint againstBits = VertexTable[base + 5u];
  const uint alongBits = VertexTable[base + 7u];
  const float againstLimit =
    againstBits != 0u ? min(asfloat(~againstBits), SCCR_SHELL_VERTEX_NO_LIMIT) : SCCR_SHELL_VERTEX_NO_LIMIT;
  const float alongLimit =
    alongBits != 0u ? min(asfloat(~alongBits), SCCR_SHELL_VERTEX_NO_LIMIT) : SCCR_SHELL_VERTEX_NO_LIMIT;
  // The second limit keeps a half float's bits 3-14 (the sign is 0). Dropping the low three bits rounds it down.
  const uint alongPacked = (f32tof16(alongLimit) >> 3u) & SCCR_SHELL_VERTEX_OUTWARD_LIMIT_MASK;

  Output[outputBase + 0u] = sccrShellEncodeDirection(direction);
  Output[outputBase + 1u] = (f32tof16(againstLimit) & SCCR_SHELL_VERTEX_LIMIT_MASK) | flags |
                            (alongPacked << SCCR_SHELL_VERTEX_OUTWARD_LIMIT_SHIFT);
}

[shader("compute")]
[numthreads(SCCR_SHELL_GROUP_SIZE, 1, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  const uint threadIndex = threadId.y * cb.rowThreads + threadId.x;
  if (threadId.x >= cb.rowThreads || threadIndex >= cb.threadCount)
  {
    return;
  }

  if (cb.phase == SCCR_SHELL_PHASE_WELD)
  {
    sccrShellWeld(threadIndex);
  }
  else if (cb.phase == SCCR_SHELL_PHASE_EDGES)
  {
    sccrShellInsertEdge(threadIndex);
  }
  else if (cb.phase == SCCR_SHELL_PHASE_RESOLVE)
  {
    sccrShellResolveEdge(threadIndex);
  }
  else if (cb.phase == SCCR_SHELL_PHASE_VERTICES)
  {
    sccrShellWriteVertex(threadIndex);
  }
  else if (cb.phase == SCCR_SHELL_PHASE_FOLD)
  {
    sccrShellFold(threadIndex);
  }
}
