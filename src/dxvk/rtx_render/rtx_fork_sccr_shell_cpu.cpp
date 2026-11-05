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
#include "rtx_fork_sccr_shell_cpu.h"

#include <algorithm>

namespace dxvk::sccr_shell_cpu {

namespace {

uint32_t tableSize(uint64_t value) {
  uint64_t size = 256;
  while (size < value && size < (uint64_t(1) << 31)) {
    size <<= 1;
  }
  return uint32_t(size);
}

// The five phases over one mesh, as SccrShellManager::build dispatches them. weldCell: SccrShellBuildArgs::weldCell.
class Builder {
public:
  explicit Builder(const Mesh& mesh, float weldCell = 0.0f)
    : m_mesh(mesh)
    , m_weldCell(weldCell)
    , m_numTriangles(mesh.triangleCount())
    , m_numVertices(uint32_t(mesh.positions.size()))
    , m_vertexTableSize(tableSize(2 * uint64_t(m_numVertices)))
    , m_edgeTableSize(tableSize(3 * uint64_t(m_numTriangles)))
    , m_vertexTable(size_t(m_vertexTableSize) * kVertexSlotDwords, 0u)
    , m_edgeTable(size_t(m_edgeTableSize) * kEdgeSlotDwords, 0u)
    , m_output(kHeaderDwords + 3 * size_t(m_numTriangles) + 3 * size_t(m_numVertices), 0u) {
    m_output[0] = m_numTriangles;
    m_output[1] = m_numVertices;
    m_output[kHeaderFlags] = kVersion | ((weldCell > 0.0f ? kWeldGrid : 0u) << kWeldModeShift);
    m_output[3] = asUint(1.0f);
  }

  const std::vector<uint32_t>& run() {
    for (uint32_t t = 0; t < m_numTriangles; ++t) { weld(t); }
    for (uint32_t t = 0; t < m_numTriangles; ++t) { fold(t); }
    for (uint32_t e = 0; e < 3 * m_numTriangles; ++e) { insertEdge(e); }
    for (uint32_t e = 0; e < 3 * m_numTriangles; ++e) { resolveEdge(e); }
    for (uint32_t v = 0; v < m_numVertices; ++v) { writeVertex(v); }
    return m_output;
  }

private:
  uint32_t loadIndex(uint32_t indexIndex) const {
    return m_mesh.indices.empty() ? indexIndex : m_mesh.indices[indexIndex];
  }

  Float3 loadPosition(uint32_t vertexIndex) const {
    return m_mesh.positions[vertexIndex];
  }

  static uint32_t floatBits(float value) {
    return value == 0.0f ? 0u : asUint(value);
  }

  struct Key {
    uint32_t x, y, z;
  };

  static uint32_t keyWord(uint32_t bits) {
    return bits == 0xFFFFFFFFu ? bits : bits + 1u;
  }

  static Key positionKey(Float3 p) {
    return { keyWord(floatBits(p.x)), keyWord(floatBits(p.y)), keyWord(floatBits(p.z)) };
  }

  static uint32_t positionSlot(Key key, uint32_t size) {
    uint32_t h = key.x * 0x9E3779B1u;
    h ^= key.y + 0x7F4A7C15u + (h << 6u) + (h >> 2u);
    h ^= key.z + 0x7F4A7C15u + (h << 6u) + (h >> 2u);
    h ^= h >> 16u;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13u;
    h *= 0xC2B2AE35u;
    h ^= h >> 16u;
    return h & (size - 1u);
  }

  static uint32_t edgeSlot(uint32_t smallerId, uint32_t largerId, uint32_t size) {
    uint32_t h = smallerId * 0x9E3779B1u;
    h ^= largerId + 0x7F4A7C15u + (h << 6u) + (h >> 2u);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 13u;
    return h & (size - 1u);
  }

  // InterlockedCompareExchange, one thread at a time.
  static uint32_t compareExchange(uint32_t& word, uint32_t compare, uint32_t value) {
    const uint32_t previous = word;
    if (previous == compare) {
      word = value;
    }
    return previous;
  }

  Float3 weldKey(Float3 p) const {
    if (m_weldCell > 0.0f) {
      return { std::round(p.x / m_weldCell) * m_weldCell, std::round(p.y / m_weldCell) * m_weldCell,
               std::round(p.z / m_weldCell) * m_weldCell };
    }
    return p;
  }

  uint32_t vertexInsert(Float3 position, uint32_t claimingTriangle) {
    const Key key = positionKey(weldKey(position));
    const uint32_t start = positionSlot(key, m_vertexTableSize);
    const uint32_t probes = std::min(kMaxProbes, m_vertexTableSize);
    for (uint32_t i = 0; i < probes; ++i) {
      const uint32_t slot = (start + i) & (m_vertexTableSize - 1u);
      const size_t base = size_t(slot) * kVertexSlotDwords;
      uint32_t previous = compareExchange(m_vertexTable[base + 0], 0u, key.x);
      if (previous != 0u && previous != key.x) {
        continue;
      }
      previous = compareExchange(m_vertexTable[base + 8], 0u, key.y);
      if (previous != 0u && previous != key.y) {
        continue;
      }
      previous = compareExchange(m_vertexTable[base + 9], 0u, key.z);
      if (previous == 0u) {
        m_vertexTable[base + 6] = claimingTriangle + 1u;
        return slot;
      }
      if (previous == key.z) {
        return slot;
      }
    }
    return kInvalidSlot;
  }

  uint32_t vertexLookup(Float3 position) const {
    const Key key = positionKey(weldKey(position));
    const uint32_t start = positionSlot(key, m_vertexTableSize);
    const uint32_t probes = std::min(kMaxProbes, m_vertexTableSize);
    for (uint32_t i = 0; i < probes; ++i) {
      const uint32_t slot = (start + i) & (m_vertexTableSize - 1u);
      const size_t base = size_t(slot) * kVertexSlotDwords;
      const uint32_t stored = m_vertexTable[base + 0];
      if (stored == 0u) {
        break;
      }
      if (stored == key.x && m_vertexTable[base + 8] == key.y && m_vertexTable[base + 9] == key.z) {
        return slot;
      }
    }
    return kInvalidSlot;
  }

  bool weldedId(Float3 p, uint32_t& id, uint32_t& slot) const {
    slot = vertexLookup(p);
    id = 0;
    if (slot == kInvalidSlot) {
      return false;
    }
    id = ~m_vertexTable[size_t(slot) * kVertexSlotDwords + 1];
    return true;
  }

  Float3 slotDirection(uint32_t slot, bool& degenerate) const {
    const size_t base = size_t(slot) * kVertexSlotDwords;
    const Float3 sum { float(int32_t(m_vertexTable[base + 2])), float(int32_t(m_vertexTable[base + 3])),
                       float(int32_t(m_vertexTable[base + 4])) };
    const float sumLength = length(sum);
    degenerate = !(sumLength > kDegenerateSum * kFixedPointScale);
    if (!degenerate) {
      return { sum.x / sumLength, sum.y / sumLength, sum.z / sumLength };
    }
    Float3 direction { 0, 0, 1 };
    const uint32_t claiming = m_vertexTable[base + 6];
    if (claiming != 0u) {
      const uint32_t tri = claiming - 1u;
      const Float3 p0 = loadPosition(loadIndex(tri * 3 + 0));
      const Float3 p1 = loadPosition(loadIndex(tri * 3 + 1));
      const Float3 p2 = loadPosition(loadIndex(tri * 3 + 2));
      const Float3 c = cross(p1 - p0, p2 - p0);
      const float l = length(c);
      if (l > 0.0f) {
        direction = { c.x / l, c.y / l, c.z / l };
      }
    }
    return direction;
  }

  static float smallestPositiveRoot(float a, float b, float c) {
    if (a == 0.0f) {
      return b < 0.0f ? -c / b : kInfinity;
    }
    const float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) {
      return kInfinity;
    }
    const float root = std::sqrt(discriminant);
    const float q = -0.5f * (b + (b >= 0.0f ? root : -root));
    float smallest = kInfinity;
    const float first = q / a;
    if (first > 0.0f) {
      smallest = first;
    }
    if (q != 0.0f) {
      const float second = c / q;
      if (second > 0.0f && second < smallest) {
        smallest = second;
      }
    }
    return smallest;
  }

  void fold(uint32_t tri) {
    Float3 p[3];
    Float3 n[3];
    uint32_t slots[3];
    for (uint32_t k = 0; k < 3; ++k) {
      p[k] = loadPosition(loadIndex(tri * 3 + k));
      slots[k] = vertexLookup(p[k]);
      if (slots[k] == kInvalidSlot) {
        return;
      }
      bool degenerate;
      n[k] = slotDirection(slots[k], degenerate);
    }
    const Float3 e1 = p[1] - p[0];
    const Float3 e2 = p[2] - p[0];
    const Float3 normal = cross(e1, e2);
    const float normalLength = length(normal);
    if (!(normalLength > 0.0f)) {
      return;
    }
    const Float3 g1 = n[1] - n[0];
    const Float3 g2 = n[2] - n[0];
    const Float3 linear = cross(e1, g2) + cross(g1, e2);
    const Float3 quadratic = cross(g1, g2);
    float againstRoot = kInfinity;
    float alongRoot = kInfinity;
    for (uint32_t k = 0; k < 3; ++k) {
      const float a = dot(n[k], normal);
      if (!(a > kFoldMinCosine * normalLength)) {
        return;
      }
      const float b = dot(n[k], linear);
      const float c = dot(n[k], quadratic);
      againstRoot = std::min(againstRoot, smallestPositiveRoot(c, -b, kFoldMargin * a));
      alongRoot = std::min(alongRoot, smallestPositiveRoot(c, b, kFoldMargin * a));
    }
    const bool againstFinite = againstRoot < kInfinity;
    const bool alongFinite = alongRoot < kInfinity;
    if (!againstFinite && !alongFinite) {
      return;
    }
    m_output[kHeaderFlags] |= kFlagFoldsClamped;
    for (uint32_t k = 0; k < 3; ++k) {
      const size_t base = size_t(slots[k]) * kVertexSlotDwords;
      if (againstFinite) {
        m_vertexTable[base + 5] = std::max(m_vertexTable[base + 5], ~asUint(againstRoot));
      }
      if (alongFinite) {
        m_vertexTable[base + 7] = std::max(m_vertexTable[base + 7], ~asUint(alongRoot));
      }
    }
  }

  static float cornerAngle(Float3 a, Float3 b) {
    const float lengths = std::sqrt(dot(a, a) * dot(b, b));
    if (!(lengths > 0.0f)) {
      return 0.0f;
    }
    return std::acos(std::clamp(dot(a, b) / lengths, -1.0f, 1.0f));
  }

  void weld(uint32_t tri) {
    uint32_t vertexIndices[3];
    Float3 p[3];
    for (uint32_t k = 0; k < 3; ++k) {
      vertexIndices[k] = loadIndex(tri * 3 + k);
      p[k] = loadPosition(vertexIndices[k]);
    }
    const Float3 faceCross = cross(p[1] - p[0], p[2] - p[0]);
    const float faceLength = length(faceCross);
    const Float3 faceNormal = faceLength > 0.0f ?
      Float3 { faceCross.x / faceLength, faceCross.y / faceLength, faceCross.z / faceLength } : Float3 { 0, 0, 0 };
    for (uint32_t k = 0; k < 3; ++k) {
      const uint32_t slot = vertexInsert(p[k], tri);
      if (slot == kInvalidSlot) {
        m_output[kHeaderFlags] |= kFlagWeldOverflow;
        continue;
      }
      const size_t base = size_t(slot) * kVertexSlotDwords;
      m_vertexTable[base + 1] = std::max(m_vertexTable[base + 1], ~vertexIndices[k]);
      const Float3 toNext = p[(k + 1) % 3] - p[k];
      const Float3 toPrevious = p[(k + 2) % 3] - p[k];
      if (faceLength > 0.0f) {
        const float weight = cornerAngle(toNext, toPrevious) / kPi * kFixedPointScale;
        m_vertexTable[base + 2] += uint32_t(int32_t(faceNormal.x * weight));
        m_vertexTable[base + 3] += uint32_t(int32_t(faceNormal.y * weight));
        m_vertexTable[base + 4] += uint32_t(int32_t(faceNormal.z * weight));
      }
    }
  }

  bool edgeKey(uint32_t tri, uint32_t slot, uint32_t& smallerId, uint32_t& largerId, uint32_t& direction,
               uint32_t& fromSlot, uint32_t& toSlot) const {
    const uint32_t fromIndex = loadIndex(tri * 3 + (slot + 1) % 3);
    const uint32_t toIndex = loadIndex(tri * 3 + (slot + 2) % 3);
    uint32_t fromId;
    uint32_t toId;
    smallerId = largerId = direction = 0;
    if (!weldedId(loadPosition(fromIndex), fromId, fromSlot) || !weldedId(loadPosition(toIndex), toId, toSlot) ||
        fromId == toId) {
      return false;
    }
    smallerId = std::min(fromId, toId);
    largerId = std::max(fromId, toId);
    direction = fromId < toId ? 0u : 1u;
    return true;
  }

  void insertEdge(uint32_t edge) {
    uint32_t smallerId, largerId, direction, fromSlot, toSlot;
    if (!edgeKey(edge / 3, edge % 3, smallerId, largerId, direction, fromSlot, toSlot)) {
      return;
    }
    const uint32_t side = ((edge << 1u) | direction) + 1u;
    const uint32_t start = edgeSlot(smallerId, largerId, m_edgeTableSize);
    const uint32_t probes = std::min(kMaxProbes, m_edgeTableSize);
    for (uint32_t i = 0; i < probes; ++i) {
      const size_t base = size_t((start + i) & (m_edgeTableSize - 1u)) * kEdgeSlotDwords;
      uint32_t previous = compareExchange(m_edgeTable[base + 0], 0u, smallerId + 1u);
      if (previous != 0u && previous != smallerId + 1u) {
        continue;
      }
      previous = compareExchange(m_edgeTable[base + 1], 0u, largerId + 1u);
      if (previous != 0u && previous != largerId + 1u) {
        continue;
      }
      if (compareExchange(m_edgeTable[base + 2], 0u, side) == 0u) {
        return;
      }
      if (compareExchange(m_edgeTable[base + 3], 0u, side) == 0u) {
        return;
      }
      m_edgeTable[base + 4] |= 1u;
      return;
    }
    m_output[kHeaderFlags] |= kFlagWeldOverflow;
  }

  void resolveEdge(uint32_t edge) {
    uint32_t smallerId, largerId, direction, fromSlot, toSlot;
    uint32_t neighbour = kNeighbourNone;
    uint32_t boundaryFlags = kFlagHasBoundary;
    if (edgeKey(edge / 3, edge % 3, smallerId, largerId, direction, fromSlot, toSlot)) {
      const uint32_t side = ((edge << 1u) | direction) + 1u;
      const uint32_t start = edgeSlot(smallerId, largerId, m_edgeTableSize);
      const uint32_t probes = std::min(kMaxProbes, m_edgeTableSize);
      for (uint32_t i = 0; i < probes; ++i) {
        const size_t base = size_t((start + i) & (m_edgeTableSize - 1u)) * kEdgeSlotDwords;
        const uint32_t storedSmaller = m_edgeTable[base + 0];
        if (storedSmaller == 0u) {
          break;
        }
        if (storedSmaller != smallerId + 1u || m_edgeTable[base + 1] != largerId + 1u) {
          continue;
        }
        if ((m_edgeTable[base + 4] & 1u) != 0u) {
          boundaryFlags |= kFlagNonManifold;
          neighbour = kNeighbourNonManifold;
          break;
        }
        const uint32_t firstSide = m_edgeTable[base + 2];
        const uint32_t otherSide = firstSide == side ? m_edgeTable[base + 3] : firstSide;
        if (otherSide == 0u || otherSide == side) {
          break;
        }
        const uint32_t otherPacked = otherSide - 1u;
        if ((otherPacked & 1u) == direction) {
          boundaryFlags |= kFlagFlippedWinding;
          neighbour = kNeighbourFlipped;
          break;
        }
        bool fromDegenerate;
        bool toDegenerate;
        slotDirection(fromSlot, fromDegenerate);
        slotDirection(toSlot, toDegenerate);
        if (fromDegenerate || toDegenerate) {
          neighbour = kNeighbourDegenerate;
          break;
        }
        const uint32_t otherEdge = otherPacked >> 1u;
        neighbour = (otherEdge / 3u) | ((otherEdge % 3u) << kNeighbourSlotShift);
        break;
      }
    }
    if (isBoundary(neighbour)) {
      m_output[kHeaderFlags] |= boundaryFlags;
    }
    m_output[kHeaderDwords + edge] = neighbour;
  }

  static uint32_t encodeDirection(Float3 d) {
    const float l1 = std::max(std::fabs(d.x) + std::fabs(d.y) + std::fabs(d.z), 1e-20f);
    const Float3 n { d.x / l1, d.y / l1, d.z / l1 };
    float ox = n.x;
    float oy = n.y;
    if (n.z < 0.0f) {
      ox = (1.0f - std::fabs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
      oy = (1.0f - std::fabs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
    }
    const int32_t qx = int32_t(std::round(std::clamp(ox, -1.0f, 1.0f) * 32767.0f));
    const int32_t qy = int32_t(std::round(std::clamp(oy, -1.0f, 1.0f) * 32767.0f));
    return (uint32_t(qx) & 0xFFFFu) | ((uint32_t(qy) & 0xFFFFu) << 16u);
  }

  void writeVertex(uint32_t vertex) {
    const size_t outputBase = kHeaderDwords + 3 * size_t(m_numTriangles) + 2 * size_t(vertex);
    const size_t weldedIdOutput = kHeaderDwords + 3 * size_t(m_numTriangles) + 2 * size_t(m_numVertices) + vertex;
    const uint32_t slot = vertexLookup(loadPosition(vertex));
    if (slot == kInvalidSlot) {
      m_output[outputBase] = 0u;
      m_output[outputBase + 1] = kVertexFlagUnwelded;
      m_output[weldedIdOutput] = vertex;
      m_depthLimits.resize(m_numVertices, { 0.0f, 0.0f });
      return;
    }
    const size_t base = size_t(slot) * kVertexSlotDwords;
    const uint32_t id = ~m_vertexTable[base + 1];
    m_output[weldedIdOutput] = id;
    uint32_t flags = (id & 1u) != 0u ? (1u << 16) : 0u;
    flags |= id == vertex ? kVertexFlagCanonical : 0u;
    bool degenerate;
    const Float3 direction = slotDirection(slot, degenerate);
    if (degenerate) {
      flags |= kVertexFlagDegenerate;
      m_output[kHeaderFlags] |= kFlagDegenerateDirections;
    }
    const uint32_t againstBits = m_vertexTable[base + 5];
    const uint32_t alongBits = m_vertexTable[base + 7];
    const float againstLimit = againstBits != 0u ? std::min(asFloat(~againstBits), kNoLimit) : kNoLimit;
    const float alongLimit = alongBits != 0u ? std::min(asFloat(~alongBits), kNoLimit) : kNoLimit;
    m_output[outputBase] = encodeDirection(direction);
    m_depthLimits.resize(m_numVertices, { 0.0f, 0.0f });
    m_depthLimits[vertex] = { againstLimit, alongLimit };
    // As the shader: the limit against the direction as a half float in the low bits, the one along it as a half
    // float's bits 3-14 (rounded down) in bits 20-31.
    const uint32_t alongPacked = (uint32_t(f32tof16(alongLimit)) >> 3u) & kVertexOutwardLimitMask;
    m_output[outputBase + 1] = (uint32_t(f32tof16(againstLimit)) & kVertexLimitMask) | flags |
                               (alongPacked << kVertexOutwardLimitShift);
  }

  const Mesh& m_mesh;
  const float m_weldCell;
  const uint32_t m_numTriangles;
  const uint32_t m_numVertices;
  const uint32_t m_vertexTableSize;
  const uint32_t m_edgeTableSize;
  std::vector<uint32_t> m_vertexTable;
  std::vector<uint32_t> m_edgeTable;
  std::vector<uint32_t> m_output;

public:
  // Per vertex: the fold limits against and along the welded direction (before the half-float conversion).
  std::vector<std::pair<float, float>> m_depthLimits;
};

} // namespace

Float3 decodeDirection(uint32_t packed) {
  const int32_t qx = int32_t(packed << 16) >> 16;
  const int32_t qy = int32_t(packed) >> 16;
  float x = std::max(float(qx) / 32767.0f, -1.0f);
  float y = std::max(float(qy) / 32767.0f, -1.0f);
  float z = 1.0f - std::fabs(x) - std::fabs(y);
  const float fold = std::max(-z, 0.0f);
  x += x >= 0.0f ? -fold : fold;
  y += y >= 0.0f ? -fold : fold;
  const float l = std::sqrt(x * x + y * y + z * z);
  return { x / l, y / l, z / l };
}

uint16_t f32tof16(float value) {
  const uint32_t bits = asUint(value);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t magnitude = bits & 0x7FFFFFFFu;
  if (magnitude >= 0x7F800000u) {
    return uint16_t(sign | 0x7C00u | (magnitude > 0x7F800000u ? 0x200u : 0u));
  }
  if (magnitude >= 0x477FF000u) {  // rounds to above the largest half float
    return uint16_t(sign | 0x7C00u);
  }
  if (magnitude < 0x38800000u) {  // subnormal half (or zero)
    if (magnitude < 0x33000000u) {
      return uint16_t(sign);
    }
    const uint32_t exponent = magnitude >> 23;
    const uint32_t mantissa = (magnitude & 0x7FFFFFu) | 0x800000u;
    const uint32_t shift = 126u - exponent;  // 14 + (113 - exponent - 1) + 1
    const uint32_t half = mantissa >> shift;
    const uint32_t rest = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    const uint32_t rounded = half + ((rest > halfway || (rest == halfway && (half & 1u) != 0u)) ? 1u : 0u);
    return uint16_t(sign | rounded);
  }
  const uint32_t rebased = magnitude - 0x38000000u;  // exponent 127 -> 15
  const uint32_t half = rebased >> 13;
  const uint32_t rest = rebased & 0x1FFFu;
  const uint32_t rounded = half + ((rest > 0x1000u || (rest == 0x1000u && (half & 1u) != 0u)) ? 1u : 0u);
  return uint16_t(sign | rounded);
}

float f16tof32(uint16_t bits) {
  const uint32_t sign = uint32_t(bits & 0x8000u) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1Fu;
  uint32_t mantissa = bits & 0x3FFu;
  if (exponent == 0x1Fu) {
    return asFloat(sign | 0x7F800000u | (mantissa << 13));
  }
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return asFloat(sign);
    }
    // Subnormal: normalise.
    uint32_t e = 113u;
    while ((mantissa & 0x400u) == 0u) {
      mantissa <<= 1;
      --e;
    }
    return asFloat(sign | (e << 23) | ((mantissa & 0x3FFu) << 13));
  }
  return asFloat(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

Result build(const Mesh& mesh, float weldCell) {
  Builder builder(mesh, weldCell);
  Result result;
  result.data = builder.run();
  result.depthLimits = builder.m_depthLimits;
  result.triangles = mesh.triangleCount();
  result.vertices = uint32_t(mesh.positions.size());
  return result;
}

} // namespace dxvk::sccr_shell_cpu
