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
#pragma once

// A CPU copy of the GPU shell build (sccr_shell_adjacency.comp.slang, SccrShellManager::build; the GPU's
// atomics run one thread at a time here). The runtime always uses the
// GPU's shell words. This copy serves the unit test (test_sccr_shell). Standard library only. Keep it in step with
// the shader and sccr_shell_args.h (the constants below mirror it).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace dxvk::sccr_shell_cpu {

// sccr_shell_args.h
constexpr uint32_t kHeaderDwords = 4;
constexpr uint32_t kHeaderFlags = 2;
constexpr uint32_t kVersion = 3;
constexpr uint32_t kWeldModeShift = 8;
constexpr uint32_t kWeldGrid = 1;
constexpr uint32_t kFlagHasBoundary = 1u << 16;
constexpr uint32_t kFlagFoldsClamped = 1u << 17;
constexpr uint32_t kFlagWeldOverflow = 1u << 18;
constexpr uint32_t kFlagDegenerateDirections = 1u << 19;
constexpr uint32_t kFlagNonManifold = 1u << 20;
constexpr uint32_t kFlagFlippedWinding = 1u << 21;
constexpr uint32_t kNeighbourNone = 0xFFFFFFFFu;
constexpr uint32_t kNeighbourNonManifold = 0x3FFFFFFFu;
constexpr uint32_t kNeighbourFlipped = 0x7FFFFFFFu;
constexpr uint32_t kNeighbourDegenerate = 0xBFFFFFFFu;
constexpr uint32_t kNeighbourTriangleMask = 0x3FFFFFFFu;
constexpr uint32_t kNeighbourSlotShift = 30;
constexpr uint32_t kVertexFlagDegenerate = 1u << 17;
constexpr uint32_t kVertexFlagUnwelded = 1u << 18;
constexpr uint32_t kVertexFlagCanonical = 1u << 19;
constexpr float kNoLimit = 65504.0f;
constexpr uint32_t kVertexLimitMask = 0xFFFFu;
constexpr uint32_t kVertexOutwardLimitShift = 20u;
constexpr uint32_t kVertexOutwardLimitMask = 0xFFFu;
constexpr float kFoldMargin = 0.75f;
constexpr uint32_t kVertexSlotDwords = 10;
constexpr uint32_t kEdgeSlotDwords = 5;

inline bool isBoundary(uint32_t word) { return (word & kNeighbourTriangleMask) == kNeighbourTriangleMask; }

// sccr_shell_adjacency.comp.slang
constexpr uint32_t kMaxProbes = 128;
constexpr uint32_t kInvalidSlot = 0xFFFFFFFFu;
constexpr float kFixedPointScale = 10000.0f;
constexpr float kDegenerateSum = 0.05f;
constexpr float kPi = 3.14159265358979f;
constexpr float kFoldMinCosine = 1e-3f;
constexpr float kInfinity = std::numeric_limits<float>::infinity();

struct Float3 {
  float x, y, z;
};

inline Float3 operator-(Float3 a, Float3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Float3 operator+(Float3 a, Float3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Float3 operator*(Float3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
inline float dot(Float3 a, Float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Float3 cross(Float3 a, Float3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
inline float length(Float3 a) { return std::sqrt(dot(a, a)); }

inline uint32_t asUint(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float asFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct Mesh {
  std::vector<Float3> positions;
  // Empty: non-indexed (triangle t is vertices 3t..3t+2).
  std::vector<uint32_t> indices;

  uint32_t triangleCount() const {
    return uint32_t((indices.empty() ? positions.size() : indices.size()) / 3);
  }
};

struct Result {
  std::vector<uint32_t> data;
  std::vector<std::pair<float, float>> depthLimits;
  uint32_t triangles = 0;
  uint32_t vertices = 0;

  uint32_t flags() const { return data[kHeaderFlags]; }
  uint32_t neighbour(uint32_t t, uint32_t e) const { return data[kHeaderDwords + 3 * t + e]; }
  uint32_t vertexWord(uint32_t v, uint32_t w) const { return data[kHeaderDwords + 3 * size_t(triangles) + 2 * size_t(v) + w]; }

  uint32_t boundaryEdges() const {
    uint32_t count = 0;
    for (uint32_t t = 0; t < triangles; ++t) {
      for (uint32_t e = 0; e < 3; ++e) {
        count += isBoundary(neighbour(t, e)) ? 1u : 0u;
      }
    }
    return count;
  }
};

// The five phases over one mesh, as SccrShellManager::build dispatches them. weldCell: SccrShellBuildArgs::weldCell.
// The vertex words carry the fold limits as the GPU writes them (half floats); depthLimits keeps them as floats.
Result build(const Mesh& mesh, float weldCell = 0.0f);

// The welded direction of vertex word 0 (octahedral snorm16x2).
Float3 decodeDirection(uint32_t packed);

// IEEE half conversions (round to nearest even, as f32tof16 on the GPU; f16tof32).
uint16_t f32tof16(float value);
float f16tof32(uint16_t bits);

} // namespace dxvk::sccr_shell_cpu
