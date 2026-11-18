// SPDX-License-Identifier: MIT
// Local runtime window lighting. See CREDITS.md.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace dxvk {
  struct WindowByteSpan {
    const uint8_t* bytes = nullptr;
    size_t byteCount = 0;
    WindowByteSpan() = default;
    WindowByteSpan(const uint8_t* data, size_t size) : bytes(data), byteCount(size) { }
    const uint8_t* data() const { return bytes; }
    size_t size() const { return byteCount; }
    bool empty() const { return bytes == nullptr || byteCount == 0; }
    WindowByteSpan first(size_t count) const { return {bytes, std::min(count, byteCount)}; }
  };

  enum class WindowPrimitiveTopology : uint32_t { TriangleList, TriangleStrip, TriangleFan, Unsupported };

  // Raw spans make buffer bounds and index widths explicit, including malformed imported meshes.
  struct WindowTopologyInput {
    WindowByteSpan positions;
    WindowByteSpan texcoords;
    WindowByteSpan indices;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t positionStride = 0;
    uint32_t texcoordStride = 0;
    uint32_t indexStride = 0;
    uint32_t indexWidth = 0; // 0 = non-indexed, 2/4 = uint16/uint32.
    uint32_t triangleCount = 0;
    WindowPrimitiveTopology topology = WindowPrimitiveTopology::TriangleList;
  };

  enum class WindowTopologyStatus : uint32_t {
    Complete = 0,
    MissingPositions = 1,
    MissingTexcoords = 2,
    MissingIndices = 3,
    UnsupportedLayout = 4,
    PartialInvalid = 5,
    CandidateLimit = 6,
    NumericalRange = 7,
  };

  struct WindowTopologyDiagnostics {
    WindowTopologyStatus status = WindowTopologyStatus::Complete;
    uint32_t trianglesValid = 0;
    uint32_t trianglesInvalid = 0;
    uint32_t degenerateTriangles = 0;
    uint32_t matchedEdges = 0;
    uint32_t mergedTriangles = 0;
    uint32_t ambiguousEdges = 0;
    uint32_t rejectedFacePairs = 0;
    uint64_t candidateComparisons = 0;
    double positionTolerance = 0;
    double uvTolerance = 1e-5;
  };

  // Positions contain float3 and texcoords contain float2 at their respective strides.
  // The caller validates formats and keeps each geometry/UV-transform domain separate.
  inline std::vector<uint32_t> buildWindowIslandRoots(const WindowTopologyInput& input,
      WindowTopologyDiagnostics* diagnostics = nullptr) {
    WindowTopologyDiagnostics localDiagnostics;
    auto& stats = diagnostics ? *diagnostics : localDiagnostics;
    stats = {};
    std::vector<uint32_t> parent(input.triangleCount);
    std::iota(parent.begin(), parent.end(), 0u);
    if ((input.topology != WindowPrimitiveTopology::TriangleList && input.topology != WindowPrimitiveTopology::TriangleStrip &&
         input.topology != WindowPrimitiveTopology::TriangleFan) ||
        (input.indexWidth != 0 && input.indexWidth != 2 && input.indexWidth != 4)) {
      stats.status = WindowTopologyStatus::UnsupportedLayout;
      return parent;
    }
    if (input.positions.empty() || input.texcoords.empty()) {
      stats.status = input.positions.empty() ? WindowTopologyStatus::MissingPositions : WindowTopologyStatus::MissingTexcoords;
      return parent;
    }
    if (input.indexWidth && input.indices.empty()) {
      stats.status = WindowTopologyStatus::MissingIndices;
      return parent;
    }
    if (input.positionStride < 12 || input.texcoordStride < 8 ||
        (input.indexWidth != 0 && input.indexStride < input.indexWidth)) {
      stats.status = WindowTopologyStatus::UnsupportedLayout;
      return parent;
    }
    if (!input.triangleCount) {
      return parent;
    }
    using Point = std::array<double, 3>;
    struct Vertex { Point position; std::array<double, 2> uv; };
    struct Triangle {
      std::array<Vertex, 3> vertices;
      Point normal;
      std::array<double, 3> lengths;
      bool valid = false;
    };
    auto subtract = [](const Point& a, const Point& b) -> Point {
      return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    auto dot = [](const Point& a, const Point& b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    auto cross = [](const Point& a, const Point& b) -> Point {
      return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    };
    std::vector<Triangle> triangles(input.triangleCount);
    std::vector<double> edgeLengths;
    edgeLengths.reserve(size_t(input.triangleCount) * 3);
    Point origin {};
    for (uint32_t triangle = 0; triangle < input.triangleCount; ++triangle) {
      auto& parsed = triangles[triangle];
      bool valid = true;
      for (uint32_t corner = 0; corner < 3; ++corner) {
        uint64_t index = uint64_t(triangle) * 3 + corner;
        if (input.topology == WindowPrimitiveTopology::TriangleStrip) {
          index = uint64_t(triangle) + (corner == 0 ? 0 : corner == 1 ? 1 + (triangle & 1) : 2 - (triangle & 1));
        } else if (input.topology == WindowPrimitiveTopology::TriangleFan) {
          index = corner == 0 ? 0 : uint64_t(triangle) + corner;
        }
        uint64_t vertex = index;
        if (input.indexWidth) {
          const uint64_t offset = index * input.indexStride;
          if (index >= input.indexCount || offset + input.indexWidth > input.indices.size()) {
            valid = false;
            break;
          }
          uint32_t decoded = 0;
          std::memcpy(&decoded, input.indices.data() + offset, input.indexWidth);
          vertex = decoded;
        }
        const uint64_t offset = uint64_t(vertex) * input.positionStride;
        const uint64_t uvOffset = uint64_t(vertex) * input.texcoordStride;
        if (vertex >= input.vertexCount || offset + 12 > input.positions.size() || uvOffset + 8 > input.texcoords.size()) {
          valid = false;
          break;
        }
        float position[3], uv[2];
        std::memcpy(position, input.positions.data() + offset, sizeof(position));
        std::memcpy(uv, input.texcoords.data() + uvOffset, sizeof(uv));
        for (uint32_t axis = 0; axis < 3; ++axis) {
          valid &= std::isfinite(position[axis]);
          parsed.vertices[corner].position[axis] = position[axis];
        }
        for (uint32_t axis = 0; axis < 2; ++axis) {
          valid &= std::isfinite(uv[axis]);
          parsed.vertices[corner].uv[axis] = uv[axis];
        }
      }
      if (!valid) {
        ++stats.trianglesInvalid;
        continue;
      }
      for (uint32_t corner = 0; corner < 3; ++corner) {
        const auto edge = subtract(parsed.vertices[(corner + 1) % 3].position, parsed.vertices[corner].position);
        parsed.lengths[corner] = std::sqrt(dot(edge, edge));
      }
      const auto edgeA = subtract(parsed.vertices[1].position, parsed.vertices[0].position);
      const auto edgeB = subtract(parsed.vertices[2].position, parsed.vertices[0].position);
      parsed.normal = cross(edgeA, edgeB);
      const double area = std::sqrt(dot(parsed.normal, parsed.normal));
      const double shortest = *std::min_element(parsed.lengths.begin(), parsed.lengths.end());
      const double longest = *std::max_element(parsed.lengths.begin(), parsed.lengths.end());
      if (shortest == 0 || area <= longest * longest * 1e-12) {
        ++stats.degenerateTriangles;
        continue;
      }
      for (double& component : parsed.normal) {
        component /= area;
      }
      if (!stats.trianglesValid) {
        origin = parsed.vertices[0].position;
      }
      ++stats.trianglesValid;
      parsed.valid = true;
      edgeLengths.insert(edgeLengths.end(), parsed.lengths.begin(), parsed.lengths.end());
    }
    if (stats.trianglesInvalid || stats.degenerateTriangles) {
      stats.status = WindowTopologyStatus::PartialInvalid;
    }
    if (edgeLengths.empty()) {
      return parent;
    }
    const auto median = edgeLengths.begin() + edgeLengths.size() / 2;
    std::nth_element(edgeLengths.begin(), median, edgeLengths.end());
    stats.positionTolerance = *median * 1e-5;
    const double bucketWidth = stats.positionTolerance * 2;
    using Bucket = std::array<int64_t, 3>;
    struct BucketHash {
      size_t operator()(const Bucket& key) const {
        return size_t(uint64_t(key[0]) * 0x9E3779B185EBCA87ull ^ uint64_t(key[1]) * 0xC2B2AE3D27D4EB4Full ^ uint64_t(key[2]) * 0x165667B19E3779F9ull);
      }
    };
    struct Edge {
      uint32_t triangle;
      uint32_t corner;
      uint32_t matches = 0;
      size_t mate = 0;
    };
    struct ReverseFace {
      uint32_t mate = 0;
      uint32_t matches = 0;
    };
    std::vector<ReverseFace> reverseFaces(input.triangleCount);
    std::vector<Edge> edges;
    std::unordered_map<Bucket, std::vector<size_t>, BucketHash> buckets;
    // Keep a small, contiguous membership filter in front of the sparse spatial map.
    // An unset bit proves that the bucket has never received an edge. Collisions
    // go through the original lookup, in the original traversal order.
    const size_t edgeCount = size_t(stats.trianglesValid) * 3;
    edges.reserve(edgeCount);
    buckets.reserve(edgeCount);
    size_t words = 1;
    while (words < edgeCount / 4 && words < (1u << 18)) {
      words *= 2;
    }
    std::vector<uint64_t> occupied(words);
    const size_t bitMask = words * 64 - 1;
    auto filterBit = [bitMask](const Bucket& cell) {
      uint64_t hash = BucketHash{}(cell);
      hash ^= hash >> 33;
      hash *= 0xff51afd7ed558ccdULL;
      return size_t(hash ^ (hash >> 32)) & bitMask;
    };
    auto endpointsMatch = [&](const Vertex& a, const Vertex& b, double tolerance) {
      const Point delta = subtract(a.position, b.position);
      return dot(delta, delta) <= tolerance * tolerance &&
        std::abs(a.uv[0] - b.uv[0]) <= stats.uvTolerance && std::abs(a.uv[1] - b.uv[1]) <= stats.uvTolerance;
    };
    for (uint32_t triangle = 0; triangle < input.triangleCount; ++triangle) {
      const auto& current = triangles[triangle];
      if (!current.valid) {
        continue;
      }
      for (uint32_t corner = 0; corner < 3; ++corner) {
        const auto& a = current.vertices[corner];
        const auto& b = current.vertices[(corner + 1) % 3];
        Bucket bucket;
        for (uint32_t axis = 0; axis < 3; ++axis) {
          const double cell = ((a.position[axis] - origin[axis]) * .5 + (b.position[axis] - origin[axis]) * .5) / bucketWidth;
          // Keep fractional precision for the neighboring-bucket search as well as integer range.
          if (!std::isfinite(cell) || std::abs(cell) >= 0x1p48) {
            stats.status = WindowTopologyStatus::NumericalRange;
            return parent;
          }
          bucket[axis] = int64_t(std::floor(cell));
        }
        Edge edge {triangle, corner};
        for (int x = -1; x <= 1; ++x) {
          for (int y = -1; y <= 1; ++y) {
            for (int z = -1; z <= 1; ++z) {
              const Bucket neighbor {bucket[0] + x, bucket[1] + y, bucket[2] + z};
              const size_t bit = filterBit(neighbor);
              if (!(occupied[bit / 64] & (uint64_t(1) << (bit % 64)))) {
                continue;
              }
              const auto found = buckets.find(neighbor);
              if (found == buckets.end()) {
                continue;
              }
              for (size_t previous : found->second) {
                auto& other = edges[previous];
                if (other.triangle == triangle) {
                  continue;
                }
                if (++stats.candidateComparisons > 2000000) {
                  stats.status = WindowTopologyStatus::CandidateLimit;
                  return parent;
                }
                const auto& otherTriangle = triangles[other.triangle];
                const auto& c = otherTriangle.vertices[other.corner];
                const auto& d = otherTriangle.vertices[(other.corner + 1) % 3];
                const double tolerance = std::min(stats.positionTolerance,
                  std::min(current.lengths[corner], otherTriangle.lengths[other.corner]) * 1e-4);
                const bool forwardEdge = endpointsMatch(a, c, tolerance) && endpointsMatch(b, d, tolerance);
                const bool reverseEdge = endpointsMatch(a, d, tolerance) && endpointsMatch(b, c, tolerance);
                if (!forwardEdge && !reverseEdge) {
                  continue;
                }
                const double normalAlignment = dot(current.normal, otherTriangle.normal);
                if (normalAlignment < .999) {
                  // A coincident, oppositely wound copy is the other side of the same pane.
                  // Record the pair once, and exclude backfaces from shared-edge ambiguity.
                  if (normalAlignment <= -.999 && corner == 0 &&
                      endpointsMatch(current.vertices[2], otherTriangle.vertices[(other.corner + 2) % 3], tolerance)) {
                    auto& left = reverseFaces[triangle];
                    auto& right = reverseFaces[other.triangle];
                    ++left.matches;
                    ++right.matches;
                    left.mate = other.triangle;
                    right.mate = triangle;
                  } else {
                    ++stats.rejectedFacePairs;
                  }
                  continue;
                }
                ++edge.matches;
                ++other.matches;
                edge.mate = previous;
                other.mate = edges.size();
              }
            }
          }
        }
        const size_t bit = filterBit(bucket);
        occupied[bit / 64] |= uint64_t(1) << (bit % 64);
        buckets[bucket].push_back(edges.size());
        edges.push_back(edge);
      }
    }
    auto root = [&parent](uint32_t i) {
      while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
      }
      return i;
    };
    auto join = [&](uint32_t a, uint32_t b) {
      const uint32_t rootA = root(a);
      const uint32_t rootB = root(b);
      if (rootA != rootB) {
        parent[std::max(rootA, rootB)] = std::min(rootA, rootB);
        ++stats.mergedTriangles;
      }
    };
    // Defer all unions until every incident edge is known, including non-manifold edges.
    for (size_t index = 0; index < edges.size(); ++index) {
      const auto& edge = edges[index];
      if (edge.matches > 1) {
        ++stats.ambiguousEdges;
      }
      if (edge.matches != 1 || edge.mate <= index || edges[edge.mate].matches != 1) {
        continue;
      }
      const auto& other = edges[edge.mate];
      const auto& a = triangles[edge.triangle];
      const auto& b = triangles[other.triangle];
      const auto& start = a.vertices[edge.corner].position;
      const auto direction = subtract(a.vertices[(edge.corner + 1) % 3].position, start);
      const auto sideA = cross(direction, subtract(a.vertices[(edge.corner + 2) % 3].position, start));
      const auto sideB = cross(direction, subtract(b.vertices[(other.corner + 2) % 3].position, start));
      if (dot(sideA, sideB) >= 0) {
        ++stats.rejectedFacePairs;
        continue;
      }
      ++stats.matchedEdges;
      join(edge.triangle, other.triangle);
    }
    for (uint32_t i = 0; i < input.triangleCount; ++i) {
      const auto& reverse = reverseFaces[i];
      if (reverse.matches == 1 && reverse.mate < i && reverseFaces[reverse.mate].matches == 1) {
        join(i, reverse.mate);
      }
    }
    for (uint32_t i = 0; i < input.triangleCount; ++i) {
      parent[i] = root(i);
    }
    return parent;
  }
}
