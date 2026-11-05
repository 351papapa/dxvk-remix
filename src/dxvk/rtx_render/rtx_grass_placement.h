// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <atomic>
#include <cfloat>
#include <map>
#include <vector>
#include "rtx/pass/grass/grass_args.h"

namespace dxvk {
// Deterministic placement shared by the asynchronous worker and CPU validation.
class GrassPatchBuilder {
public:
  static constexpr size_t kMaxPatchPoints = 65536;
  static std::vector<GrassRoot> build(const std::vector<std::array<Vector3, 3>>& triangles,
      const Matrix4& world, float slope, float density, float margin, bool inset,
      bool zUp, const std::atomic<bool>& cancelled) {
    if (!std::isfinite(slope) || !std::isfinite(density) || !std::isfinite(margin) || density <= 0 || margin < 0 || cancelled) {
      return {};
    }
    std::vector<GrassRoot> points;
    std::vector<Triangle> eligible;
    std::map<EdgeKey, Edge> boundary;
    const bool mirrored = dot(cross(world[0].xyz(), world[1].xyz()), world[2].xyz()) < 0;
    for (const auto& local : triangles) {
      if (cancelled.load(std::memory_order_relaxed)) { return {}; }
      Triangle tri;
      for (unsigned j = 0; j < 3; ++j) { tri.p[j] = (world * Vector4(local[j], 1)).xyz(); }
      if (!finite(tri.p[0]) || !finite(tri.p[1]) || !finite(tri.p[2])) { continue; }
      Vector3 normal = cross(tri.p[1] - tri.p[0], tri.p[2] - tri.p[0]);
      const float length = std::sqrt(dot(normal, normal));
      if (length < 1e-8f) { continue; }
      normal /= length;
      if (mirrored) { normal = -normal; }
      if (normal[zUp ? 2 : 1] < slope) { continue; }
      tri.normal = normal;
      tri.area = length * 0.5f;
      tri.seed = XXH64(tri.p, sizeof(tri.p), 0x4752415353ull);
      eligible.push_back(tri);
      if (inset) {
        for (unsigned j = 0; j < 3; ++j) {
          auto a = vertexKey(tri.p[j]), b = vertexKey(tri.p[(j + 1) % 3]);
          if (b < a) { std::swap(a, b); }
          auto& edge = boundary[{ a, b }];
          edge.a = tri.p[j]; edge.b = tri.p[(j + 1) % 3]; ++edge.count;
        }
      }
    }
    std::vector<Edge> edges;
    for (const auto& edge : boundary) { if (edge.second.count == 1) { edges.push_back(edge.second); } }
    const EdgeTree tree(std::move(edges));
    size_t attempts = 0;
    for (const auto& tri : eligible) {
      const size_t count = size_t(std::min(double(tri.area) * density + randomUnit(tri.seed, 1), double(kMaxPatchPoints)));
      for (size_t i = 0; i < count; ++i) {
        if (points.size() >= kMaxPatchPoints || ++attempts > 262144 || cancelled.load(std::memory_order_relaxed)) { break; }
        const uint64_t seed = XXH64(&i, sizeof(i), tri.seed);
        const float u = std::sqrt(randomUnit(seed, 2)), v = randomUnit(seed, 3);
        const Vector3 point = tri.p[0] * (1 - u) + tri.p[1] * (u * (1 - v)) + tri.p[2] * (u * v);
        const float scale = 0.85f + randomUnit(seed, 4) * 0.3f;
        if (tree.intersects(point, margin * scale)) { continue; }
        GrassRoot root {};
        root.positionScale = { point.x, point.y, point.z, scale };
        root.normalAngle = { tri.normal.x, tri.normal.y, tri.normal.z, randomUnit(seed, 5) * 6.28318530718f };
        root.identityLo = static_cast<uint32_t>(seed);
        root.identityHi = static_cast<uint32_t>(seed >> 32);
        points.push_back(root);
      }
      if (points.size() >= kMaxPatchPoints || attempts > 262144 || cancelled.load(std::memory_order_relaxed)) { break; }
    }
    return cancelled ? std::vector<GrassRoot>() : std::move(points);
  }

private:
  static float randomUnit(uint64_t seed, uint64_t channel) {
    return float(XXH64(&seed, sizeof(seed), channel) >> 40) * (1.0f / 16777216.0f);
  }

  struct Triangle { Vector3 p[3]; Vector3 normal; float area; uint64_t seed; };
  struct Edge { Vector3 a; Vector3 b; uint32_t count = 0; };
  using VertexKey = std::array<int64_t, 3>;
  using EdgeKey = std::pair<VertexKey, VertexKey>;

  static VertexKey vertexKey(const Vector3& p) {
    return { int64_t(std::llround(p.x * 10000.0)), int64_t(std::llround(p.y * 10000.0)), int64_t(std::llround(p.z * 10000.0)) };
  }

  static float segmentDistanceSquared(const Vector3& p, const Edge& edge) {
    const Vector3 v = edge.b - edge.a;
    const float t = std::clamp(dot(p - edge.a, v) / std::max(dot(v, v), 1e-12f), 0.0f, 1.0f);
    const Vector3 d = p - (edge.a + v * t);
    return dot(d, d);
  }

  class EdgeTree {
  public:
    explicit EdgeTree(std::vector<Edge>&& edges) : m_edges(std::move(edges)) {
      if (!m_edges.empty()) { build(0, static_cast<uint32_t>(m_edges.size())); }
    }
    bool intersects(const Vector3& point, float radius) const {
      return !m_nodes.empty() && intersects(0, point, radius * radius);
    }
  private:
    struct Node { Vector3 low, high; uint32_t begin, end, left = 0, right = 0; };
    std::vector<Edge> m_edges;
    std::vector<Node> m_nodes;
    uint32_t build(uint32_t begin, uint32_t end) {
      Node node { Vector3(FLT_MAX), Vector3(-FLT_MAX), begin, end };
      for (uint32_t i = begin; i < end; ++i) {
        node.low = min(node.low, min(m_edges[i].a, m_edges[i].b));
        node.high = max(node.high, max(m_edges[i].a, m_edges[i].b));
      }
      const uint32_t index = static_cast<uint32_t>(m_nodes.size());
      m_nodes.push_back(node);
      if (end - begin > 8) {
        const auto extent = node.high - node.low;
        const unsigned axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2);
        const uint32_t middle = (begin + end) / 2;
        std::nth_element(m_edges.begin() + begin, m_edges.begin() + middle, m_edges.begin() + end,
          [axis](const Edge& a, const Edge& b) { return a.a[axis] + a.b[axis] < b.a[axis] + b.b[axis]; });
        const uint32_t left = build(begin, middle);
        const uint32_t right = build(middle, end);
        m_nodes[index].left = left;
        m_nodes[index].right = right;
      }
      return index;
    }
    bool intersects(uint32_t index, const Vector3& point, float radiusSquared) const {
      const auto& node = m_nodes[index];
      const Vector3 distance = max(max(node.low - point, point - node.high), Vector3(0.0f));
      if (dot(distance, distance) >= radiusSquared) { return false; }
      if (node.left) { return intersects(node.left, point, radiusSquared) || intersects(node.right, point, radiusSquared); }
      for (uint32_t i = node.begin; i < node.end; ++i) {
        if (segmentDistanceSquared(point, m_edges[i]) < radiusSquared) { return true; }
      }
      return false;
    }
  };

  static bool finite(const Vector3& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
           std::abs(p.x) < 1e7f && std::abs(p.y) < 1e7f && std::abs(p.z) < 1e7f;
  }

};
}
