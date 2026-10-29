// SPDX-License-Identifier: MIT
// Local runtime window lighting. See CREDITS.md.
#pragma once
#include <algorithm>
#include <array>
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

  // Raw spans make buffer bounds and index widths explicit, including malformed imported meshes.
  struct WindowTopologyInput {
    WindowByteSpan positions;
    WindowByteSpan indices;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t positionStride = 0;
    uint32_t indexStride = 0;
    uint32_t indexWidth = 0; // 0 = non-indexed, 2/4 = uint16/uint32.
    uint32_t triangleCount = 0;
  };

  inline std::vector<uint32_t> buildWindowIslandRoots(const WindowTopologyInput& input) {
    std::vector<uint32_t> parent(input.triangleCount);
    std::iota(parent.begin(), parent.end(), 0u);
    if ((input.indexWidth != 0 && input.indexWidth != 2 && input.indexWidth != 4) || input.positions.empty() || input.positionStride < 12 ||
        (input.indexWidth != 0 && (input.indices.empty() || input.indexStride < input.indexWidth))) {
      return parent;
    }
    struct PositionHash {
      size_t operator()(const std::array<uint32_t, 3>& p) const {
        return size_t(p[0]) * 0x9E3779B1u ^ size_t(p[1]) * 0x85EBCA77u ^ size_t(p[2]) * 0xC2B2AE3Du;
      }
    };
    std::unordered_map<std::array<uint32_t, 3>, uint32_t, PositionHash> previous;
    auto root = [&parent](uint32_t i) {
      while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
      }
      return i;
    };
    for (uint32_t triangle = 0; triangle < input.triangleCount; ++triangle) {
      for (uint32_t corner = 0; corner < 3; ++corner) {
        const uint64_t index = uint64_t(triangle) * 3 + corner;
        uint32_t vertex = uint32_t(index);
        if (input.indexWidth) {
          const uint64_t offset = index * input.indexStride;
          if (index >= input.indexCount || offset + input.indexWidth > input.indices.size()) {
            continue;
          }
          vertex = 0;
          std::memcpy(&vertex, input.indices.data() + offset, input.indexWidth);
        }
        const uint64_t offset = uint64_t(vertex) * input.positionStride;
        if (vertex >= input.vertexCount || offset + 12 > input.positions.size()) {
          continue;
        }
        std::array<uint32_t, 3> position;
        std::memcpy(position.data(), input.positions.data() + offset, 12);
        const auto [it, inserted] = previous.emplace(position, triangle);
        if (!inserted) {
          const uint32_t a = root(triangle), b = root(it->second);
          // Canonical smallest triangle keeps identities independent of union traversal order.
          parent[std::max(a, b)] = std::min(a, b);
        }
      }
    }
    for (uint32_t i = 0; i < input.triangleCount; ++i) {
      parent[i] = root(i);
    }
    return parent;
  }
}
