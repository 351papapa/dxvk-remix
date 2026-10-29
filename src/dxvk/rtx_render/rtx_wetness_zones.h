// SPDX-License-Identifier: MIT
// Local runtime feature source. See CREDITS.md.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace dxvk {
  class WetnessZoneIndex {
  public:
    struct Node {
      std::array<float, 4> lower {};
      std::array<float, 4> upper {};
    };
    static_assert(sizeof(Node) == 32);

    bool update(const std::string& wire) {
      if (wire == m_wire) {
        return false;
      }
      std::istringstream input(wire);
      input.imbue(std::locale::classic());
      uint32_t version = 0, count = 0;
      if (!(input >> version >> count) || version != 1 || count > 4096) {
        return false;
      }
      std::vector<Node> boxes(count);
      for (auto& box : boxes) {
        for (int axis = 0; axis < 3; ++axis) {
          if (!(input >> box.lower[axis] >> box.upper[axis]) ||
              !std::isfinite(box.lower[axis]) || !std::isfinite(box.upper[axis]) ||
              box.lower[axis] >= box.upper[axis]) {
            return false;
          }
        }
      }
      input >> std::ws;
      if (!input.eof()) {
        return false;
      }
      m_nodes.clear();
      if (!boxes.empty()) {
        build(boxes, 0, boxes.size());
      }
      m_wire = wire;
      return true;
    }

    const std::vector<Node>& nodes() const { return m_nodes; }

    bool contains(float x, float y, float z) const {
      const std::array<float, 3> position { x, y, z };
      size_t index = 0;
      while (index < m_nodes.size()) {
        const Node& node = m_nodes[index];
        bool inside = true;
        for (size_t axis = 0; axis < 3; ++axis) {
          inside = inside && position[axis] > node.lower[axis] && position[axis] < node.upper[axis];
        }
        if (inside) {
          if (node.upper[3] > 0.5f) { return true; }
          ++index;
        } else {
          index = static_cast<size_t>(node.lower[3]);
        }
      }
      return false;
    }

  private:
    void build(std::vector<Node>& boxes, size_t begin, size_t end) {
      Node bounds = boxes[begin];
      for (size_t i = begin + 1; i < end; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
          bounds.lower[axis] = std::min(bounds.lower[axis], boxes[i].lower[axis]);
          bounds.upper[axis] = std::max(bounds.upper[axis], boxes[i].upper[axis]);
        }
      }
      const size_t index = m_nodes.size();
      m_nodes.push_back(bounds);
      if (end - begin == 1) {
        m_nodes[index].upper[3] = 1.0f;
      } else {
        int axis = 0;
        for (int a = 1; a < 3; ++a) {
          if (bounds.upper[a] - bounds.lower[a] > bounds.upper[axis] - bounds.lower[axis]) {
            axis = a;
          }
        }
        const size_t mid = begin + (end - begin) / 2;
        std::nth_element(boxes.begin() + begin, boxes.begin() + mid, boxes.begin() + end,
          [axis](const Node& a, const Node& b) {
            return a.lower[axis] + a.upper[axis] < b.lower[axis] + b.upper[axis];
          });
        build(boxes, begin, mid);
        build(boxes, mid, end);
      }
      // Preorder skip links let the GPU traverse without a per-pixel stack.
      m_nodes[index].lower[3] = static_cast<float>(m_nodes.size());
    }
    std::string m_wire;
    std::vector<Node> m_nodes;
  };
}
