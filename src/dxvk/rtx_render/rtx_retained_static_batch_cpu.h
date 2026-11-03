/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <new>
#include <utility>
#include <vector>

namespace dxvk {
template<typename Key, typename Value>
class RetainedStaticBatchLru {
  struct Item {
    uint64_t hash;
    Key key;
    Value value;
    size_t bytes;
  };
  std::list<Item> m_items;
  size_t m_bytes = 0;
  size_t m_limitBytes;
  size_t m_limitEntries;
  uint64_t m_hits = 0;
  uint64_t m_misses = 0;
  uint64_t m_evictions = 0;
public:
  RetainedStaticBatchLru(size_t limitBytes, size_t limitEntries)
    : m_limitBytes(limitBytes), m_limitEntries(limitEntries) { }

  const Value* find(uint64_t hash, const Key& key) {
    for (auto it = m_items.begin(); it != m_items.end(); ++it) {
      if (it->hash == hash && it->key == key) {
        m_items.splice(m_items.begin(), m_items, it);
        ++m_hits;
        return &m_items.front().value;
      }
    }
    ++m_misses;
    return nullptr;
  }

  bool insert(uint64_t hash, Key&& key, Value value, size_t bytes) {
    if (!bytes || bytes > m_limitBytes || !m_limitEntries) {
      return false;
    }
    for (const auto& item : m_items) {
      if (item.hash == hash && item.key == key) {
        return false;
      }
    }
    m_items.push_front({hash, std::move(key), std::move(value), bytes});
    m_bytes += bytes;
    while (m_bytes > m_limitBytes || m_items.size() > m_limitEntries) {
      m_bytes -= m_items.back().bytes;
      m_items.pop_back();
      ++m_evictions;
    }
    return true;
  }

  void clear() { m_items.clear(); m_bytes = 0; }
  size_t bytes() const { return m_bytes; }
  size_t size() const { return m_items.size(); }
  uint64_t hits() const { return m_hits; }
  uint64_t misses() const { return m_misses; }
  uint64_t evictions() const { return m_evictions; }
};

class RetainedStaticBatchCpu {
public:
  static constexpr uint32_t kVertexStride = 64;

  struct Source {
    const uint8_t* vertices = nullptr;
    size_t vertexBytes = 0;
    uint32_t vertexCount = 0;
    const uint8_t* indices = nullptr;
    size_t indexBytes = 0;
    uint32_t indexCount = 0;
  };

  struct Range {
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
  };

  struct Data {
    std::vector<uint8_t> vertices;
    std::vector<uint32_t> indices;
    std::vector<Range> ranges;
  };

  struct Limits {
    uint32_t maxVertices = 131072;
    uint32_t maxIndices = 192000;
    size_t maxBytes = 8 * 1024 * 1024;
  };

  enum class Result {
    Success,
    TooFewSources,
    InvalidRange,
    InvalidIndex,
    Limit,
    Allocation
  };

  // Sources must remain immutable for the call. Failure leaves output unchanged.
  static Result pack(const std::vector<Source>& sources, const Limits& limits, Data& output) {
    if (sources.size() < 2) {
      return Result::TooFewSources;
    }
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    for (const Source& source : sources) {
      if (!source.vertices || !source.indices || !source.vertexCount || !source.indexCount ||
          source.indexCount % 3 || uint64_t(source.vertexCount) * kVertexStride > source.vertexBytes ||
          uint64_t(source.indexCount) * sizeof(uint32_t) > source.indexBytes) {
        return Result::InvalidRange;
      }
      vertexCount += source.vertexCount;
      indexCount += source.indexCount;
      if (vertexCount > limits.maxVertices || indexCount > limits.maxIndices ||
          vertexCount > std::numeric_limits<uint32_t>::max() ||
          indexCount > std::numeric_limits<uint32_t>::max() ||
          vertexCount * kVertexStride + indexCount * sizeof(uint32_t) > limits.maxBytes) {
        return Result::Limit;
      }
      for (uint32_t i = 0; i < source.indexCount; ++i) {
        uint32_t index = 0;
        std::memcpy(&index, source.indices + size_t(i) * sizeof(index), sizeof(index));
        if (index >= source.vertexCount) {
          return Result::InvalidIndex;
        }
      }
    }
    try {
      Data packed;
      packed.vertices.resize(size_t(vertexCount) * kVertexStride);
      packed.indices.resize(size_t(indexCount));
      packed.ranges.reserve(sources.size());
      uint32_t firstVertex = 0;
      uint32_t firstIndex = 0;
      for (const Source& source : sources) {
        std::memcpy(packed.vertices.data() + size_t(firstVertex) * kVertexStride,
                    source.vertices, size_t(source.vertexCount) * kVertexStride);
        for (uint32_t i = 0; i < source.indexCount; ++i) {
          uint32_t index = 0;
          std::memcpy(&index, source.indices + size_t(i) * sizeof(index), sizeof(index));
          packed.indices[firstIndex + i] = firstVertex + index;
        }
        packed.ranges.push_back({firstVertex, source.vertexCount, firstIndex, source.indexCount});
        firstVertex += source.vertexCount;
        firstIndex += source.indexCount;
      }
      output = std::move(packed);
      return Result::Success;
    } catch (const std::bad_alloc&) {
      return Result::Allocation;
    }
  }
};

}
