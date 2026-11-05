/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#include "../../../src/dxvk/rtx_render/rtx_retained_static_batch_cpu.h"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
  void require(bool value, const char* message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }
struct Part {
    std::shared_ptr<int> buffer;
    uint32_t offset;
    uint32_t stride;
    bool operator==(const Part& other) const {
      return buffer == other.buffer && offset == other.offset && stride == other.stride;
    }
  };

  void cacheTests() {
    using Key = std::vector<Part>;
    using Cache = dxvk::RetainedStaticBatchLru<Key, std::shared_ptr<int>>;
    Cache cache(20, 2);
    auto a = std::make_shared<int>(1);
    auto b = std::make_shared<int>(2);
    std::weak_ptr<int> oldSource = b;
    auto liveChild = std::make_shared<int>(202);
    require(cache.insert(7, Key{{a, 0, 64}, {a, 192, 64}}, std::make_shared<int>(101), 10), "Cache insertion failed");
    require(cache.insert(7, Key{{b, 0, 64}}, liveChild, 10), "Colliding hash insertion failed");
    require(!cache.find(7, Key{{a, 192, 64}, {a, 0, 64}}), "Source order ignored");
    require(!cache.find(7, Key{{a, 0, 32}, {a, 192, 64}}), "Full layout key ignored");
    const auto* hit = cache.find(7, Key{{a, 0, 64}, {a, 192, 64}});
    require(hit && **hit == 101, "Exact key miss");
    b.reset();
    require(!oldSource.expired(), "Cached source lifetime was not held");
    auto c = std::make_shared<int>(3);
    require(cache.insert(9, Key{{c, 0, 64}}, std::make_shared<int>(303), 10), "LRU insertion failed");
    require(oldSource.expired(), "LRU eviction retained its source key");
    require(*liveChild == 202, "LRU eviction invalidated a live child");
    require(cache.bytes() == 20 && cache.size() == 2 && cache.evictions() == 1, "Cache budget or LRU accounting failed");
    require(cache.find(7, Key{{a, 0, 64}, {a, 192, 64}}), "MRU entry was evicted");
    require(!cache.insert(11, Key{{a, 0, 64}}, std::make_shared<int>(404), 21), "Oversized cache entry accepted");
    require(cache.bytes() == 20 && cache.size() == 2, "Oversized insertion changed cache");
    cache.clear();
    require(cache.bytes() == 0 && cache.size() == 0 && *liveChild == 202, "Cache clear invalidated a child");
    Cache entryBound(100, 1);
    require(entryBound.insert(1, Key{{a, 0, 64}}, std::make_shared<int>(1), 1), "Entry cap setup failed");
    require(entryBound.insert(2, Key{{c, 0, 64}}, std::make_shared<int>(2), 1), "Entry cap insertion failed");
    require(entryBound.size() == 1 && entryBound.evictions() == 1 &&
            !entryBound.find(1, Key{{a, 0, 64}}), "Entry cap not enforced");
  }
}

int main() {
  try {
    cacheTests();
    using Batch = dxvk::RetainedStaticBatchCpu;
    std::array<uint8_t, 3 * Batch::kVertexStride> a {};
    std::array<uint8_t, 4 * Batch::kVertexStride> b {};
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = uint8_t(i);
    }
    for (size_t i = 0; i < b.size(); ++i) {
      b[i] = uint8_t(i + 17);
    }
    std::array<uint32_t, 3> ia {2, 0, 1};
    std::array<uint32_t, 6> ib {0, 1, 2, 2, 1, 3};
    std::vector<Batch::Source> sources {
      {a.data(), a.size(), 3, reinterpret_cast<const uint8_t*>(ia.data()), sizeof(ia), 3},
      {b.data(), b.size(), 4, reinterpret_cast<const uint8_t*>(ib.data()), sizeof(ib), 6}
    };
    Batch::Data output;
    Batch::Limits limits;
    require(Batch::pack(sources, limits, output) == Batch::Result::Success, "Valid source fusion failed");
    require(output.vertices.size() == a.size() + b.size(), "Vertex count changed");
    require(!std::memcmp(output.vertices.data(), a.data(), a.size()), "First source bytes changed");
    require(!std::memcmp(output.vertices.data() + a.size(), b.data(), b.size()), "Second source bytes changed");
    require(output.indices == std::vector<uint32_t>({2, 0, 1, 3, 4, 5, 5, 4, 6}), "Indices were not rebased");
    require(output.ranges.size() == 2 && output.ranges[1].firstVertex == 3 &&
            output.ranges[1].firstIndex == 3 && output.ranges[1].indexCount == 6, "Source identity range changed");

    const auto originalVertices = output.vertices;
    const auto originalIndices = output.indices;
    ib[5] = 4;
    require(Batch::pack(sources, limits, output) == Batch::Result::InvalidIndex, "Out-of-range source index accepted");
    require(output.vertices == originalVertices && output.indices == originalIndices, "Rejected source changed output");
    ib[5] = 3;
    sources[1].vertexBytes--;
    require(Batch::pack(sources, limits, output) == Batch::Result::InvalidRange, "Truncated vertices accepted");
    sources[1].vertexBytes++;
    sources[1].indexBytes--;
    require(Batch::pack(sources, limits, output) == Batch::Result::InvalidRange, "Truncated indices accepted");
    sources[1].indexBytes++;
    limits.maxBytes = a.size() + b.size() + (ia.size() + ib.size()) * sizeof(uint32_t) - 1;
    require(Batch::pack(sources, limits, output) == Batch::Result::Limit, "Payload cap not enforced");
    limits = {};
    limits.maxVertices = 6;
    require(Batch::pack(sources, limits, output) == Batch::Result::Limit, "Vertex cap not enforced");
    limits = {};
    limits.maxIndices = 8;
    require(Batch::pack(sources, limits, output) == Batch::Result::Limit, "Index cap not enforced");
    limits = {};
    const auto originalCount = sources[1].vertexCount;
    const auto originalBytes = sources[1].vertexBytes;
    sources[1].vertexCount = std::numeric_limits<uint32_t>::max();
    sources[1].vertexBytes = std::numeric_limits<size_t>::max();
    require(Batch::pack(sources, limits, output) == Batch::Result::Limit, "Large source bypassed overflow preflight");
    sources[1].vertexCount = originalCount;
    sources[1].vertexBytes = originalBytes;
    sources[1].indexCount = 5;
    require(Batch::pack(sources, limits, output) == Batch::Result::InvalidRange, "Incomplete triangle accepted");
    require(Batch::pack({}, limits, output) == Batch::Result::TooFewSources, "Empty fusion accepted");
    require(output.vertices == originalVertices && output.indices == originalIndices, "Failure was not atomic");

    std::cout << "Retained static batch CPU tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
