/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#include "../../../src/dxvk/rtx_render/rtx_retained_world_batch_cpu.h"
#include "../../../src/dxvk/rtx_render/rtx_retained_batch_policy.h"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
  int64_t allocationBudget = -1;
}

void* operator new(std::size_t bytes) {
  if (allocationBudget == 0) {
    throw std::bad_alloc();
  }
  if (allocationBudget > 0) {
    --allocationBudget;
  }
  if (void* result = std::malloc(bytes ? bytes : 1)) {
    return result;
  }
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

namespace {
  using Cpu = dxvk::RetainedWorldBatchCpu;

  void require(bool value, const char* message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  bool near(float a, float b) {
    return std::abs(a - b) < 0.0001f;
  }

  // Round-based reference retained only for equivalence tests of the production coverage planner.
  std::vector<bool> completeInputsReference(const std::vector<uint32_t>& partCounts,
                                          const std::vector<std::vector<Cpu::PartRef>>& buckets) {
    std::vector<bool> active(partCounts.size(), true);
    for (size_t i = 0; i < partCounts.size(); ++i) {
      active[i] = partCounts[i] != 0;
    }
    bool changed;
    do {
      changed = false;
      std::vector<std::vector<uint8_t>> covered(partCounts.size());
      for (size_t i = 0; i < partCounts.size(); ++i) {
        covered[i].resize(partCounts[i]);
      }
      for (const auto& bucket : buckets) {
        uint32_t first = UINT32_MAX;
        bool crossInput = false;
        for (const Cpu::PartRef& ref : bucket) {
          if (ref.input >= active.size() || ref.part >= partCounts[ref.input] || !active[ref.input]) {
            continue;
          }
          if (first == UINT32_MAX) {
            first = ref.input;
          } else if (first != ref.input) {
            crossInput = true;
          }
        }
        if (crossInput) {
          for (const Cpu::PartRef& ref : bucket) {
            if (ref.input < active.size() && active[ref.input] && ref.part < partCounts[ref.input]) {
              covered[ref.input][ref.part] = 1;
            }
          }
        }
      }
      for (size_t i = 0; i < active.size(); ++i) {
        if (active[i] && std::find(covered[i].begin(), covered[i].end(), uint8_t(0)) != covered[i].end()) {
          active[i] = false;
          changed = true;
        }
      }
    } while (changed);
    return active;
  }

  void coverageDifferentialTests() {
    using Ref = Cpu::PartRef;
    uint64_t checked = 0, nonempty = 0;
    const auto compare = [&](const std::vector<uint32_t>& counts, const std::vector<std::vector<Ref>>& buckets) {
      const auto expected = completeInputsReference(counts, buckets);
      require(Cpu::completeInputs(counts, buckets) == expected, "CSR coverage differs from the round reference");
      ++checked;
      nonempty += std::find(expected.begin(), expected.end(), true) != expected.end();
    };

    // Every ordered pair of empty/one/two-ref buckets, including duplicate and invalid refs.
    const std::vector<Ref> alphabet {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {0, 2}};
    std::vector<std::vector<Ref>> choices(1);
    for (const Ref& a : alphabet) {
      choices.push_back({a});
      for (const Ref& b : alphabet) {
        choices.push_back({a, b});
      }
    }
    for (uint32_t a = 0; a <= 2; ++a) {
      for (uint32_t b = 0; b <= 2; ++b) {
        for (const auto& first : choices) {
          for (const auto& second : choices) {
            compare({a, b}, {first, second});
          }
        }
      }
    }
    // Exhaust all three-owner bucket graphs while varying zero/multipart input descriptions.
    for (uint32_t counts = 0; counts < 27; ++counts) {
      const std::vector<uint32_t> parts {counts % 3, (counts / 3) % 3, counts / 9};
      for (uint32_t graph = 0; graph < 512; ++graph) {
        std::vector<std::vector<Ref>> buckets(3);
        for (uint32_t bucket = 0; bucket < 3; ++bucket) {
          const uint32_t mask = (graph >> (bucket * 3)) & 7;
          for (uint32_t owner = 0; owner < 3; ++owner) {
            if ((mask & (1u << owner)) != 0) {
              buckets[bucket].push_back({owner, 0});
            }
          }
        }
        compare(parts, buckets);
      }
    }
    const uint64_t exhaustive = checked;
    uint64_t randomState = 0x7e295041386bcda1ull;
    const auto random = [&]() {
      randomState ^= randomState << 13;
      randomState ^= randomState >> 7;
      randomState ^= randomState << 17;
      return randomState;
    };
    for (uint32_t iteration = 0; iteration < 12000; ++iteration) {
      std::vector<uint32_t> counts(size_t(random() % 11));
      for (auto& count : counts) {
        count = uint32_t(random() % 4);
      }
      std::vector<std::vector<Ref>> buckets(size_t(random() % 13));
      for (auto& bucket : buckets) {
        const size_t refs = size_t(random() % 17);
        for (size_t i = 0; i < refs; ++i) {
          const Ref ref {uint32_t(random() % (counts.size() + 3)), uint32_t(random() % 6)};
          bucket.push_back(ref);
          if (random() % 4 == 0) {
            bucket.push_back(ref);
          }
        }
      }
      if (iteration % 3 == 0) {
        buckets.emplace_back();
        for (uint32_t input = 0; input < counts.size(); ++input) {
          for (uint32_t part = 0; part < counts[input]; ++part) {
            buckets.back().push_back({input, part});
          }
        }
      }
      compare(counts, buckets);
      std::reverse(buckets.begin(), buckets.end());
      for (auto& bucket : buckets) {
        std::reverse(bucket.begin(), bucket.end());
      }
      compare(counts, buckets);
    }
    require(nonempty > 100, "Coverage differential fixtures did not exercise surviving ownership");

    // Only the tail's missing part starts this cascade. Every preceding loss crosses a bucket's 2->1 boundary.
    const auto chain = [](uint32_t count) {
      std::vector<std::vector<Ref>> buckets;
      buckets.reserve(count - 1);
      for (uint32_t input = 0; input + 1 < count; ++input) {
        buckets.push_back({{input, input ? 1u : 0u}, {input + 1, 0}});
      }
      return buckets;
    };
    auto chainBuckets = chain(512);
    std::vector<uint32_t> chainCounts(512, 2);
    chainCounts.front() = 1;
    compare(chainCounts, chainBuckets);
    require(Cpu::completeInputs(chainCounts, chainBuckets) == std::vector<bool>(512, false),
            "The missing tail part did not propagate through the entire long chain");
    chainCounts.back() = 1;
    compare(chainCounts, chainBuckets);
    require(Cpu::completeInputs(chainCounts, chainBuckets) == std::vector<bool>(512, true),
            "A complete long chain was peeled despite covering every part");
    chainBuckets = chain(8192);
    chainCounts.assign(8192, 2);
    chainCounts.front() = 1;
    require(Cpu::completeInputs(chainCounts, chainBuckets) == std::vector<bool>(8192, false),
            "Deep cascade stopped early or lost an input during queue growth");
    const std::vector<uint32_t> failureCounts {2, 2, 1, 0};
    const std::vector<std::vector<Ref>> failureBuckets {
      {{0, 0}, {0, 0}, {1, 0}, {3, 0}}, {{0, 1}, {1, 1}, {2, 0}}, {{8, 0}, {1, 9}}
    };
    const auto expected = completeInputsReference(failureCounts, failureBuckets);
    uint32_t allocationThrows = 0, allocationSuccesses = 0;
    for (int64_t budget = 0; budget < 40; ++budget) {
      const std::vector<bool> before {true, false, true};
      std::vector<bool> output = before;
      bool completed = false;
      try {
        allocationBudget = budget;
        output = Cpu::completeInputs(failureCounts, failureBuckets);
        completed = true;
      } catch (const std::bad_alloc&) {
        ++allocationThrows;
      }
      allocationBudget = -1;
      require(output == (completed ? expected : before),
              "A real CSR allocation failure changed the caller's previously published coverage");
      allocationSuccesses += completed;
    }
    require(allocationThrows && allocationSuccesses, "CSR allocation injection missed failure or successful output");
    std::cout << "Coverage differential fixtures: exhaustive=" << exhaustive << " compared=" << checked
              << " nonempty=" << nonempty << " allocationThrows=" << allocationThrows << '\n';
  }

  void coverageTests() {
    using Ref = Cpu::PartRef;
    // Input zero has two enhanced parts. Losing either part also removes its paired input.
    const std::vector<uint32_t> counts {2, 1, 1};
    const std::vector<std::vector<Ref>> incomplete {{{0, 0}, {1, 0}}, {{0, 1}}, {{2, 0}}};
    require(Cpu::completeInputs(counts, incomplete) == std::vector<bool>({false, false, false}),
            "Incomplete enhanced input or its cascading partner survived");
    const std::vector<std::vector<Ref>> complete {{{0, 0}, {1, 0}}, {{0, 1}, {2, 0}}};
    require(Cpu::completeInputs(counts, complete) == std::vector<bool>({true, true, true}),
            "Fully covered multipart enhanced input was rejected");
    require(Cpu::completeInputs({2}, {{{0, 0}, {0, 1}}}) == std::vector<bool>({false}),
            "A single input was mistaken for a cross-parent batch");
    require(Cpu::completeInputs({0, 1}, {{{1, 0}, {0, 0}}}) == std::vector<bool>({false, false}),
            "Empty replacement or invalid part gained ownership");
    // Removing input zero breaks input one; that breaks input two on the next pass.
    require(Cpu::completeInputs({2, 2, 1}, {{{0, 0}, {1, 0}}, {{1, 1}, {2, 0}}}) ==
            std::vector<bool>({false, false, false}), "Coverage pruning stopped before reaching a fixed point");
  }

  // Frozen R10 implementation is an independent byte/acceptance oracle, not production code.
  bool validTransformReferenceR10(const std::array<float, 16>& m) {
    for (float value : m) {
      if (!std::isfinite(value)) {
        return false;
      }
    }
    const double determinant = double(m[0]) * (double(m[5]) * m[10] - double(m[9]) * m[6]) -
      double(m[4]) * (double(m[1]) * m[10] - double(m[9]) * m[2]) +
      double(m[8]) * (double(m[1]) * m[6] - double(m[5]) * m[2]);
    return m[3] == 0 && m[7] == 0 && m[11] == 0 && m[15] == 1 &&
      std::isfinite(determinant) && determinant > 1e-12;
  }

  bool packReferenceR10(const std::vector<Cpu::Source>& sources, const Cpu::Limits& limits, Cpu::Data& output) {
    using Source = Cpu::Source;
    using Vertex = Cpu::Vertex;
    using Data = Cpu::Data;
    uint64_t vertices = 0;
    uint64_t indices = 0;
    if (sources.size() < 2) {
      return false;
    }
    for (const Source& source : sources) {
      if (!source.vertices || !source.indices || !source.vertexCount || !source.indexCount ||
          source.indexCount % 3 || !validTransformReferenceR10(source.transform)) {
        return false;
      }
      vertices += source.vertexCount;
      indices += source.indexCount;
      if (vertices > limits.maxVertices || indices > limits.maxIndices ||
          vertices > UINT32_MAX || indices > UINT32_MAX ||
          vertices * sizeof(Vertex) + indices * sizeof(uint32_t) > limits.maxBytes) {
        return false;
      }
      for (uint32_t i = 0; i < source.indexCount; ++i) {
        if (source.indices[i] >= source.vertexCount) {
          return false;
        }
      }
    }
    try {
      Data packed;
      packed.vertices.reserve(size_t(vertices));
      packed.indices.reserve(size_t(indices));
      packed.ranges.reserve(sources.size());
      for (const Source& source : sources) {
        const uint32_t firstVertex = uint32_t(packed.vertices.size());
        const uint32_t firstIndex = uint32_t(packed.indices.size());
        const auto& m = source.transform;
        // Cofactor rows are the inverse transpose, apart from a positive scalar.
        const double cofactors[9] = {
          double(m[5]) * m[10] - double(m[9]) * m[6], double(m[9]) * m[2] - double(m[1]) * m[10], double(m[1]) * m[6] - double(m[5]) * m[2],
          double(m[8]) * m[6] - double(m[4]) * m[10], double(m[0]) * m[10] - double(m[8]) * m[2], double(m[4]) * m[2] - double(m[0]) * m[6],
          double(m[4]) * m[9] - double(m[8]) * m[5], double(m[8]) * m[1] - double(m[0]) * m[9], double(m[0]) * m[5] - double(m[4]) * m[1]
        };
        for (uint32_t i = 0; i < source.vertexCount; ++i) {
          Vertex vertex = source.vertices[i];
          const double x = vertex.position[0];
          const double y = vertex.position[1];
          const double z = vertex.position[2];
          for (uint32_t axis = 0; axis < 3; ++axis) {
            vertex.position[axis] = float(m[axis] * x + m[4 + axis] * y + m[8 + axis] * z + m[12 + axis]);
            if (!std::isfinite(vertex.position[axis])) {
              return false;
            }
          }
          if (source.normals) {
            const double nx = cofactors[0] * vertex.normal[0] + cofactors[1] * vertex.normal[1] + cofactors[2] * vertex.normal[2];
            const double ny = cofactors[3] * vertex.normal[0] + cofactors[4] * vertex.normal[1] + cofactors[5] * vertex.normal[2];
            const double nz = cofactors[6] * vertex.normal[0] + cofactors[7] * vertex.normal[1] + cofactors[8] * vertex.normal[2];
            const double length = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (!std::isfinite(length) || length <= 1e-20) {
              return false;
            }
            vertex.normal[0] = float(nx / length);
            vertex.normal[1] = float(ny / length);
            vertex.normal[2] = float(nz / length);
          }
          if (!std::isfinite(vertex.texcoord[0]) || !std::isfinite(vertex.texcoord[1])) {
            return false;
          }
          packed.vertices.push_back(vertex);
        }
        for (uint32_t i = 0; i < source.indexCount; ++i) {
          packed.indices.push_back(firstVertex + source.indices[i]);
        }
        packed.ranges.push_back({firstVertex, source.vertexCount, firstIndex, source.indexCount});
      }
      output = std::move(packed);
      return true;
    } catch (const std::bad_alloc&) {
      return false;
    }
  }

  bool samePackedBytes(const Cpu::Data& a, const Cpu::Data& b) {
    if (a.vertices.size() != b.vertices.size() || a.indices != b.indices || a.ranges.size() != b.ranges.size() ||
        (!a.vertices.empty() && std::memcmp(a.vertices.data(), b.vertices.data(), a.vertices.size() * sizeof(Cpu::Vertex)))) {
      return false;
    }
    for (size_t i = 0; i < a.ranges.size(); ++i) {
      const auto& x = a.ranges[i];
      const auto& y = b.ranges[i];
      if (x.firstVertex != y.firstVertex || x.vertexCount != y.vertexCount ||
          x.firstIndex != y.firstIndex || x.indexCount != y.indexCount) {
        return false;
      }
    }
    return true;
  }

  // Byte readers exercise the production callback contract without Vulkan dependencies.
  // Runtime format admission remains separately reviewed in the real RasterBuffer adapter.
  struct ReaderFixture {
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> indexBytes;
    std::vector<Cpu::Vertex> decoded;
    std::vector<uint32_t> decodedIndices;
    Cpu::ReadSource source;
    uint32_t positionComponents = 3;
    uint32_t normalComponents = 3;
    uint32_t uvComponents = 2;
    bool oct = false;
    bool uv = true;
    bool color = true;
    bool index16 = false;

    bool vertex(uint32_t i, Cpu::Vertex& out) const {
      const size_t start = 8 + size_t(i) * 64;
      const auto fits = [&](size_t offset, size_t length) {
        return offset <= bytes.size() && length <= bytes.size() - offset;
      };
      if (i >= source.vertexCount || !fits(start, positionComponents * 4) ||
          (source.normals && !fits(start + 16, oct ? 4 : normalComponents * 4)) ||
          (uv && !fits(start + 32, uvComponents * 4)) || (color && !fits(start + 48, 4))) {
        return false;
      }
      std::memcpy(out.position, bytes.data() + start, 12);
      if (source.normals) {
        if (oct) {
          uint32_t value;
          std::memcpy(&value, bytes.data() + start + 16, 4);
          Cpu::decodeOctNormal(value, out.normal);
        } else {
          std::memcpy(out.normal, bytes.data() + start + 16, 12);
        }
      }
      if (uv) {
        std::memcpy(out.texcoord, bytes.data() + start + 32, 8);
      }
      if (color) {
        std::memcpy(&out.color, bytes.data() + start + 48, 4);
      }
      return true;
    }

    bool index(uint32_t i, uint32_t& out) const {
      const size_t width = index16 ? 2 : 4;
      const size_t offset = 4 + size_t(i) * width;
      if (i >= source.indexCount || offset > indexBytes.size() || width > indexBytes.size() - offset) {
        return false;
      }
      if (index16) {
        uint16_t value;
        std::memcpy(&value, indexBytes.data() + offset, 2);
        out = value;
      } else {
        std::memcpy(&out, indexBytes.data() + offset, 4);
      }
      return true;
    }
  };

  ReaderFixture makeReaderFixture(uint32_t variant) {
    ReaderFixture f;
    f.source.vertexCount = 3 + variant;
    f.source.indexCount = variant == 1 ? 6 : 3;
    f.source.normals = variant != 2;
    f.positionComponents = variant == 1 ? 4 : 3;
    f.normalComponents = variant == 0 ? 4 : 3;
    f.uvComponents = variant == 0 ? 4 : variant == 1 ? 3 : 2;
    f.oct = variant == 1;
    f.uv = f.color = variant != 2;
    f.index16 = variant != 0;
    f.source.transform = {0, 2, 0, 0, -3, 1, 0, 0, 0, 0, 4, 0, -17, 8, 3, 1};
    f.bytes.resize(8 + size_t(f.source.vertexCount) * 64, 0xcd);
    f.indexBytes.resize(4 + size_t(f.source.indexCount) * (f.index16 ? 2 : 4), 0xef);
    f.decoded.resize(f.source.vertexCount);
    for (uint32_t i = 0; i < f.source.vertexCount; ++i) {
      const size_t start = 8 + size_t(i) * 64;
      const float ignored = std::numeric_limits<float>::quiet_NaN();
      const float position[] = {float(i) * .375f - 2, float(variant) - .125f * i, i ? .25f * i : -0.f, ignored};
      const float normal[] = {.25f + .125f * i, 1.f, -.375f, ignored};
      const float uv[] = {i ? float(i) / 7.f : -0.f, -.125f * variant, ignored, ignored};
      const uint32_t color = 0x13579bdfu ^ (i * 0x01010101u);
      std::memcpy(f.bytes.data() + start, position, 16);
      std::memcpy(f.decoded[i].position, position, 12);
      if (f.source.normals) {
        if (f.oct) {
          const uint32_t encoded[] = {0x80008000u, 0xffffffffu, 0x8000ffffu, 0x1234a001u};
          const uint32_t value = encoded[i % 4];
          std::memcpy(f.bytes.data() + start + 16, &value, 4);
          Cpu::decodeOctNormal(value, f.decoded[i].normal);
        } else {
          std::memcpy(f.bytes.data() + start + 16, normal, 16);
          std::memcpy(f.decoded[i].normal, normal, 12);
        }
      }
      if (f.uv) {
        std::memcpy(f.bytes.data() + start + 32, uv, 16);
        std::memcpy(f.decoded[i].texcoord, uv, 8);
      }
      if (f.color) {
        std::memcpy(f.bytes.data() + start + 48, &color, 4);
        f.decoded[i].color = color;
      }
    }
    for (uint32_t i = 0; i < f.source.indexCount; ++i) {
      const uint32_t index = (2 + i) % 3; // Extra vertices remain unreferenced but must still be processed.
      f.decodedIndices.push_back(index);
      if (f.index16) {
        const uint16_t narrow = uint16_t(index);
        std::memcpy(f.indexBytes.data() + 4 + i * 2, &narrow, 2);
      } else {
        std::memcpy(f.indexBytes.data() + 4 + i * 4, &index, 4);
      }
    }
    return f;
  }

  void readerPackingTests() {
    std::vector<ReaderFixture> fixtures;
    fixtures.push_back(makeReaderFixture(0));
    fixtures.push_back(makeReaderFixture(1));
    fixtures.push_back(makeReaderFixture(2));
    std::vector<Cpu::ReadSource> descriptions;
    std::vector<Cpu::Source> referenceSources;
    for (const auto& f : fixtures) {
      descriptions.push_back(f.source);
      referenceSources.push_back({f.decoded.data(), f.source.vertexCount, f.decodedIndices.data(), f.source.indexCount,
                                 f.source.transform, f.source.normals});
    }
    uint32_t vertexReads = 0, indexReads = 0;
    auto vertex = [&](size_t source, uint32_t index, Cpu::Vertex& out) {
      ++vertexReads;
      return fixtures.at(source).vertex(index, out);
    };
    auto index = [&](size_t source, uint32_t at, uint32_t& out) {
      ++indexReads;
      return fixtures.at(source).index(at, out);
    };
    Cpu::Data expected, actual;
    require(packReferenceR10(referenceSources, {}, expected) &&
            Cpu::packRead(descriptions, {}, vertex, index, actual) && samePackedBytes(expected, actual),
            "Streaming strided float/octet/u16/u32 pack changed frozen R10 bytes, indices or ownership ranges");
    require(vertexReads == 12 && indexReads == 12, "A source vertex or index was decoded more than once");
    require(Cpu::pack(referenceSources, {}, actual) && samePackedBytes(expected, actual),
            "Legacy pointer pack did not use the byte-identical common reader implementation");
    const uint32_t absent = expected.ranges.back().firstVertex;
    require(actual.vertices[absent].normal[0] == 0 && actual.vertices[absent].normal[1] == 0 &&
            actual.vertices[absent].normal[2] == 0 && actual.vertices[absent].texcoord[0] == 0 &&
            actual.vertices[absent].texcoord[1] == 0 && actual.vertices[absent].color == 0xffffffffu,
            "Missing attributes did not receive exact zero/white defaults");

    // Differential transform/rounding coverage uses actual frozen arithmetic, not a mirrored new algorithm.
    uint32_t random = 0x7b13a9u;
    for (uint32_t iteration = 0; iteration < 192; ++iteration) {
      for (size_t source = 0; source < descriptions.size(); ++source) {
        random = random * 1664525u + 1013904223u;
        const float sx = .25f + float(random & 15u) * .125f;
        random = random * 1664525u + 1013904223u;
        const float sy = .5f + float(random & 15u) * .25f;
        const float sh = float(int((random >> 16) & 15u) - 7) * .0625f;
        const std::array<float, 16> matrix = {0, sx, 0, 0, -sy, sh, 0, 0, 0, 0, 1.25f, 0,
          float(int(random & 255u) - 127), -.125f * iteration, float(source) * .25f, 1};
        descriptions[source].transform = matrix;
        referenceSources[source].transform = matrix;
      }
      require(packReferenceR10(referenceSources, {}, expected) &&
              Cpu::packRead(descriptions, {}, vertex, index, actual) && samePackedBytes(expected, actual),
              "Reader pack changed exact normal/position rounding under rotation, scale or shear");
    }
    Cpu::Data sentinel = expected;
    sentinel.vertices[0].position[0] = -12345.f;
    sentinel.indices[0] = 98765;
    const auto failAtomic = [&](auto&& vr, auto&& ir) {
      actual = sentinel;
      require(!Cpu::packRead(descriptions, {}, vr, ir, actual) && samePackedBytes(actual, sentinel),
              "Reader rejection/exception published a partial vertex or index prefix");
    };
    failAtomic([&](size_t source, uint32_t at, Cpu::Vertex& out) {
      return source == 2 && at == 4 ? false : vertex(source, at, out);
    }, index);
    failAtomic([&](size_t source, uint32_t at, Cpu::Vertex& out) {
      if (source == 2 && at == 4) {
        throw std::runtime_error("late vertex-reader failure");
      }
      return vertex(source, at, out);
    }, index);
    failAtomic(vertex, [&](size_t source, uint32_t at, uint32_t& out) {
      if (source == 2 && at == 2) {
        return false;
      }
      return index(source, at, out);
    });
    failAtomic(vertex, [&](size_t source, uint32_t at, uint32_t& out) {
      if (source == 2 && at == 2) {
        throw std::runtime_error("late index-reader failure");
      }
      return index(source, at, out);
    });
    failAtomic(vertex, [&](size_t source, uint32_t at, uint32_t& out) {
      if (source == 1 && at == 5) {
        out = descriptions[source].vertexCount;
        return true;
      }
      return index(source, at, out);
    });
    failAtomic([&](size_t source, uint32_t at, Cpu::Vertex& out) {
      const bool result = vertex(source, at, out);
      if (source == 2 && at == 4) {
        out.position[0] = std::numeric_limits<float>::infinity();
      }
      return result;
    }, index); // Includes the unreferenced final vertex.
    failAtomic([&](size_t source, uint32_t at, Cpu::Vertex& out) {
      const bool result = vertex(source, at, out);
      if (source == 1 && at == 0) {
        out.normal[0] = std::numeric_limits<float>::quiet_NaN();
      }
      return result;
    }, index);
    failAtomic([&](size_t source, uint32_t at, Cpu::Vertex& out) {
      const bool result = vertex(source, at, out);
      if (source == 1 && at == 0) {
        out.normal[0] = out.normal[1] = out.normal[2] = 0;
      }
      return result;
    }, index);
    failAtomic([&](size_t source, uint32_t at, Cpu::Vertex& out) {
      const bool result = vertex(source, at, out);
      if (source == 0 && at == 0) {
        out.texcoord[1] = std::numeric_limits<float>::infinity();
      }
      return result;
    }, index);

    const auto savedDescriptions = descriptions;
    const auto metadataRejected = [&]() {
      actual = sentinel;
      vertexReads = indexReads = 0;
      require(!Cpu::packRead(descriptions, {}, vertex, index, actual) && samePackedBytes(actual, sentinel) &&
              !vertexReads && !indexReads, "Invalid metadata reached readers or modified output");
      descriptions = savedDescriptions;
    };
    descriptions[0].transform = {-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    metadataRejected();
    descriptions[0].transform = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    metadataRejected();
    descriptions[0].transform[3] = 1; metadataRejected();
    descriptions[0].transform[12] = std::numeric_limits<float>::infinity(); metadataRejected();
    descriptions[0].vertexCount = UINT32_MAX; metadataRejected();
    descriptions[0].indexCount = 4; metadataRejected();
    descriptions[0].vertexCount = 0; metadataRejected();
    for (uint32_t mode = 0; mode < 3; ++mode) {
      Cpu::Limits limits;
      if (mode == 0) {
        limits.maxVertices = 11;
      }
      if (mode == 1) {
        limits.maxIndices = 11;
      }
      if (mode == 2) {
        limits.maxBytes = 12 * sizeof(Cpu::Vertex) + 12 * sizeof(uint32_t) - 1;
      }
      actual = sentinel;
      require(!Cpu::packRead(descriptions, limits, vertex, index, actual) && samePackedBytes(actual, sentinel),
              "Reader pack exceeded byte/count limits or changed its prior output");
    }
    uint32_t throws = 0, successes = 0;
    for (int64_t budget = 0; budget < 12; ++budget) {
      actual = sentinel;
      allocationBudget = budget;
      const bool ok = Cpu::packRead(descriptions, {}, vertex, index, actual);
      allocationBudget = -1;
      require(samePackedBytes(actual, ok ? expected : sentinel), "Real pack allocation failure changed prior output");
      throws += !ok;
      successes += ok;
    }
    require(throws >= 3 && successes, "Reader pack allocation injection missed a staging allocation or success");
    std::cout << "Reader pack: 192 exact-byte transform comparisons, strided formats/defaults, one-read counts and fault atomicity passed\n";
  }

  void packingTests() {
    std::vector<Cpu::Vertex> a(3);
    std::vector<Cpu::Vertex> b(4);
    for (auto& vertex : a) {
      vertex.position[0] = 1;
      vertex.normal[0] = 1;
      vertex.normal[1] = 1;
      vertex.texcoord[0] = 0.25f;
      vertex.color = 0x12345678u;
    }
    for (auto& vertex : b) {
      vertex.position[1] = 2;
      vertex.normal[2] = 1;
      vertex.texcoord[1] = 0.75f;
      vertex.color = 0xfedcba98u;
    }
    std::vector<uint32_t> ia {2, 0, 1};
    std::vector<uint32_t> ib {0, 1, 2, 2, 1, 3};
    Cpu::Source sa {a.data(), uint32_t(a.size()), ia.data(), uint32_t(ia.size())};
    Cpu::Source sb {b.data(), uint32_t(b.size()), ib.data(), uint32_t(ib.size())};
    sa.normals = sb.normals = true;
    sa.transform[0] = 2;
    sa.transform[5] = 3;
    sa.transform[12] = 10;
    // Positive determinant rotation changes the member's orientation without changing its index winding.
    sb.transform = {0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, -4, 7, 2, 1};
    Cpu::Data data;
    require(Cpu::pack({sa, sb}, {}, data), "Different-geometry world batch failed");
    require(data.vertices.size() == 7 && data.indices == std::vector<uint32_t>({2, 0, 1, 3, 4, 5, 5, 4, 6}),
            "Physical vertex append or source index rebasing failed");
    require(near(data.vertices[0].position[0], 12) && near(data.vertices[3].position[0], -6) &&
            near(data.vertices[3].position[1], 7) && near(data.vertices[3].position[2], 2),
            "Cell-local transforms were not baked into vertices");
    const float length = std::sqrt(0.25f + 1.f / 9.f);
    require(near(data.vertices[0].normal[0], 0.5f / length) && near(data.vertices[0].normal[1], (1.f / 3.f) / length),
            "Normals did not use the inverse transpose under nonuniform scale");
    require(data.vertices[0].color == a[0].color && data.vertices[3].color == b[0].color &&
            data.vertices[0].texcoord[0] == a[0].texcoord[0] && data.vertices[3].texcoord[1] == b[0].texcoord[1],
            "Source color or texture coordinates changed");
    require(data.ranges.size() == 2 && data.ranges[1].firstVertex == 3 && data.ranges[1].vertexCount == 4 &&
            data.ranges[1].firstIndex == 3 && data.ranges[1].indexCount == 6, "Member ownership ranges changed");

    const auto vertices = data.vertices;
    const auto allIndices = data.indices;
    std::vector<uint32_t> selected;
    require(Cpu::selectIndices(data.indices, data.ranges, {true, false}, selected) && selected == ia,
            "Hidden member's triangles remained in the selected topology");
    require(Cpu::selectIndices(data.indices, data.ranges, {false, true}, selected) &&
            selected == std::vector<uint32_t>({3, 4, 5, 5, 4, 6}), "Member visibility changed source index order");
    require(Cpu::selectIndices(data.indices, data.ranges, {false, false}, selected) && selected.empty(),
            "All-hidden batch did not request a hidden view");
    require(Cpu::selectIndices(data.indices, data.ranges, {true, true}, selected) && selected == data.indices,
            "Revealing members did not restore their complete topology");
    require(!std::memcmp(vertices.data(), data.vertices.data(), vertices.size() * sizeof(Cpu::Vertex)) &&
            allIndices == data.indices, "Visibility changed the immutable vertex/topology source");
    const auto preserved = selected;
    auto ranges = data.ranges;
    ranges[1].indexCount = UINT32_MAX;
    require(!Cpu::selectIndices(data.indices, ranges, {true, true}, selected) && selected == preserved,
            "Malformed index range was not rejected atomically");

    auto invalid = sa;
    invalid.transform[0] = -2;
    require(!Cpu::pack({invalid, sb}, {}, data), "Mirrored transform bypassed facing fallback");
    invalid = sa;
    invalid.transform[5] = 0;
    require(!Cpu::pack({invalid, sb}, {}, data), "Singular transform accepted");
    invalid = sa;
    invalid.transform[3] = 1;
    require(!Cpu::pack({invalid, sb}, {}, data), "Projective transform accepted");
    invalid = sa;
    invalid.transform[12] = std::numeric_limits<float>::infinity();
    require(!Cpu::pack({invalid, sb}, {}, data), "Nonfinite transform accepted");
    ib.back() = 4;
    require(!Cpu::pack({sa, sb}, {}, data), "Out-of-range index accepted");
    ib.back() = 3;
    Cpu::Limits limits;
    limits.maxVertices = 6;
    require(!Cpu::pack({sa, sb}, limits, data), "Vertex budget exceeded");
    limits = {};
    limits.maxBytes = 7 * sizeof(Cpu::Vertex) + 9 * sizeof(uint32_t) - 1;
    require(!Cpu::pack({sa, sb}, limits, data), "Byte budget exceeded");
    a[0].normal[0] = a[0].normal[1] = 0;
    require(!Cpu::pack({sa, sb}, {}, data), "Invalid authored normal accepted");
    require(data.indices == allIndices && !std::memcmp(data.vertices.data(), vertices.data(), vertices.size() * sizeof(Cpu::Vertex)),
            "Rejected pack changed the published output");
    sa.normals = false;
    require(Cpu::pack({sa, sb}, {}, data), "Normal-absent source was treated as an invalid authored normal");
  }

  void packedNormalTests() {
    float normal[3];
    Cpu::decodeOctNormal(0x80008000u, normal);
    require(near(normal[0], 0) && near(normal[1], 0) && near(normal[2], 1), "USD unsigned oct normal +Z failed");
    Cpu::decodeOctNormal(0xffffffffu, normal);
    require(near(normal[0], 0) && near(normal[1], 0) && near(normal[2], -1), "USD unsigned oct normal -Z failed");
    Cpu::decodeOctNormal(0x8000ffffu, normal);
    require(near(normal[0], 1) && near(normal[1], 0) && near(normal[2], 0), "USD unsigned oct normal +X failed");
  }

  void normalTransformTests() {
    std::vector<Cpu::Vertex> vertices(3);
    for (auto& vertex : vertices) {
      vertex.normal[0] = 1;
    }
    const uint32_t indices[] = {0, 1, 2};
    Cpu::Source a {vertices.data(), 3, indices, 3};
    Cpu::Source b = a;
    a.normals = b.normals = true;
    // A positive quarter-turn sends the +X normal to +Y, as it sends a +X position.
    a.transform = {0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    Cpu::Data data;
    require(Cpu::pack({a, b}, {}, data), "Rotated normal fixture failed");
    require(near(data.vertices[0].normal[0], 0) && near(data.vertices[0].normal[1], 1),
            "Normal rotated by the inverse instead of the inverse transpose");
    // A = rotationZ(90) * scale(2,3,4) * shearXByY(0.5).
    // A^-T * +X = (1/6, 1/2, 0), apart from normalization.
    a.transform = {0, 2, 0, 0, -3, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1};
    require(Cpu::pack({a, b}, {}, data), "Sheared normal fixture failed");
    const float inverseLength = 1.f / std::sqrt(1.f / 36.f + 0.25f);
    require(near(data.vertices[0].normal[0], (1.f / 6.f) * inverseLength) &&
            near(data.vertices[0].normal[1], 0.5f * inverseLength),
            "Rotated nonuniform scale/shear normal used the wrong cofactor orientation");
  }

  void emissionPolicyTests() {
    using dxvk::retained_batch_policy::hasNoEffectiveAuthoredEmission;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    require(hasNoEffectiveAuthoredEmission(true, 0.f) && hasNoEffectiveAuthoredEmission(true, -0.f),
            "An enabled zero-intensity legacy default was mistaken for effective emission");
    require(!hasNoEffectiveAuthoredEmission(true, 1.f) && !hasNoEffectiveAuthoredEmission(true, -1.f) &&
            !hasNoEffectiveAuthoredEmission(true, std::numeric_limits<float>::min()) &&
            !hasNoEffectiveAuthoredEmission(true, nan) && !hasNoEffectiveAuthoredEmission(true, infinity) &&
            !hasNoEffectiveAuthoredEmission(true, -infinity), "Effective or invalid enabled emission entered a static batch");
    require(hasNoEffectiveAuthoredEmission(false, 0.f) && hasNoEffectiveAuthoredEmission(false, 1.f) &&
            hasNoEffectiveAuthoredEmission(false, nan) && hasNoEffectiveAuthoredEmission(false, infinity),
            "Disabled emission disagreed with the shader's forced-zero behavior");
  }

  void budgetTests() {
    using Bucket = Cpu::BudgetBucket;
    Cpu::BudgetSelection selected;
    const std::vector<uint32_t> counts {1, 1, 1, 1};
    std::vector<Bucket> buckets {{{{0, 0}, {1, 0}}, 60, 0, 10}, {{{2, 0}, {3, 0}}, 60, 7, 30}};
    require(Cpu::selectBudget(counts, buckets, {100, 2}, selected), "Normal capacity rejection was treated as global failure");
    require(selected.inputs == std::vector<bool>({false, false, true, true}) &&
            selected.buckets == std::vector<bool>({false, true}) && selected.selectedBytes == 60 &&
            selected.projectedBytes == 120 && selected.projectedBatches == 2 && selected.selectedBatches == 1 &&
            selected.rejectedInputs == 2 && selected.rejectedBatches == 1 && selected.rejectedComponents == 1,
            "Existing cached component was not preserved within the byte budget");
    const auto stable = selected.inputs;
    std::reverse(buckets.begin(), buckets.end());
    require(Cpu::selectBudget(counts, buckets, {100, 2}, selected) && selected.inputs == stable,
            "Candidate bucket iteration changed the resident capacity choice");
    for (auto& bucket : buckets) {
      bucket.cachedId = 0;
    }
    require(Cpu::selectBudget(counts, buckets, {100, 2}, selected) &&
            selected.inputs == std::vector<bool>({true, true, false, false}), "New components were not selected by stable canonical identity");
    require(Cpu::selectBudget(counts, buckets, {120, 2}, selected) &&
            selected.inputs == std::vector<bool>({true, true, true, true}) && selected.selectedBytes == 120,
            "Exact byte/batch capacity incorrectly rejected a component");

    // The first input has two enhanced parts in separate materials. Both buckets and all partners are atomic.
    std::vector<Bucket> multipart {
      {{{0, 0}, {1, 0}}, 10, 1, 10}, {{{0, 1}, {2, 0}}, 10, 2, 11}, {{{3, 0}, {4, 0}}, 10, 0, 30}
    };
    require(Cpu::selectBudget({2, 1, 1, 1, 1}, multipart, {100, 1}, selected) &&
            selected.inputs == std::vector<bool>({false, false, false, true, true}) &&
            selected.buckets == std::vector<bool>({false, false, true}) && selected.rejectedComponents == 1,
            "Oversized multipart component blocked an independent batch or gained partial ownership");
    require(Cpu::selectBudget({2, 1, 1, 1, 1}, multipart, {20, 2}, selected) &&
            selected.inputs == std::vector<bool>({true, true, true, false, false}),
            "A cached multipart component did not retain complete ownership");
    require(Cpu::selectBudget(counts, buckets, {0, 0}, selected) && selected.selectedBatches == 0 &&
            selected.inputs == std::vector<bool>({false, false, false, false}), "Zero capacity was not a valid all-canonical plan");

    // The previous 257th-cohort failure must become a local, allocation-free planning rejection.
    std::vector<uint32_t> manyCounts(514, 1);
    std::vector<Bucket> many;
    for (uint32_t i = 0; i < 257; ++i) {
      many.push_back({{{i * 2, 0}, {i * 2 + 1, 0}}, 128, i == 256 ? 1ull : 0ull, uint64_t(i * 2 + 1)});
    }
    require(Cpu::selectBudget(manyCounts, many, {128 * 256, 256}, selected) && selected.selectedBatches == 256 &&
            selected.selectedBytes == 128 * 256 && selected.projectedBatches == 257 && selected.rejectedBatches == 1 &&
            selected.inputs[512] && selected.inputs[513] && !selected.inputs[510] && !selected.inputs[511],
            "257th cohort triggered a global fallback or displaced the cached resident component");

    const auto preservedInputs = selected.inputs;
    const auto preservedBuckets = selected.buckets;
    auto invalid = many;
    invalid[0].parts[0].input = 514;
    require(!Cpu::selectBudget(manyCounts, invalid, {128 * 256, 256}, selected) &&
            selected.inputs == preservedInputs && selected.buckets == preservedBuckets,
            "Invalid preflight changed its published selection");
    invalid = many;
    invalid[1].parts[0] = invalid[0].parts[0];
    require(!Cpu::selectBudget(manyCounts, invalid, {128 * 256, 256}, selected), "Duplicate source-part ownership entered preflight");
    multipart[0].bytes = UINT64_MAX;
    multipart[1].bytes = 1;
    require(Cpu::selectBudget({2, 1, 1, 1, 1}, multipart, {100, 8}, selected) &&
            selected.inputs == std::vector<bool>({false, false, false, true, true}) &&
            selected.projectedBytes == UINT64_MAX && selected.selectedBytes == 10,
            "Overflowing component cost wrapped through the capacity guard");
  }

  void visibilityTests() {
    struct Change { uint64_t id; bool visible; };
    const std::vector<std::vector<uint64_t>> ids {{11, 22, 11}, {11, 33}};
    Cpu::VisibilityIndex index;
    require(Cpu::buildVisibilityIndex(ids, index) && index.at(11).size() == 3,
            "A multipart child's locations were lost across plans or within one plan");
    const std::vector<std::vector<bool>> masks {{true, true, true}, {true, false}};
    uint32_t maskReads = 0;
    const auto getMask = [&](uint32_t batch) -> const std::vector<bool>& {
      ++maskReads;
      return masks.at(batch);
    };
    std::vector<Cpu::VisibilityPatch> patches;
    require(Cpu::visibilityPatches(index, 2, std::vector<Change>{{11, false}}, getMask, patches) &&
            patches.size() == 2 && patches[0].batch == 0 && patches[1].batch == 1 &&
            patches[0].visible == std::vector<bool>({false, true, false}) &&
            patches[1].visible == std::vector<bool>({false, false}),
            "A child change did not reach every one of its parts/plans atomically");
    require(masks[0] == std::vector<bool>({true, true, true}) && masks[1] == std::vector<bool>({true, false}),
            "Planning a visibility change mutated the source cache");
    const auto multi = patches;
    require(Cpu::visibilityPatches(index, 2, std::vector<Change>{{11, false}, {11, true}, {999, false}}, getMask, patches) &&
            patches.empty(), "Coalesced hide/show or an unowned child created an unnecessary index build");
    maskReads = 0;
    require(Cpu::visibilityPatches(index, 2, std::vector<Change>{{999, false}}, getMask, patches) &&
            patches.empty() && maskReads == 0, "An unowned child scanned batch masks");

    std::vector<std::pair<uint32_t, uint32_t>> publications;
    Cpu::visitVisibilityPublications(2, patches, false, [&](uint32_t batch, uint32_t patch) {
      publications.emplace_back(batch, patch);
    });
    require(publications.empty(), "A no-op visibility update republished unaffected plans");
    Cpu::visitVisibilityPublications(2, patches, true, [&](uint32_t batch, uint32_t patch) {
      publications.emplace_back(batch, patch);
    });
    require(publications == std::vector<std::pair<uint32_t, uint32_t>>({{0, UINT32_MAX}, {1, UINT32_MAX}}),
            "Resume failed to republish every unchanged view without marking its index state changed");
    publications.clear();
    Cpu::visitVisibilityPublications(3, std::vector<Cpu::VisibilityPatch>{{1, {false}}}, true,
      [&](uint32_t batch, uint32_t patch) { publications.emplace_back(batch, patch); });
    require(publications == std::vector<std::pair<uint32_t, uint32_t>>({{0, UINT32_MAX}, {1, 0}, {2, UINT32_MAX}}),
            "Republish-all did not distinguish a changed index plan from unchanged presentation views");

    struct Stage {
      uint32_t batch = 99;
      std::vector<uint32_t> indices;
    };
    const std::vector<std::vector<uint32_t>> topology {{0, 1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14}};
    const std::vector<std::vector<Cpu::Range>> ranges {
      {{0, 3, 0, 3}, {3, 3, 3, 3}, {6, 3, 6, 3}}, {{9, 3, 0, 3}, {12, 3, 3, 3}}
    };
    uint32_t attempts = 0;
    const auto build = [&](const Cpu::VisibilityPatch& patch, Stage& stage) {
      ++attempts;
      stage.batch = patch.batch;
      return Cpu::selectIndices(topology[patch.batch], ranges[patch.batch], patch.visible, stage.indices);
    };
    std::vector<Stage> staged(1);
    require(!Cpu::stageVisibilityPatches<Stage>(multi, [&](const Cpu::VisibilityPatch& patch, Stage& stage) {
      return patch.batch != 1 && build(patch, stage);
    }, staged) && staged.size() == 1 && staged[0].batch == 99,
            "A second-plan index failure published the first successful prefix");
    attempts = 0;
    require(Cpu::stageVisibilityPatches<Stage>(multi, build, staged) && attempts == 2 && staged.size() == 2 &&
            staged[0].indices == std::vector<uint32_t>({3, 4, 5}) && staged[1].indices.empty(),
            "Affected index selection leaked hidden parts or submitted an all-hidden plan's old indices");
    require(topology[0] == std::vector<uint32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8}),
            "Visibility changed the immutable all-member topology");
    require(Cpu::stageVisibilityPatches<Stage>({}, build, staged) && staged.empty() && attempts == 2,
            "No-op staging invoked an index builder");

    patches = {{77, {true}}};
    require(!Cpu::visibilityPatches(index, 2, std::vector<Change>{{0, false}}, getMask, patches) &&
            patches.size() == 1 && patches[0].batch == 77, "Malformed input changed the published mask output");
    auto corrupt = index;
    corrupt[11][0].batch = 2;
    require(!Cpu::visibilityPatches(corrupt, 2, std::vector<Change>{{11, false}}, getMask, patches),
            "Invalid membership batch location reached the getter");
    corrupt = index;
    corrupt[11][0].member = 3;
    require(!Cpu::visibilityPatches(corrupt, 2, std::vector<Change>{{11, false}}, getMask, patches),
            "Invalid membership member location escaped bounds checks");
    require(!Cpu::buildVisibilityIndex({{0}}, index) && index.at(11).size() == 3,
            "Failed structural indexing changed the previous ownership map");

    Cpu::MembershipRevision generation;
    require(generation.value() == 0, "A never-prepared membership advertised a usable revision");
    generation.publish();
    const uint64_t first = generation.value();
    require(first != 0, "A successful membership publish retained revision zero");
    generation.clear();
    require(generation.value() == 0, "Clear left a stale membership revision active");
    generation.publish();
    require(generation.value() != 0 && generation.value() != first,
            "Rebuilding after clear accepted an old membership generation");

    uint32_t allocationFailures = 0, successfulPlans = 0;
    for (int64_t budget = 0; budget < 48; ++budget) {
      std::vector<Cpu::VisibilityPatch> output {{77, {true}}};
      bool completed = false;
      try {
        allocationBudget = budget;
        completed = Cpu::visibilityPatches(index, 2, std::vector<Change>{{11, false}}, getMask, output);
      } catch (const std::bad_alloc&) {
        ++allocationFailures;
      }
      allocationBudget = -1;
      if (!completed) {
        require(output.size() == 1 && output[0].batch == 77 && output[0].visible == std::vector<bool>({true}),
                "A real mask-planning allocation failure changed the caller's output");
      } else {
        ++successfulPlans;
        require(output.size() == 2, "An allocation-budget success lost one affected plan");
      }
    }
    require(allocationFailures > 0 && successfulPlans > 0, "Allocation injection did not cover failed and successful planners");
    for (int64_t budget = 0; budget < 24; ++budget) {
      std::vector<Stage> output(1);
      bool completed = false;
      try {
        allocationBudget = budget;
        completed = Cpu::stageVisibilityPatches<Stage>(multi, build, output);
      } catch (const std::bad_alloc&) {
        ++allocationFailures;
      }
      allocationBudget = -1;
      if (!completed) {
        require(output.size() == 1 && output[0].batch == 99 && output[0].indices.empty(),
                "A real multi-plan staging allocation failure published a partial prefix");
      } else {
        require(output.size() == 2 && output[1].indices.empty(), "A successful hidden plan reused old topology");
      }
    }
  }

  void detachmentTests() {
    struct Change { uint64_t id; bool visible; };
    const std::vector<std::vector<uint64_t>> ids {{11, 22, 11}, {11, 33}};
    Cpu::VisibilityIndex index;
    require(Cpu::buildVisibilityIndex(ids, index), "Detachment fixture membership failed");
    std::vector<std::vector<bool>> masks {{true, true, true}, {true, false}};
    const auto getMask = [&](uint32_t batch) -> const std::vector<bool>& { return masks.at(batch); };
    const std::vector<uint64_t> detach {11, 11, 999};
    const std::vector<Change> visibility {{11, true}, {22, false}, {33, true}};
    std::vector<Change> changes;
    require(Cpu::detachmentChanges(detach, visibility, changes) && changes.size() == 6 &&
            !changes.back().visible, "Combined detachment did not follow the final visibility changes");
    std::vector<Cpu::VisibilityPatch> patches;
    require(Cpu::visibilityPatches(index, 2, changes, getMask, patches) && patches.size() == 2 &&
            patches[0].visible == std::vector<bool>({false, false, false}) &&
            patches[1].visible == std::vector<bool>({false, true}),
            "Detachment failed to override show for every part of a reused child handle");

    struct Stage { uint32_t batch = 99; std::vector<uint32_t> indices; };
    const std::vector<std::vector<uint32_t>> topology {{0, 1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14}};
    const std::vector<std::vector<Cpu::Range>> ranges {
      {{0, 3, 0, 3}, {3, 3, 3, 3}, {6, 3, 6, 3}}, {{9, 3, 0, 3}, {12, 3, 3, 3}}
    };
    const auto build = [&](const Cpu::VisibilityPatch& patch, Stage& stage) {
      stage.batch = patch.batch;
      return Cpu::selectIndices(topology[patch.batch], ranges[patch.batch], patch.visible, stage.indices);
    };
    std::vector<Stage> staged(1);
    require(!Cpu::stageVisibilityPatches<Stage>(patches, [&](const Cpu::VisibilityPatch& patch, Stage& stage) {
      return patch.batch != 1 && build(patch, stage);
    }, staged) && staged.size() == 1 && staged[0].batch == 99 && index.count(11) &&
            masks[0] == std::vector<bool>({true, true, true}),
            "Failed detachment published an index prefix or removed ownership before success");
    require(Cpu::stageVisibilityPatches<Stage>(patches, build, staged) && staged.size() == 2 &&
            staged[0].indices.empty() && staged[1].indices == std::vector<uint32_t>({12, 13, 14}),
            "Detachment submitted a removed part or an all-hidden batch's old triangles");
    for (auto& patch : patches) {
      masks[patch.batch].swap(patch.visible);
    }
    allocationBudget = 0;
    const size_t forgotten = Cpu::forgetDetachedChildren(index, detach);
    allocationBudget = -1;
    require(forgotten == 1 && !index.count(11) && index.count(22) && index.count(33),
            "Post-commit detachment allocated, lost stable ownership, or mishandled duplicate/unknown IDs");
    require(Cpu::visibilityPatches(index, 2, std::vector<Change>{{11, true}}, getMask, patches) && patches.empty(),
            "A later visible update revived the old geometry of a reused canonical child handle");
    require(Cpu::visibilityPatches(index, 2, std::vector<Change>{{22, true}}, getMask, patches) &&
            patches.size() == 1 && patches[0].batch == 0 && patches[0].visible == std::vector<bool>({false, true, false}),
            "Detachment prevented stable siblings from updating or revived a detached range");
    require(topology[0] == std::vector<uint32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8}) && ranges[0].size() == 3,
            "Detachment repacked the immutable source topology or member ranges");
    Cpu::VisibilityIndex alreadyHidden;
    require(Cpu::buildVisibilityIndex({{44, 55}}, alreadyHidden), "Hidden detachment fixture failed");
    const std::vector<bool> hiddenMask {false, true};
    std::vector<Change> hiddenChanges;
    const std::vector<uint64_t> hiddenChild {44};
    require(Cpu::detachmentChanges(hiddenChild, hiddenChanges) &&
            Cpu::visibilityPatches(alreadyHidden, 1, hiddenChanges,
              [&](uint32_t) -> const std::vector<bool>& { return hiddenMask; }, patches) && patches.empty() &&
            Cpu::forgetDetachedChildren(alreadyHidden, hiddenChild) == 1 && !alreadyHidden.count(44),
            "An already-hidden child needed an index rebuild or retained revivable membership");

    changes = {{77, true}};
    require(!Cpu::detachmentChanges(std::vector<uint64_t>{0}, changes) && changes.size() == 1 && changes[0].id == 77,
            "Malformed detachment changed the transaction input output");
    uint32_t allocationFailures = 0, successes = 0;
    for (int64_t budget = 0; budget < 8; ++budget) {
      std::vector<Change> output {{77, true}};
      bool completed = false;
      try {
        allocationBudget = budget;
        completed = Cpu::detachmentChanges(detach, visibility, output);
      } catch (const std::bad_alloc&) {
        ++allocationFailures;
      }
      allocationBudget = -1;
      if (!completed) {
        require(output.size() == 1 && output[0].id == 77 && output[0].visible,
                "A real combined-input allocation failure changed published input");
      } else {
        ++successes;
        require(output.size() == 6 && !output.back().visible, "Successful combined input lost forced detachment");
      }
    }
    require(allocationFailures && successes, "Detachment allocation injection missed failure or success");
    allocationFailures = successes = 0;
    for (int64_t budget = 0; budget < 32; ++budget) {
      auto published = index;
      bool completed = false;
      try {
        allocationBudget = budget;
        completed = Cpu::buildVisibilityIndex(ids, published);
      } catch (const std::bad_alloc&) {
        ++allocationFailures;
      }
      allocationBudget = -1;
      if (!completed) {
        require(!published.count(11) && published.count(22) && published.count(33),
                "Failed structural readmission resurrected a detached child");
      } else {
        ++successes;
        require(published.at(11).size() == 3, "Successful structural readmission lost one of the child's parts");
      }
    }
    require(allocationFailures && successes, "Readmission allocation injection missed failure or success");
  }

  void indexReuseTests() {
    using Action = Cpu::VisibilityGeometryAction;
    const std::vector<bool> last {true, false, true};
    const std::vector<bool> hidden {false, false, false};
    const std::vector<bool> different {false, true, true};
    require(Cpu::visibilityGeometryAction(hidden, last, true) == Action::Hide &&
            Cpu::visibilityGeometryRevision(17, Action::Hide) == 17,
            "Hiding discarded the last topology or changed its geometry revision");
    require(Cpu::visibilityGeometryAction(last, last, true) == Action::Reuse &&
            Cpu::visibilityGeometryRevision(17, Action::Reuse) == 17,
            "An exact-mask reveal requested another index state or geometry revision");
    require(Cpu::visibilityGeometryAction(different, last, true) == Action::Build &&
            Cpu::visibilityGeometryRevision(17, Action::Build) == 18,
            "A different partial mask reused triangles from the old topology");
    require(Cpu::visibilityGeometryAction(last, last, false) == Action::Build &&
            Cpu::visibilityGeometryAction(last, {}, true) == Action::Build,
            "A clear/new source or a missing rendering state reused an unavailable index buffer");
    const uint64_t firstHidden = Cpu::visibilityGeometryRevision(0, Action::Hide);
    require(firstHidden != 0 && Cpu::visibilityGeometryRevision(firstHidden, Action::Build) != firstHidden,
            "Initially hidden plans violated the nonzero token or first-build contract");

    std::vector<bool> remembered = last;
    allocationBudget = 0;
    const auto reveal = Cpu::visibilityGeometryAction(last, remembered, true);
    Cpu::stageLastNonemptyMask(Action::Hide, hidden, remembered);
    Cpu::stageLastNonemptyMask(reveal, last, remembered);
    allocationBudget = -1;
    require(reveal == Action::Reuse && remembered == last,
            "Hide/reveal decision or mask staging allocated or changed the last nonempty mask");
    Cpu::stageLastNonemptyMask(Action::Build, different, remembered);
    require(remembered == different &&
            Cpu::visibilityGeometryAction(last, remembered, true) == Action::Build,
            "New topology did not replace the remembered exact mask");

    // Erased child 11 occupied slots zero and two. Its old all-true mask must not revive those triangles.
    Cpu::VisibilityIndex index;
    require(Cpu::buildVisibilityIndex({{11, 22, 11}}, index), "Reuse/detachment fixture failed");
    const std::vector<uint64_t> retired {11};
    require(Cpu::forgetDetachedChildren(index, retired) == 1, "Reuse fixture did not detach the old child");
    struct Change { uint64_t id; bool visible; };
    const std::vector<bool> current {false, false, false};
    std::vector<Cpu::VisibilityPatch> patches;
    require(Cpu::visibilityPatches(index, 1, std::vector<Change>{{11, true}, {22, true}},
      [&](uint32_t) -> const std::vector<bool>& { return current; }, patches) && patches.size() == 1 &&
      patches[0].visible == std::vector<bool>({false, true, false}) &&
      Cpu::visibilityGeometryAction(patches[0].visible, {true, true, true}, true) == Action::Build,
            "Exact-mask reuse revived a detached child from the old nonempty rendering state");
    require(Cpu::visibilityGeometryAction(patches[0].visible, {false, true, false}, true) == Action::Reuse,
            "A retained topology that already excludes detached children failed its exact-mask proof");

    uint32_t allocationFailures = 0, successes = 0;
    for (int64_t budget = 0; budget < 4; ++budget) {
      std::vector<bool> output = last;
      bool completed = false;
      try {
        allocationBudget = budget;
        Cpu::stageLastNonemptyMask(Action::Build, different, output);
        completed = true;
      } catch (const std::bad_alloc&) {
        ++allocationFailures;
      }
      allocationBudget = -1;
      require(output == (completed ? different : last),
              "Index-state mask staging allocation failure changed a published cache mask");
      successes += completed;
    }
    require(allocationFailures && successes, "Last-mask allocation fixture missed failure or success");

    struct Stage {
      uint32_t batch = 99;
      uint64_t revision = 17;
      Cpu::VisibilityGeometryAction action = Cpu::VisibilityGeometryAction::Hide;
      std::vector<bool> remembered;
    };
    const std::vector<std::vector<bool>> oldMasks {{true, true}, {true, true}};
    const std::vector<Cpu::VisibilityPatch> two {{0, {true, false}}, {1, {false, false}}};
    const auto build = [&](const Cpu::VisibilityPatch& patch, Stage& stage) {
      stage.batch = patch.batch;
      stage.action = Cpu::visibilityGeometryAction(patch.visible, oldMasks[patch.batch], true);
      stage.revision = Cpu::visibilityGeometryRevision(stage.revision, stage.action);
      Cpu::stageLastNonemptyMask(stage.action, patch.visible, stage.remembered);
      return true;
    };
    std::vector<Stage> staged(1);
    require(!Cpu::stageVisibilityPatches<Stage>(two, [&](const Cpu::VisibilityPatch& patch, Stage& stage) {
      return build(patch, stage) && patch.batch != 1;
    }, staged) && staged.size() == 1 && staged[0].batch == 99 &&
            oldMasks[0] == std::vector<bool>({true, true}),
            "A second-plan failure published the new remembered mask or revision from a prefix");
    require(Cpu::stageVisibilityPatches<Stage>(two, build, staged) && staged.size() == 2 &&
            staged[0].action == Action::Build && staged[0].revision == 18 &&
            staged[0].remembered == std::vector<bool>({true, false}) &&
            staged[1].action == Action::Hide && staged[1].revision == 17 && staged[1].remembered.empty(),
            "Shared transaction staging mixed a visibility-only hide with a geometry rebuild");
  }
  void fixedChunkTests() {
    const std::vector<std::vector<Cpu::FixedSource>> sources {
      {{10, 0}, {20, 0}, {30, 0}}, {{10, 1}, {40, 0}}
    };
    Cpu::VisibilityIndex index;
    require(Cpu::buildVisibilityIndex({{10, 20, 30}, {10, 40}}, index), "Fixed chunk fixture index failed");
    const std::vector<Cpu::FixedInput> inputs {{10, 2}, {20, 1}, {30, 1}, {40, 1}, {50, 1}, {60, 1}};
    const auto equal = [](uint32_t, uint32_t, uint32_t) { return true; };
    Cpu::FixedSelection fixed;
    require(Cpu::retainFixedChunks(inputs, sources, index, {}, equal, fixed) &&
            fixed.inputs == std::vector<bool>({true, true, true, true, false, false}) &&
            fixed.slots == std::vector<std::vector<bool>>({{true, true, true}, {true, true}}) &&
            fixed.fixedChunks == 2 && fixed.residentParts == 5 && fixed.activeParts == 5,
            "Appending compatible new children changed old whole-child ownership");

    const std::vector<Cpu::FixedInput> removed {{10, 2}, {30, 1}, {40, 1}, {50, 1}};
    require(Cpu::retainFixedChunks(removed, sources, index, {}, equal, fixed) &&
            fixed.slots == std::vector<std::vector<bool>>({{true, false, true}, {true, true}}) &&
            fixed.fixedChunks == 2 && fixed.residentParts == 5 && fixed.activeParts == 4,
            "Removing a middle slot compacted its surviving physical ranges or stopped charging a hole");
    std::vector<Cpu::FixedSource> active;
    require(Cpu::filterPhysicalMembers(sources[0], fixed.slots[0], active) && active.size() == 2 &&
            active[0].id == 10 && active[1].id == 30 && sources[0][1].id == 20,
            "Filtered output mutated the immutable physical source sidecar");
    Cpu::VisibilityIndex remapped;
    require(Cpu::buildPhysicalVisibilityIndex({3, 2}, {{{10, 0}, {30, 2}}, {{10, 0}, {40, 1}}}, remapped) &&
            remapped.at(30)[0].member == 2 && remapped.at(10).size() == 2,
            "Filtered ownership used its dense ordinal as a physical visibility slot");
    std::vector<Cpu::VisibilityPatch> patches;
    const std::vector<std::vector<bool>> visible {{true, false, true}, {true, true}};
    struct Change { uint64_t id; bool visible; };
    require(Cpu::visibilityPatches(remapped, 2, std::vector<Change>{{30, false}},
        [&](uint32_t batch) -> const std::vector<bool>& { return visible[batch]; }, patches) &&
            patches.size() == 1 && patches[0].visible == std::vector<bool>({true, false, false}),
            "A surviving child toggled a retired neighbor's physical triangles");

    require(Cpu::retainFixedChunks(inputs, sources, index, {},
        [](uint32_t, uint32_t batch, uint32_t slot) { return batch != 1 || slot != 0; }, fixed) &&
            !fixed.inputs[0] && !fixed.slots[0][0] && !fixed.slots[1][0] &&
            fixed.inputs[1] && fixed.inputs[2] && fixed.inputs[3],
            "A changed enhanced part retained another part of the same child in a different chunk");
    require(Cpu::retainFixedChunks(inputs, sources, index, {20, 20}, equal, fixed) && !fixed.inputs[1] &&
            !fixed.slots[0][1] && fixed.inputs[0],
            "A same-source pending detach was retained by a due full preparation");
    auto incomplete = index;
    incomplete.at(10).pop_back();
    require(Cpu::retainFixedChunks(inputs, sources, incomplete, {}, equal, fixed) && !fixed.inputs[0] &&
            !fixed.slots[0][0] && !fixed.slots[1][0],
            "Incomplete old multipart coverage seeded partial fixed ownership");
    auto duplicatePart = sources;
    duplicatePart[1][0].part = 0;
    require(Cpu::retainFixedChunks(inputs, duplicatePart, index, {}, equal, fixed) && !fixed.inputs[0],
            "Duplicate old part ordinals replaced a missing enhanced part");

    auto reusedSources = sources;
    reusedSources.push_back({{20, 0}, {50, 0}});
    Cpu::VisibilityIndex reusedIndex;
    require(Cpu::buildPhysicalVisibilityIndex({3, 2, 2},
        {{{10, 0}, {30, 2}}, {{10, 0}, {40, 1}}, {{20, 0}, {50, 1}}}, reusedIndex),
            "Reused-handle fixture failed explicit physical indexing");
    uint32_t oldDeadVisits = 0;
    require(Cpu::retainFixedChunks(inputs, reusedSources, reusedIndex, {},
        [&](uint32_t, uint32_t batch, uint32_t slot) {
          oldDeadVisits += batch == 0 && slot == 1;
          return true;
        }, fixed) && !oldDeadVisits && !fixed.slots[0][1] && fixed.slots[2][0] &&
            fixed.residentParts == 7 && fixed.activeParts == 6,
            "The presence of a reused ID revived a permanently detached old physical slot");
    require(Cpu::retainFixedChunks(inputs, reusedSources, reusedIndex, {20}, equal, fixed) &&
            !fixed.slots[0][1] && !fixed.slots[2][0],
            "Pending detach removed only one generation of a reused child ID");

    // A completely dead chunk releases capacity. A partially dead chunk retains all of its slots.
    require(Cpu::retainFixedChunks({{30, 1}}, sources, index, {}, equal, fixed) &&
            fixed.fixedChunks == 1 && fixed.residentParts == 3 && fixed.activeParts == 1 &&
            fixed.slots[1] == std::vector<bool>({false, false}),
            "A dead chunk or resident hole was counted as active/available storage");
    Cpu::BudgetSelection selected;
    const std::vector<Cpu::BudgetBucket> delta {{{{0, 0}, {1, 0}}, 64, 0, 50}};
    require(Cpu::selectBudget({1, 1}, delta, {128, 2, 1}, selected) &&
            selected.inputs == std::vector<bool>({false, false}) && !selected.selectedParts,
            "Physical part capacity admitted a partial delta component");
    require(Cpu::selectBudget({1, 1}, delta, {128, 2, 2}, selected) && selected.selectedParts == 2 &&
            selected.inputs == std::vector<bool>({true, true}),
            "A complete new delta child component was rejected within residual capacity");
    require(Cpu::selectBudget({1, 1}, delta, {128 * 1024 * 1024, 256, 0}, selected) &&
            !selected.selectedBatches && !selected.selectedParts,
            "Resident holes occupying all 65536 physical slots allowed speculative delta allocation");
    require(Cpu::selectBudget({1, 1}, delta, {0, 256, 65536}, selected) && !selected.selectedBatches,
            "Full retained byte charge was ignored by delta admission");
    require(Cpu::selectBudget({1, 1}, delta, {128 * 1024 * 1024, 0, 65536}, selected) && !selected.selectedBatches,
            "A fixed 256-chunk cache admitted more chunks");
    const std::vector<Cpu::BudgetBucket> multiDelta {
      {{{1, 0}, {2, 0}}, 32, 0, 50}, {{{1, 1}, {3, 0}}, 32, 0, 50}
    };
    require(Cpu::selectBudget({0, 2, 1, 1}, multiDelta, {128, 2, 3}, selected) && !selected.selectedParts &&
            selected.inputs == std::vector<bool>({false, false, false, false}),
            "A retained input placeholder or part capacity split a new multipart child");
    require(Cpu::selectBudget({0, 2, 1, 1}, multiDelta, {128, 2, 4}, selected) && selected.selectedParts == 4 &&
            selected.inputs == std::vector<bool>({false, true, true, true}),
            "Delta admission required already retained children to appear in its new buckets");

    Cpu::VisibilityIndex shifted;
    require(Cpu::buildPhysicalVisibilityIndex({2}, {{{10, 0}, {40, 1}}}, shifted) &&
            shifted.at(40)[0].batch == 0 && shifted.at(40)[0].member == 1,
            "Retiring an entire preceding chunk left stale next-generation batch indices");
    const auto indexEqual = [](const Cpu::VisibilityIndex& a, const Cpu::VisibilityIndex& b) {
      if (a.size() != b.size()) {
        return false;
      }
      for (const auto& item : a) {
        const auto found = b.find(item.first);
        if (found == b.end() || found->second.size() != item.second.size()) {
          return false;
        }
        for (size_t i = 0; i < item.second.size(); ++i) {
          if (found->second[i].batch != item.second[i].batch ||
              found->second[i].member != item.second[i].member) {
            return false;
          }
        }
      }
      return true;
    };
    uint32_t proofThrows = 0, proofSuccesses = 0, indexThrows = 0, indexSuccesses = 0;
    const std::vector<uint64_t> pendingDetach {20};
    const std::vector<uint32_t> physicalCounts {3, 2};
    const std::vector<std::vector<Cpu::IndexedMember>> physicalMembers {
      {{10, 0}, {30, 2}}, {{10, 0}, {40, 1}}
    };
    for (int64_t budget = 0; budget < 80; ++budget) {
      Cpu::FixedSelection result;
      result.inputs = {true};
      result.slots = {{true}};
      bool completed = false;
      try {
        allocationBudget = budget;
        completed = Cpu::retainFixedChunks(inputs, sources, index, pendingDetach, equal, result);
      } catch (const std::bad_alloc&) {
        ++proofThrows;
      }
      allocationBudget = -1;
      require(completed ? result.inputs.size() == inputs.size() && !result.slots[0][1] :
              result.inputs == std::vector<bool>({true}) && result.slots == std::vector<std::vector<bool>>({{true}}),
              "A real allocation failure published partial fixed ownership");
      proofSuccesses += completed;
      Cpu::VisibilityIndex output = shifted;
      completed = false;
      try {
        allocationBudget = budget;
        completed = Cpu::buildPhysicalVisibilityIndex(physicalCounts, physicalMembers, output);
      } catch (const std::bad_alloc&) {
        ++indexThrows;
      }
      allocationBudget = -1;
      require(completed ? indexEqual(output, remapped) : indexEqual(output, shifted),
              "A real allocation failure changed the published physical location index");
      indexSuccesses += completed;
    }
    require(proofThrows && proofSuccesses && indexThrows && indexSuccesses,
            "Fixed proof/index allocation injection missed actual failure or successful publication");
    uint32_t filterThrows = 0, filterSuccesses = 0;
    const std::vector<bool> filteredMask {true, false, true};
    for (int64_t budget = 0; budget < 4; ++budget) {
      std::vector<Cpu::FixedSource> output {{999, 0}};
      bool completed = false;
      try {
        allocationBudget = budget;
        completed = Cpu::filterPhysicalMembers(sources[0], filteredMask, output);
      } catch (const std::bad_alloc&) {
        ++filterThrows;
      }
      allocationBudget = -1;
      require(completed ? output.size() == 2 && output[0].id == 10 && output[1].id == 30 :
              output.size() == 1 && output[0].id == 999,
              "A real active-members allocation failure changed the published output sidecar");
      filterSuccesses += completed;
    }
    require(filterThrows && filterSuccesses, "Active-members fixture missed actual allocation failure or success");
    require(!Cpu::buildPhysicalVisibilityIndex({3}, {{{10, 2}, {20, 2}}}, shifted) && shifted.size() == 2,
            "Duplicate physical slots corrupted a published index");
    Cpu::FixedSelection sentinel;
    sentinel.inputs = {true};
    require(!Cpu::retainFixedChunks(inputs, sources, index, {0}, equal, sentinel) &&
            sentinel.inputs == std::vector<bool>({true}), "Invalid pending detach changed fixed output");

    struct FixedStage { uint32_t batch = 99; std::vector<bool> slots; };
    const std::vector<Cpu::VisibilityPatch> fixedPatches {{0, {true, false, true}}, {1, {true, true}}};
    std::vector<FixedStage> stages(1);
    uint32_t builds = 0;
    require(!Cpu::stageVisibilityPatches<FixedStage>(fixedPatches,
        [&](const Cpu::VisibilityPatch& patch, FixedStage& stage) {
          stage.batch = patch.batch;
          stage.slots = patch.visible;
          return ++builds != 2;
        }, stages) && builds == 2 && stages.size() == 1 && stages[0].batch == 99 &&
            index.at(20)[0].member == 1,
            "A second fixed-chunk build failure published its first prefix or moved old ownership");
    uint32_t prefixThrows = 0, stageSuccesses = 0;
    for (int64_t budget = 0; budget < 16; ++budget) {
      std::vector<FixedStage> output(1);
      bool completed = false;
      builds = 0;
      try {
        allocationBudget = budget;
        completed = Cpu::stageVisibilityPatches<FixedStage>(fixedPatches,
          [&](const Cpu::VisibilityPatch& patch, FixedStage& stage) {
            ++builds;
            stage.batch = patch.batch;
            stage.slots = patch.visible;
            return true;
          }, output);
      } catch (const std::bad_alloc&) {
        prefixThrows += builds >= 2;
      }
      allocationBudget = -1;
      require(completed ? output.size() == 2 && output[0].batch == 0 && output[1].batch == 1 :
              output.size() == 1 && output[0].batch == 99,
              "A real staged-prefix allocation failure changed the old fixed-chunk publication");
      stageSuccesses += completed;
    }
    require(prefixThrows && stageSuccesses, "Fixed-chunk staging did not exercise a real second-prefix failure");
    Cpu::MembershipRevision generation;
    generation.publish();
    const uint64_t before = generation.value();
    generation.clear();
    require(!generation.value(), "Clear retained the old fixed membership generation");
    require(Cpu::retainFixedChunks(inputs, {}, {}, {}, equal, fixed) && !fixed.fixedChunks &&
            fixed.inputs == std::vector<bool>(inputs.size(), false),
            "A full source reset kept a previously admitted slot");
    generation.publish();
    require(generation.value() && generation.value() != before, "Full admission reused a cleared generation token");
  }
}

int main() {
  try {
    coverageTests();
    coverageDifferentialTests();
    readerPackingTests();
    packingTests();
    packedNormalTests();
    normalTransformTests();
    emissionPolicyTests();
    budgetTests();
    visibilityTests();
    detachmentTests();
    indexReuseTests();
    fixedChunkTests();
    std::cout << "Retained world batch CPU tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
