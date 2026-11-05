/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dxvk {

class RetainedWorldBatchCpu {
public:
  struct Vertex {
    float position[3] {};
    float normal[3] {};
    float texcoord[2] {};
    uint32_t color = 0xffffffffu;
  };
  static_assert(sizeof(Vertex) == 36);

  struct Source {
    const Vertex* vertices = nullptr;
    uint32_t vertexCount = 0;
    const uint32_t* indices = nullptr;
    uint32_t indexCount = 0;
    // Column-major affine source-to-cell transform.
    std::array<float, 16> transform {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    bool normals = false;
  };
  // Metadata only: payload readers are borrowed for the duration of packRead.
  struct ReadSource {
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    std::array<float, 16> transform {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    bool normals = false;
  };
  struct Range {
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
  };
  struct Data {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Range> ranges;
  };
  struct Limits {
    uint32_t maxVertices = 131072;
    uint32_t maxIndices = 393216;
    size_t maxBytes = 8 * 1024 * 1024;
  };
  struct PartRef {
    uint32_t input = 0;
    uint32_t part = 0;
  };
  struct BudgetBucket {
    std::vector<PartRef> parts;
    uint64_t bytes = 0;
    uint64_t cachedId = 0;
    uint64_t stableKey = 0;
  };
  struct BudgetLimits {
    uint64_t maxBytes = 128 * 1024 * 1024;
    uint32_t maxBatches = 256;
    uint32_t maxParts = 65536;
  };
  struct BudgetSelection {
    std::vector<bool> inputs;
    std::vector<bool> buckets;
    uint64_t projectedBytes = 0;
    uint64_t projectedBatches = 0;
    uint64_t selectedBytes = 0;
    uint64_t selectedBatches = 0;
    uint64_t projectedParts = 0;
    uint64_t selectedParts = 0;
    uint64_t rejectedInputs = 0;
    uint64_t rejectedBatches = 0;
    uint64_t rejectedComponents = 0;
  };

  struct VisibilityLocation {
    uint32_t batch = 0;
    uint32_t member = 0;
  };
  using VisibilityIndex = std::unordered_map<uint64_t, std::vector<VisibilityLocation>>;
  struct FixedSource {
    uint64_t id = 0;
    uint32_t part = 0;
  };
  struct FixedInput {
    uint64_t id = 0;
    uint32_t partCount = 0;
  };
  struct FixedSelection {
    std::vector<bool> inputs;
    // These masks use immutable physical source slots, including retired holes.
    std::vector<std::vector<bool>> slots;
    uint64_t fixedChunks = 0;
    uint64_t residentParts = 0;
    uint64_t activeParts = 0;
  };
  struct IndexedMember {
    uint64_t id = 0;
    uint32_t slot = 0;
  };

  template<typename Member>
  static bool filterPhysicalMembers(const std::vector<Member>& physical, const std::vector<bool>& active,
                                    std::vector<Member>& output) {
    if (physical.size() != active.size()) {
      return false;
    }
    std::vector<Member> next;
    next.reserve(size_t(std::count(active.begin(), active.end(), true)));
    for (size_t slot = 0; slot < physical.size(); ++slot) {
      if (active[slot]) {
        next.push_back(physical[slot]);
      }
    }
    output.swap(next);
    return true;
  }
  struct VisibilityPatch {
    uint32_t batch = 0;
    std::vector<bool> visible;
  };
  enum class VisibilityGeometryAction { Hide, Reuse, Build };

  // Exact masks prove topology equality without relying on a hash or comparing GPU buffers.
  static VisibilityGeometryAction visibilityGeometryAction(const std::vector<bool>& visible,
      const std::vector<bool>& lastNonemptyMask, bool hasNonemptyRenderingState) noexcept {
    if (std::find(visible.begin(), visible.end(), true) == visible.end()) {
      return VisibilityGeometryAction::Hide;
    }
    if (hasNonemptyRenderingState && visible == lastNonemptyMask) {
      return VisibilityGeometryAction::Reuse;
    }
    return VisibilityGeometryAction::Build;
  }

  // Revision describes geometry, not presentation. A never-rendered hidden plan still needs a valid token.
  static uint64_t visibilityGeometryRevision(uint64_t previous, VisibilityGeometryAction action) noexcept {
    if (action != VisibilityGeometryAction::Build) {
      return previous ? previous : 1;
    }
    const uint64_t next = previous + 1;
    return next ? next : 1;
  }

  // A staged build changes the remembered mask only after its index state is ready.
  // Hide/reuse preserve it, and an allocation failure preserves the caller's staged output.
  static void stageLastNonemptyMask(VisibilityGeometryAction action, const std::vector<bool>& visible,
                                    std::vector<bool>& output) {
    if (action == VisibilityGeometryAction::Build) {
      std::vector<bool> next = visible;
      output.swap(next);
    }
  }

  class MembershipRevision {
  public:
    uint64_t value() const { return m_ready ? m_serial : 0; }
    void publish() noexcept { advance(); m_ready = true; }
    void clear() noexcept { advance(); m_ready = false; }
  private:
    void advance() noexcept {
      if (++m_serial == 0) {
        ++m_serial;
      }
    }
    uint64_t m_serial = 0;
    bool m_ready = false;
  };

  // Structural preparation proves ownership; this index only records all locations of each accepted child.
  static bool buildVisibilityIndex(const std::vector<std::vector<uint64_t>>& memberIds, VisibilityIndex& output) {
    if (memberIds.size() > UINT32_MAX) {
      return false;
    }
    VisibilityIndex next;
    for (size_t batch = 0; batch < memberIds.size(); ++batch) {
      if (memberIds[batch].empty() || memberIds[batch].size() > UINT32_MAX) {
        return false;
      }
      for (size_t member = 0; member < memberIds[batch].size(); ++member) {
        const uint64_t id = memberIds[batch][member];
        if (!id) {
          return false;
        }
        next[id].push_back({uint32_t(batch), uint32_t(member)});
      }
    }
    output.swap(next);
    return true;
  }

  // Output ownership can omit holes. Visibility locations must still name the original physical slots.
  static bool buildPhysicalVisibilityIndex(const std::vector<uint32_t>& physicalCounts,
      const std::vector<std::vector<IndexedMember>>& members, VisibilityIndex& output) {
    if (physicalCounts.size() != members.size() || members.size() > UINT32_MAX) {
      return false;
    }
    VisibilityIndex next;
    for (size_t batch = 0; batch < members.size(); ++batch) {
      uint32_t previous = UINT32_MAX;
      if (members[batch].empty()) {
        return false;
      }
      for (const IndexedMember& member : members[batch]) {
        if (!member.id || member.slot >= physicalCounts[batch] ||
            (previous != UINT32_MAX && member.slot <= previous)) {
          return false;
        }
        previous = member.slot;
        next[member.id].push_back({uint32_t(batch), member.slot});
      }
    }
    output.swap(next);
    return true;
  }

  // Every old location of a child must match its entire current description before any slot is retained.
  // The callback compares the real source state; tests exercise this same ownership/coverage proof.
  template<typename Matcher>
  static bool retainFixedChunks(const std::vector<FixedInput>& inputs,
      const std::vector<std::vector<FixedSource>>& sources, const VisibilityIndex& index,
      const std::vector<uint64_t>& pendingDetachIds, Matcher&& matches, FixedSelection& output) {
    std::unordered_map<uint64_t, uint32_t> current;
    if (inputs.size() > UINT32_MAX || sources.size() > UINT32_MAX) {
      return false;
    }
    for (size_t input = 0; input < inputs.size(); ++input) {
      if (!inputs[input].id || !current.emplace(inputs[input].id, uint32_t(input)).second) {
        return false;
      }
    }
    std::unordered_map<uint64_t, bool> detached;
    for (uint64_t id : pendingDetachIds) {
      if (!id) {
        return false;
      }
      detached.emplace(id, true);
    }
    FixedSelection next;
    next.inputs.resize(inputs.size(), false);
    next.slots.reserve(sources.size());
    for (const auto& batch : sources) {
      if (batch.empty() || batch.size() > UINT32_MAX) {
        return false;
      }
      next.slots.emplace_back(batch.size(), false);
    }
    for (const auto& child : index) {
      const auto found = current.find(child.first);
      if (!child.first || child.second.empty()) {
        return false;
      }
      // Validate index locations even when this child is removed or pending detachment.
      for (const VisibilityLocation& location : child.second) {
        if (location.batch >= sources.size() || location.member >= sources[location.batch].size() ||
            sources[location.batch][location.member].id != child.first) {
          return false;
        }
      }
      if (found == current.end() || detached.count(child.first)) {
        continue;
      }
      const uint32_t input = found->second;
      const uint32_t count = inputs[input].partCount;
      if (!count || child.second.size() != count) {
        continue;
      }
      std::vector<bool> covered(count, false);
      bool complete = true;
      for (const VisibilityLocation& location : child.second) {
        const FixedSource& source = sources[location.batch][location.member];
        if (source.part >= count || covered[source.part] || !matches(input, location.batch, location.member)) {
          complete = false;
          break;
        }
        covered[source.part] = true;
      }
      if (!complete) {
        continue;
      }
      next.inputs[input] = true;
      for (const VisibilityLocation& location : child.second) {
        next.slots[location.batch][location.member] = true;
      }
    }
    for (const auto& slots : next.slots) {
      const size_t active = size_t(std::count(slots.begin(), slots.end(), true));
      if (active) {
        ++next.fixedChunks;
        next.residentParts += slots.size();
        next.activeParts += active;
      }
    }
    output = std::move(next);
    return true;
  }

  // All allocations and input validation precede the shared index-state transaction.
  template<typename Change>
  static bool detachmentChanges(const std::vector<uint64_t>& ids, std::vector<Change>& output) {
    return detachmentChanges(ids, std::vector<Change>{}, output);
  }

  template<typename Change>
  static bool detachmentChanges(const std::vector<uint64_t>& ids, const std::vector<Change>& changes,
                                std::vector<Change>& output) {
    if (ids.size() > SIZE_MAX - changes.size()) {
      return false;
    }
    std::vector<Change> next = changes;
    next.reserve(changes.size() + ids.size());
    for (uint64_t id : ids) {
      if (!id) {
        return false;
      }
      next.push_back({id, false});
    }
    output.swap(next);
    return true;
  }

  // Called only after all affected masks, index states and output have been published successfully.
  // Integer hashing, equality, node destruction and vector destruction cannot allocate or throw.
  static size_t forgetDetachedChildren(VisibilityIndex& index, const std::vector<uint64_t>& ids) noexcept {
    size_t removed = 0;
    for (uint64_t id : ids) {
      removed += index.erase(id);
    }
    return removed;
  }

  // Build only affected masks. Changes may contain unowned children and repeated IDs; the last value wins.
  // Getter returns a const reference to the current mask. Invalid input and allocation failure preserve output.
  template<typename Changes, typename Getter>
  static bool visibilityPatches(const VisibilityIndex& index, uint32_t batchCount, const Changes& changes,
                                Getter&& getMask, std::vector<VisibilityPatch>& output) {
    std::vector<VisibilityPatch> next;
    std::unordered_map<uint32_t, size_t> positions;
    for (const auto& change : changes) {
      if (!change.id) {
        return false;
      }
      const auto found = index.find(change.id);
      if (found == index.end()) {
        continue;
      }
      for (const VisibilityLocation& location : found->second) {
        if (location.batch >= batchCount) {
          return false;
        }
        const auto& old = getMask(location.batch);
        if (location.member >= old.size()) {
          return false;
        }
        auto where = positions.find(location.batch);
        if (where == positions.end()) {
          const size_t position = next.size();
          next.push_back({location.batch, old});
          where = positions.emplace(location.batch, position).first;
        }
        next[where->second].visible[location.member] = change.visible;
      }
    }
    next.erase(std::remove_if(next.begin(), next.end(), [&](const VisibilityPatch& patch) {
      return patch.visible == getMask(patch.batch);
    }), next.end());
    std::sort(next.begin(), next.end(), [](const VisibilityPatch& a, const VisibilityPatch& b) {
      return a.batch < b.batch;
    });
    output.swap(next);
    return true;
  }

  // GPU/index builders and CPU fixtures share the same all-staged-before-publication transaction.
  template<typename Stage, typename Builder>
  static bool stageVisibilityPatches(const std::vector<VisibilityPatch>& patches, Builder&& build,
                                     std::vector<Stage>& output, size_t capacity = 0) {
    std::vector<Stage> next;
    next.reserve(std::max(patches.size(), capacity));
    for (const VisibilityPatch& patch : patches) {
      Stage stage;
      if (!build(patch, stage)) {
        return false;
      }
      next.push_back(std::move(stage));
    }
    output.swap(next);
    return true;
  }

  // The visitor receives a batch and its staged patch index, or UINT32_MAX for an unchanged republished plan.
  template<typename Visitor>
  static void visitVisibilityPublications(uint32_t batchCount, const std::vector<VisibilityPatch>& patches,
                                         bool republishAll, Visitor&& visit) {
    if (republishAll) {
      uint32_t changed = 0;
      for (uint32_t batch = 0; batch < batchCount; ++batch) {
        const bool updated = changed < patches.size() && patches[changed].batch == batch;
        visit(batch, updated ? changed : UINT32_MAX);
        changed += updated;
      }
    } else {
      for (uint32_t patch = 0; patch < patches.size(); ++patch) {
        visit(patches[patch].batch, patch);
      }
    }
  }

  // Unsigned oct16x2 is the static USD importer's R32_UINT normal format.
  static void decodeOctNormal(uint32_t encoded, float normal[3]) {
    const double x = double(encoded & 0xffffu) * (2.0 / 65535.0) - 1.0;
    const double y = double(encoded >> 16) * (2.0 / 65535.0) - 1.0;
    const double z = 1.0 - std::abs(x) - std::abs(y);
    const double t = std::max(-z, 0.0);
    const double nx = x + (x >= 0.0 ? -t : t);
    const double ny = y + (y >= 0.0 ? -t : t);
    const double length = std::sqrt(nx * nx + ny * ny + z * z);
    normal[0] = float(nx / length);
    normal[1] = float(ny / length);
    normal[2] = float(z / length);
  }

  // Removes incomplete inputs until every remaining part belongs to a cross-input batch.
  static std::vector<bool> completeInputs(const std::vector<uint32_t>& partCounts,
                                         const std::vector<std::vector<PartRef>>& buckets) {
    const size_t count = partCounts.size();
    const size_t limit = std::numeric_limits<size_t>::max();
    if (count == limit) {
      throw std::length_error("Retained input offset overflow");
    }
    std::vector<size_t> partOffsets(count + 1, 0);
    for (size_t input = 0; input < count; ++input) {
      if (partCounts[input] > limit - partOffsets[input]) {
        throw std::length_error("Retained part offset overflow");
      }
      partOffsets[input + 1] = partOffsets[input] + partCounts[input];
    }
    const auto valid = [&](const PartRef& ref) {
      return ref.input < count && ref.part < partCounts[ref.input];
    };
    std::vector<size_t> owners(buckets.size(), 0);
    std::vector<size_t> offsets(count + 1, 0);
    std::vector<size_t> lastBucket(count, limit);
    // Each input owns one adjacency edge per bucket, irrespective of duplicate refs or multiple parts.
    for (size_t bucket = 0; bucket < buckets.size(); ++bucket) {
      for (const PartRef& ref : buckets[bucket]) {
        if (valid(ref) && lastBucket[ref.input] != bucket) {
          lastBucket[ref.input] = bucket;
          ++owners[bucket];
          ++offsets[size_t(ref.input) + 1];
        }
      }
    }
    for (size_t input = 0; input < count; ++input) {
      if (offsets[input + 1] > limit - offsets[input]) {
        throw std::length_error("Retained adjacency offset overflow");
      }
      offsets[input + 1] += offsets[input];
    }
    std::vector<size_t> adjacency(offsets.back());
    std::vector<size_t> cursor = offsets;
    std::vector<size_t> coverage(partOffsets.back(), 0);
    std::vector<uint32_t> missing = partCounts;
    std::fill(lastBucket.begin(), lastBucket.end(), limit);
    for (size_t bucket = 0; bucket < buckets.size(); ++bucket) {
      for (const PartRef& ref : buckets[bucket]) {
        if (!valid(ref)) {
          continue;
        }
        if (lastBucket[ref.input] != bucket) {
          lastBucket[ref.input] = bucket;
          adjacency[cursor[ref.input]++] = bucket;
        }
        if (owners[bucket] < 2) {
          continue;
        }
        size_t& covered = coverage[partOffsets[ref.input] + ref.part];
        if (covered == limit) {
          throw std::length_error("Retained part coverage overflow");
        }
        if (covered++ == 0) {
          --missing[ref.input];
        }
      }
    }
    std::vector<bool> active(count, false);
    std::vector<bool> queued(count, false);
    std::vector<size_t> pending;
    pending.reserve(count);
    for (size_t input = 0; input < count; ++input) {
      active[input] = partCounts[input] != 0;
      if (active[input] && missing[input]) {
        queued[input] = true;
        pending.push_back(input);
      }
    }
    for (size_t next = 0; next < pending.size(); ++next) {
      const size_t input = pending[next];
      // A queued owner stays active until this pop applies each of its bucket losses exactly once.
      active[input] = false;
      for (size_t edge = offsets[input]; edge < offsets[input + 1]; ++edge) {
        const size_t bucket = adjacency[edge];
        if (owners[bucket]-- != 2) {
          continue;
        }
        // A bucket stops covering every part only once, when its distinct owner count falls below two.
        // Duplicate refs contributed separately, so withdrawing each ref is symmetric.
        for (const PartRef& ref : buckets[bucket]) {
          if (!valid(ref)) {
            continue;
          }
          if (--coverage[partOffsets[ref.input] + ref.part] == 0 && active[ref.input]) {
            ++missing[ref.input];
            if (!queued[ref.input]) {
              queued[ref.input] = true;
              pending.push_back(ref.input);
            }
          }
        }
      }
    }
    return active;
  }

  // Costs describe complete eligible buckets after incomplete inputs have been removed.
  // Inputs connect all of their buckets, so a selected component never owns a partial replacement.
  static bool selectBudget(const std::vector<uint32_t>& partCounts, const std::vector<BudgetBucket>& buckets,
                           const BudgetLimits& limits, BudgetSelection& output) {
    std::vector<std::vector<PartRef>> refs;
    std::vector<std::vector<uint8_t>> occurrences(partCounts.size());
    for (size_t i = 0; i < partCounts.size(); ++i) {
      occurrences[i].resize(partCounts[i]);
    }
    refs.reserve(buckets.size());
    for (const BudgetBucket& bucket : buckets) {
      for (const PartRef& part : bucket.parts) {
        if (part.input >= partCounts.size() || part.part >= partCounts[part.input] ||
            occurrences[part.input][part.part]++) {
          return false;
        }
      }
      refs.push_back(bucket.parts);
    }
    const auto eligible = completeInputs(partCounts, refs);
    std::vector<uint32_t> parents(partCounts.size());
    for (size_t i = 0; i < parents.size(); ++i) {
      parents[i] = uint32_t(i);
    }
    const auto root = [&](uint32_t input) {
      while (parents[input] != input) {
        parents[input] = parents[parents[input]];
        input = parents[input];
      }
      return input;
    };
    for (const BudgetBucket& bucket : buckets) {
      uint32_t first = UINT32_MAX;
      for (const PartRef& part : bucket.parts) {
        if (eligible[part.input]) {
          if (first == UINT32_MAX) {
            first = part.input;
          } else {
            parents[root(part.input)] = root(first);
          }
        }
      }
    }
    struct Component {
      uint64_t bytes = 0;
      uint64_t parts = 0;
      uint64_t cachedId = UINT64_MAX;
      uint64_t stableKey = UINT64_MAX;
      bool overflow = false;
      std::vector<uint32_t> inputs;
      std::vector<uint32_t> buckets;
    };
    std::vector<Component> components(partCounts.size());
    for (size_t i = 0; i < eligible.size(); ++i) {
      if (eligible[i]) {
        components[root(uint32_t(i))].inputs.push_back(uint32_t(i));
      }
    }
    BudgetSelection selected;
    selected.inputs.resize(partCounts.size(), false);
    selected.buckets.resize(buckets.size(), false);
    for (size_t i = 0; i < buckets.size(); ++i) {
      const BudgetBucket& bucket = buckets[i];
      auto first = std::find_if(bucket.parts.begin(), bucket.parts.end(), [&](const PartRef& part) { return eligible[part.input]; });
      if (first == bucket.parts.end()) {
        continue;
      }
      Component& component = components[root(first->input)];
      component.buckets.push_back(uint32_t(i));
      component.parts += bucket.parts.size();
      component.overflow = component.overflow || bucket.bytes > UINT64_MAX - component.bytes;
      if (!component.overflow) {
        component.bytes += bucket.bytes;
      }
      component.stableKey = std::min(component.stableKey, bucket.stableKey);
      if (bucket.cachedId) {
        component.cachedId = std::min(component.cachedId, bucket.cachedId);
      }
      selected.projectedBatches += 1;
      selected.projectedParts += bucket.parts.size();
      selected.projectedBytes = bucket.bytes > UINT64_MAX - selected.projectedBytes
        ? UINT64_MAX : selected.projectedBytes + bucket.bytes;
    }
    std::vector<uint32_t> order;
    for (size_t i = 0; i < components.size(); ++i) {
      if (!components[i].inputs.empty()) {
        order.push_back(uint32_t(i));
      }
    }
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      const Component& ca = components[a];
      const Component& cb = components[b];
      return ca.cachedId != cb.cachedId ? ca.cachedId < cb.cachedId : ca.stableKey < cb.stableKey;
    });
    for (uint32_t index : order) {
      const Component& component = components[index];
      if (component.overflow || component.bytes > limits.maxBytes - selected.selectedBytes ||
          component.buckets.size() > limits.maxBatches - selected.selectedBatches ||
          component.parts > limits.maxParts - selected.selectedParts) {
        selected.rejectedInputs += component.inputs.size();
        selected.rejectedBatches += component.buckets.size();
        ++selected.rejectedComponents;
        continue;
      }
      selected.selectedBytes += component.bytes;
      selected.selectedBatches += component.buckets.size();
      selected.selectedParts += component.parts;
      for (uint32_t input : component.inputs) {
        selected.inputs[input] = true;
      }
      for (uint32_t bucket : component.buckets) {
        selected.buckets[bucket] = true;
      }
    }
    std::vector<std::vector<PartRef>> chosen;
    for (size_t i = 0; i < buckets.size(); ++i) {
      if (selected.buckets[i]) {
        chosen.push_back(buckets[i].parts);
      }
    }
    // A final proof also catches malformed caller buckets before any packing or GPU allocation.
    if (completeInputs(partCounts, chosen) != selected.inputs) {
      return false;
    }
    output = std::move(selected);
    return true;
  }

  // Negative determinants remain on the canonical path to preserve facing and normal conventions.
  static bool validTransform(const std::array<float, 16>& m) {
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

  // Readers are synchronous and receive the source ordinal plus its local vertex/index.
  // A default Vertex preserves missing normal/UV = zero and color = opaque white.
  // Reader failure or any exception leaves output and its previously published bytes unchanged.
  template<typename VertexReader, typename IndexReader>
  static bool packRead(const std::vector<ReadSource>& sources, const Limits& limits,
                       VertexReader&& readVertex, IndexReader&& readIndex, Data& output) {
    try {
      uint64_t vertices = 0;
      uint64_t indices = 0;
      if (sources.size() < 2) {
        return false;
      }
      for (const ReadSource& source : sources) {
        if (!source.vertexCount || !source.indexCount ||
            source.indexCount % 3 || !validTransform(source.transform)) {
          return false;
        }
        vertices += source.vertexCount;
        indices += source.indexCount;
        if (vertices > limits.maxVertices || indices > limits.maxIndices ||
            vertices > UINT32_MAX || indices > UINT32_MAX ||
            vertices * sizeof(Vertex) + indices * sizeof(uint32_t) > limits.maxBytes) {
          return false;
        }
      }
      Data packed;
      packed.vertices.reserve(size_t(vertices));
      packed.indices.reserve(size_t(indices));
      packed.ranges.reserve(sources.size());
      // Validate/decode each index once into private rebased topology before reading vertices.
      uint32_t firstVertex = 0;
      for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        const auto& source = sources[sourceIndex];
        const uint32_t firstIndex = uint32_t(packed.indices.size());
        for (uint32_t i = 0; i < source.indexCount; ++i) {
          uint32_t index = 0;
          if (!readIndex(sourceIndex, i, index) || index >= source.vertexCount) {
            return false;
          }
          packed.indices.push_back(firstVertex + index);
        }
        packed.ranges.push_back({firstVertex, source.vertexCount, firstIndex, source.indexCount});
        firstVertex += source.vertexCount;
      }
      for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        const auto& source = sources[sourceIndex];
        const auto& m = source.transform;
        // Cofactor rows are the inverse transpose, apart from a positive scalar.
        const double cofactors[9] = {
          double(m[5]) * m[10] - double(m[9]) * m[6], double(m[9]) * m[2] - double(m[1]) * m[10], double(m[1]) * m[6] - double(m[5]) * m[2],
          double(m[8]) * m[6] - double(m[4]) * m[10], double(m[0]) * m[10] - double(m[8]) * m[2], double(m[4]) * m[2] - double(m[0]) * m[6],
          double(m[4]) * m[9] - double(m[8]) * m[5], double(m[8]) * m[1] - double(m[0]) * m[9], double(m[0]) * m[5] - double(m[4]) * m[1]
        };
        for (uint32_t i = 0; i < source.vertexCount; ++i) {
          Vertex vertex {};
          if (!readVertex(sourceIndex, i, vertex)) {
            return false;
          }
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
      }
      output = std::move(packed);
      return true;
    } catch (...) {
      return false;
    }
  }

  // Compatibility pointer reader; transformation/validation lives only in packRead.
  static bool pack(const std::vector<Source>& sources, const Limits& limits, Data& output) {
    try {
      std::vector<ReadSource> descriptions;
      descriptions.reserve(sources.size());
      for (const Source& source : sources) {
        if (!source.vertices || !source.indices) {
          return false;
        }
        descriptions.push_back({source.vertexCount, source.indexCount, source.transform, source.normals});
      }
      return packRead(descriptions, limits,
        [&](size_t source, uint32_t vertex, Vertex& value) {
          value = sources[source].vertices[vertex];
          return true;
        },
        [&](size_t source, uint32_t index, uint32_t& value) {
          value = sources[source].indices[index];
          return true;
        }, output);
    } catch (...) {
      return false;
    }
  }

  // The immutable all-member topology remains available when every member is hidden.
  static bool selectIndices(const std::vector<uint32_t>& allIndices, const std::vector<Range>& ranges,
                            const std::vector<bool>& visible, std::vector<uint32_t>& output) {
    if (ranges.size() != visible.size()) {
      return false;
    }
    try {
      std::vector<uint32_t> selected;
      selected.reserve(allIndices.size());
      for (size_t i = 0; i < ranges.size(); ++i) {
        const Range& range = ranges[i];
        if (uint64_t(range.firstIndex) + range.indexCount > allIndices.size() || range.indexCount % 3) {
          return false;
        }
        if (visible[i]) {
          selected.insert(selected.end(), allIndices.begin() + range.firstIndex,
                          allIndices.begin() + range.firstIndex + range.indexCount);
        }
      }
      output = std::move(selected);
      return true;
    } catch (const std::bad_alloc&) {
      return false;
    }
  }
};

}
