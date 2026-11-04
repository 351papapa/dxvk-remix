/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstdint>
#include <chrono>
#include <memory>
#include <remix/remix_retained.h>

namespace dxvk::retained_compat {

template<typename Entry, typename Queue, typename Apply>
void setVisibilityIntent(Entry& entry, bool desired, uint32_t& count, Queue&& queue, Apply&& apply) {
  if (entry.desiredVisible == desired) {
    return;
  }
  queue(entry.record.id);
  entry.desiredVisible = desired;
  if (desired) {
    entry.record.flags &= ~REMIXAPI_RETAINED_HIDDEN;
    ++count;
  } else {
    entry.record.flags |= REMIXAPI_RETAINED_HIDDEN;
    --count;
  }
  apply();
}

enum class SurfaceCheck { Valid, MissingMaterial, IncompatibleLegacy };

inline SurfaceCheck checkSurface(bool hasMaterialHandle, bool hasMaterial, bool hasLegacyHash,
                                 uint64_t capturedRule, uint64_t activeRule, bool opaque) {
  if (hasMaterialHandle && !hasMaterial) {
    return SurfaceCheck::MissingMaterial;
  }
  if (hasLegacyHash && (capturedRule != activeRule || !hasMaterial || !opaque)) {
    return SurfaceCheck::IncompatibleLegacy;
  }
  return SurfaceCheck::Valid;
}

// No ID may change visibility or consume a sort ordinal until the whole group passes validation.
template<typename Ids, typename Validate, typename Apply>
bool applyValidatedGroup(const Ids& ids, Validate&& validate, Apply&& apply) {
  for (auto id : ids) {
    if (!validate(id)) {
      return false;
    }
  }
  for (auto id : ids) {
    if (!apply(id)) {
      return false;
    }
  }
  return true;
}

// Admission changes optional aggregation, never canonical rendering visibility.
class WorldBatchAdmissionState {
public:
  using Clock = std::chrono::steady_clock;
  bool pending() const { return m_pending; }
  void request() noexcept { m_pending = true; }
  bool due(Clock::time_point now) const { return m_pending && now >= m_next; }
  void begin(Clock::time_point now) noexcept {
    m_pending = false;
    m_next = now + std::chrono::seconds(1);
  }
  void cancel() noexcept { m_pending = false; }
private:
  bool m_pending = false;
  Clock::time_point m_next {};
};

// Pending descriptions do not replace the recipe of an already-published canonical view.
template<typename Source, typename Record>
struct PublishedSourceState {
  std::shared_ptr<const Source> source;
  Record record {};
  bool pending = false;
  void stage() noexcept { pending = true; }
  void publish(const std::shared_ptr<const Source>& value, const Record& description) noexcept {
    source = value;
    record = description;
    pending = false;
  }
  // Only fields actually applied to existing children may pass through a staged description.
  template<typename Apply>
  void updateApplied(const std::shared_ptr<const Source>& current, Apply&& apply) {
    if (!source) {
      return;
    }
    if (!pending) {
      source = current;
    } else {
      auto updated = std::make_shared<Source>(*source);
      apply(*updated, *current);
      source = std::move(updated);
    }
  }
  void forget() noexcept { source.reset(); record = {}; }
};

enum class SourceDestruction { None, RemoveEntry, ForgetPublished };
inline SourceDestruction sourceDestruction(bool currentMatches, bool publishedMatches) {
  return currentMatches ? SourceDestruction::RemoveEntry
    : publishedMatches ? SourceDestruction::ForgetPublished : SourceDestruction::None;
}

// Publication readiness is distinct from immutable geometry cache ownership.
struct WorldBatchUpdateState {
  bool ready = false;
  bool structureDirty = true;
  bool presentationDirty = false;
  uint64_t revision = 0;

  void invalidateStructure() { structureDirty = true; }
  void hide() { presentationDirty = true; }
  bool incremental(uint64_t currentRevision) const {
    return ready && !structureDirty && revision && revision == currentRevision;
  }
  void prepared(uint64_t currentRevision) {
    ready = currentRevision != 0;
    structureDirty = false;
    presentationDirty = false;
    revision = currentRevision;
  }
  void presented() { presentationDirty = false; }
  void lostPublication() {
    ready = false;
    structureDirty = true;
    presentationDirty = false;
    revision = 0;
  }
};

// Desired host visibility survives missing commits and child/asset rebuilds.
// The callback changes render children only when the applied state changes.
template<typename Entry, typename Apply>
void applyVisibility(Entry& entry, bool active, Apply&& apply) {
  const bool target = active && entry.desiredVisible;
  if (entry.visible != target) {
    apply(target);
    entry.visible = target;
  }
}

// New/rebuilt children can initially be visible even when the last children were hidden.
template<typename Entry, typename Apply>
void finishBuild(Entry& entry, Apply&& apply) {
  apply(entry.desiredVisible);
  entry.visible = entry.desiredVisible;
}

// A host transform is absolute; a replacement prim's authored transform remains relative.
template<typename Matrix>
Matrix childTransform(const Matrix& parent, const Matrix* relative) {
  return relative != nullptr ? parent * *relative : parent;
}

}
