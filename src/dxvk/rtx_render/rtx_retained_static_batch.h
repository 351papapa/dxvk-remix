/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "../dxvk_context.h"
#include "rtx_scene_types.h"
#include "rtx_retained_static_batch_cpu.h"

namespace dxvk {

struct ExternalDrawState;
enum class RetainedBatchReason : uint8_t {
  None, Source, Special, Categories, Material, MaterialService, Alpha, CameraTransform,
  Identity, Skinning, PairState, PairTransform, PairMaterial, PairLegacyMaterial,
  PairAlpha, PairLegacySurface, PairLayout, Replacement, Wetness, BufferLayout,
  PendingWrite, HostVisible, Map, Proof, Pack, NonFinite, Allocation, Count
};
using RetainedBatchRejectCounts = std::array<uint64_t, size_t(RetainedBatchReason::Count)>;
class RetainedStaticBatchCache {
  struct Key {
    std::vector<RasterGeometry> sources;
    bool operator==(const Key& other) const;
  };
  using Geometry = std::shared_ptr<const std::vector<RasterGeometry>>;
  RetainedStaticBatchLru<Key, Geometry> m_lru {64 * 1024 * 1024, 256};
  uint64_t m_creates = 0;
  uint64_t m_buildAttempts = 0;
  RetainedBatchRejectCounts m_buildRejects {};
  friend class RetainedStaticBatch;
public:
  void clear() { m_lru.clear(); }
  size_t bytes() const { return m_lru.bytes(); }
  size_t size() const { return m_lru.size(); }
  uint64_t hits() const { return m_lru.hits(); }
  uint64_t misses() const { return m_lru.misses(); }
  uint64_t creates() const { return m_creates; }
  uint64_t evictions() const { return m_lru.evictions(); }
  uint64_t buildAttempts() const { return m_buildAttempts; }
  const RetainedBatchRejectCounts& buildRejects() const { return m_buildRejects; }
};

struct RetainedStaticBatchProof {
  // All sources belong to one retained record and share its visibility lifetime.
  uint64_t parent = 0;
  bool apiOwnedOriginals = false;
  bool noMeshReplacements = false;
  bool sameWetnessRule = false;
  bool identityToolsInactive = false;
};

class RetainedStaticBatch {
public:
  // Original descriptions remain owned by the caller for invalidation and identity tools.
  // Proofs are established on the scene thread against the current asset/options epochs.
  static bool build(const Rc<DxvkContext>& ctx, const std::vector<ExternalDrawState>& sources,
                    const RetainedStaticBatchProof& proof, RetainedStaticBatchCache& cache, ExternalDrawState& output);

  // Tests the state contract before a caller partitions originals into fusion candidates.
  static bool compatible(const ExternalDrawState& a, const ExternalDrawState& b);
  static RetainedBatchReason rejectionReason(const ExternalDrawState& a, const ExternalDrawState& b);
  static const char* reasonName(RetainedBatchReason reason);
};

}
