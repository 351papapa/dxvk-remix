/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtx_scene_manager.h"

namespace dxvk {

class RetainedWorldBatcher {
public:
  struct VisibilityInput {
    uint64_t id = 0;
    bool visible = false;
  };
  struct ParentInput {
    // The canonical child handle remains owned by RetainedCompat.
    uint64_t id = 0;
    uint64_t passDomain = 0;
    uint64_t wetnessRule = 0;
    bool visible = false;
    // Borrowed only during synchronous prepare(). The caller keeps the vector stable until it returns.
    const std::vector<ExternalDrawState>& parts;
  };

  struct Member {
    uint64_t id = 0;
    uint32_t part = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
  };

  struct Plan {
    // Internal IDs never overlap the caller's retained-handle namespace.
    uint64_t id = 0;
    uint64_t revision = 0;
    bool changed = false;
    bool visible = false;
    // A hidden plan retains its last nonempty rendering state, if one exists.
    ExternalDrawState renderingState;
    // Full prepare returns active ownership. Visibility-only publications leave this vector empty.
    // Callers change ownership only from complete prepare output or successful detachment.
    std::vector<Member> members;
  };

  struct Stats {
    uint64_t prepares = 0;
    uint64_t visibilityUpdates = 0;
    uint64_t visibilityCommits = 0;
    uint64_t visibilityAffectedBatches = 0;
    uint64_t visibilityIndexBuildAttempts = 0;
    uint64_t visibilityIndexBuilds = 0;
    uint64_t visibilityFailures = 0;
    uint64_t visibilityNoops = 0;
    uint64_t visibilityRepublishes = 0;
    uint64_t lastVisibilityChanges = 0;
    uint64_t lastVisibilityAffected = 0;
    uint64_t lastVisibilityPlans = 0;
    uint64_t detachUpdates = 0;
    uint64_t detachCommits = 0;
    uint64_t detachFailures = 0;
    uint64_t detachedChildren = 0;
    uint64_t vertexBuilds = 0;
    uint64_t indexBuilds = 0;
    uint64_t vertexReuses = 0;
    uint64_t vertexBuildAttempts = 0;
    uint64_t indexBuildAttempts = 0;
    // Actual helper VB resource creation/upload work, including staging that later fails.
    uint64_t vertexAllocationAttempts = 0;
    uint64_t vertexAllocations = 0;
    uint64_t vertexBytesUploaded = 0;
    // Legacy indexBuild counters above count logical mask transitions, including all-hidden views.
    // The following counters measure actual helper IB allocation/upload work, even if later staging fails.
    uint64_t indexAllocationAttempts = 0;
    uint64_t indexAllocations = 0;
    uint64_t indexBytesUploaded = 0;
    uint64_t indexTrianglesUploaded = 0;
    // Committed transitions that needed no new index buffer.
    uint64_t revealReuse = 0;
    uint64_t allHiddenNoAllocation = 0;
    uint64_t rejectedInputs = 0;
    uint64_t rejectedParts = 0;
    uint64_t preflightRejectedInputs = 0;
    uint64_t preflightRejectedBatches = 0;
    uint64_t preflightRejectedComponents = 0;
    uint64_t lastProjectedBatches = 0;
    uint64_t lastProjectedBytes = 0;
    uint64_t lastSelectedBatches = 0;
    uint64_t lastSelectedBytes = 0;
    // Gauges from the last successful structural preparation. Resident storage includes retired holes.
    uint64_t lastFixedChunks = 0;
    uint64_t lastFixedBytes = 0;
    uint64_t lastResidentParts = 0;
    uint64_t lastActiveParts = 0;
    uint64_t lastDeltaCandidateChildren = 0;
    uint64_t lastDeltaCandidateParts = 0;
    uint64_t lastDeltaAdmittedChildren = 0;
    uint64_t lastDeltaAdmittedParts = 0;
    uint64_t lastDeadChunks = 0;
    uint64_t failures = 0;
    uint64_t failedInput = 0;
    uint64_t failedPlan = 0;
    uint64_t failedPack = 0;
    uint64_t failedAllocation = 0;
    uint64_t failedMap = 0;
    uint64_t failedException = 0;
    size_t batches = 0;
    size_t bytes = 0;
  };

  RetainedWorldBatcher();
  ~RetainedWorldBatcher();
  RetainedWorldBatcher(const RetainedWorldBatcher&) = delete;
  RetainedWorldBatcher& operator=(const RetainedWorldBatcher&) = delete;

  // Called only after source, visibility, asset or identity-tool state changes.
  // Parts are fully resolved opaque candidates from RetainedScene::resolveStaticParts.
  // Each accepted input has every part represented exactly once across the plans.
  // Unaccepted inputs are absent from all plans and keep their canonical children.
  // Failure leaves output and the active cache unchanged. Publication is the caller's transaction.
  // Pending detachments retire their old physical slots even if their current descriptions still match.
  // Fixed chunks retain all physical storage; only complete unowned inputs form new delta chunks.
  bool prepare(const Rc<DxvkContext>& ctx, const std::vector<ParentInput>& inputs,
               float cellSize, const std::vector<uint64_t>& pendingDetachIds, std::vector<Plan>& output);

  // Successful structural preparation publishes a nonzero membership revision. Clear invalidates it.
  uint64_t membershipRevision() const;

  // Only final visibility changes are accepted; the expected structural revision must still match.
  // Unknown unowned child IDs are ignored and repeated IDs use their last value. A no-op may return no plans.
  // Returns affected plans, or every plan when republishAll is true. Unchanged plans have changed=false.
  // All affected index states and output are staged before publication; failure changes neither cache nor output.
  bool updateVisibility(const Rc<DxvkContext>& ctx, uint64_t expectedRevision,
                        const std::vector<VisibilityInput>& changes, bool republishAll, std::vector<Plan>& output);

  // Permanently remove every part of these children from the current packing generation.
  // Reuses the visibility transaction, then forgets their locations without changing immutable VB/ranges.
  // Later visibility updates cannot revive a detached child, even when its canonical handle is reused.
  // Only successful full prepare can readmit it. Unknown nonzero IDs and duplicate IDs are harmless.
  // Failure preserves output, cache and ownership index; success keeps membershipRevision unchanged.
  bool detachChildren(const Rc<DxvkContext>& ctx, uint64_t expectedRevision,
                      const std::vector<uint64_t>& childIds, std::vector<Plan>& output);
  // Combined form applies final visibility/presentation and detachment in one index transaction.
  // Detachment wins over visibility=true for the same child.
  bool detachChildren(const Rc<DxvkContext>& ctx, uint64_t expectedRevision, const std::vector<uint64_t>& childIds,
                      const std::vector<VisibilityInput>& changes, bool republishAll, std::vector<Plan>& output);

  void clear();
  const Stats& stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}
