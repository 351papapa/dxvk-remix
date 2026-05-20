#pragma once

// rtx_fork_static_promotion.h — fork-owned persistent BLAS promotion tier.
// Sits above the per-frame merged BLAS pool in AccelManager::mergeInstancesIntoBlas:
// stable static world geometry is detected via three per-instance stability counters
// (geometry / transform / material) and routed into long-lived "persistent buckets"
// whose BLASes only rebuild on membership change. See
// docs/superpowers/specs/2026-05-20-static-geometry-promotion-design.md.

#include <cstdint>
#include <vector>
#include <unordered_map>
#include "../../util/util_singleton.h"
#include "../../util/util_vector.h"

namespace dxvk {
  class RtInstance;
  class AccelManager;
  class DxvkContext;
  class DxvkDevice;
  struct BlasEntry;

  namespace fork_hooks {

    // Called once per frame, after instance manager finalization and before
    // AccelManager::mergeInstancesIntoBlas runs. Updates the three stability
    // counters on every RtInstance based on per-frame deltas (transform diff,
    // BlasEntry::frameLastUpdated, BlasBucketKey). Implementation in
    // rtx_fork_static_promotion.cpp.
    void tickStabilityCounters(const std::vector<RtInstance*>& instances, uint32_t currentFrame);

    // Called from AccelManager::mergeInstancesIntoBlas for each non-skipped
    // RtInstance. If the instance is currently a member of a persistent
    // bucket, restores its TLAS instance from the persistent pool and returns
    // true (caller should skip the per-frame routing for this instance). If
    // the instance is newly eligible for promotion, adds it to a persistent
    // bucket and returns true. Returns false otherwise — the caller proceeds
    // with the existing per-frame merged/dynamic routing.
    bool tryRouteToPersistentBucket(AccelManager& mgr, RtInstance* instance, uint32_t currentFrame);

    // Called from the AccelManager full-skip fast path. Touches every
    // persistent bucket's BLAS so the GC does not collect them while the
    // scene is unchanged.
    void touchPersistentBlasesForFastSkip(AccelManager& mgr, uint32_t currentFrame);

    // Called from AccelManager::removeInstanceFromBucketCache when an
    // instance is being torn down (GC, hidden, mask=0). Demotes the instance
    // from its persistent bucket if it had been promoted.
    void onInstanceRemoved(RtInstance* instance);

    // Renders the "Static Geometry Promotion" ImGui panel inside the
    // Rendering section of the dev menu. Reads per-frame counters captured
    // during the most recent mergeInstancesIntoBlas pass.
    void showStaticPromotionPanel();

    // Writes the debug view (per-pixel "BLAS source" colorization) when the
    // active debug view enum is DebugViewIdx::BlasSource.
    void writeStaticPromotionDebugView(DxvkContext& ctx);

  } // namespace fork_hooks

  namespace static_promotion {
    // Per-frame diagnostic counters populated during mergeInstancesIntoBlas and
    // displayed in the ImGui panel. All fields reset to 0 at the top of each
    // frame's tick.
    struct PromotionFrameCounters {
      uint32_t totalInstances = 0;
      uint32_t promotedThisFrame = 0;
      uint32_t demotedThisFrame = 0;
      uint32_t totalBlasBuilds = 0;
      uint32_t persistentBucketCount = 0;
      uint32_t persistentBucketMembers = 0;
      uint32_t tlasPersistent = 0;
      uint32_t tlasMergedEphemeral = 0;
      uint32_t tlasDynamic = 0;
      uint32_t tlasPointInstancer = 0;
      // Stability histogram: count of RtInstances whose min(stability) lands in
      // each band (0, 1-3, 4-7, 8-15, 16-31, 32+).
      uint32_t stabilityHistogram[6] = {};
      uint64_t persistentBlasBytes = 0;
      uint32_t lruEvictionsThisFrame = 0;
    };

    // Accessor for the latest-completed frame's counters.
    const PromotionFrameCounters& getFrameCounters();

    // A persistent merged-BLAS bucket. Long-lived analog of AccelManager::BlasBucket:
    // groups instances by BlasBucketKey, holds a PooledBlas that survives across
    // frames until membership changes.
    struct PersistentBlasBucket {
      uint64_t bucketKeyHash = 0;       // hashBucketKeyFields() — bucket identity
      std::vector<RtInstance*> members; // current member set
      uint64_t blasBytes = 0;           // tracked for memory budget
      uint32_t lastTouchedFrame = 0;    // updated each frame the bucket is rendered
      bool dirty = true;                // true if members have changed since last BLAS build
      // PooledBlas binding happens in StaticPromotionPool::ensureBlasBuilt (Task 5).
    };

    // Owns the set of persistent buckets, the bucket lookup map, and the LRU
    // eviction policy. All access is single-threaded (called from the merge pass
    // which is itself single-threaded).
    class StaticPromotionPool {
    public:
      // Adds an instance to a bucket keyed by the given hash. Creates the bucket
      // if it does not exist. Returns the bucket the instance now belongs to.
      PersistentBlasBucket& addInstance(uint64_t bucketKeyHash, RtInstance* instance, uint32_t currentFrame);

      // Removes an instance from its bucket (if any). Marks the bucket dirty.
      // If the bucket is left empty, releases the bucket.
      void removeInstance(RtInstance* instance);

      // Returns the bucket the instance is currently a member of, or nullptr.
      PersistentBlasBucket* findBucketFor(RtInstance* instance);

      // Touches every bucket's lastTouchedFrame to currentFrame. Called from the
      // fast-skip path to keep recency information accurate.
      void touchAll(uint32_t currentFrame);

      // Returns the total bytes currently held by persistent BLASes.
      uint64_t getTotalBytes() const { return m_totalBytes; }

      // Evicts the least-recently-touched bucket. Returns the number of members
      // demoted. Called by enforceMemoryBudget when the budget is exceeded.
      uint32_t evictLeastRecentlyUsed();

      // Drives eviction until total bytes <= budgetBytes. Returns the number of
      // buckets evicted.
      uint32_t enforceMemoryBudget(uint64_t budgetBytes);

      // Diagnostic accessors used by the ImGui panel and unit tests.
      size_t getBucketCount() const { return m_buckets.size(); }
      size_t getTotalMembers() const {
        size_t n = 0;
        for (const auto& [k, b] : m_buckets) n += b.members.size();
        return n;
      }

      // For tests: register a bucket's BLAS size so the LRU policy has bytes to
      // account against. In production this is called when a BLAS is built.
      void setBucketBytes(uint64_t bucketKeyHash, uint64_t bytes);

    private:
      std::unordered_map<uint64_t, PersistentBlasBucket> m_buckets;
      std::unordered_map<RtInstance*, uint64_t> m_instanceToBucket;
      uint64_t m_totalBytes = 0;
    };

    StaticPromotionPool& getPool();
  } // namespace static_promotion
} // namespace dxvk
