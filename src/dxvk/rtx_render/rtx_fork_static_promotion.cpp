#include "rtx_fork_static_promotion.h"
#include "rtx_options.h"
#include "rtx_imgui.h"
#include "rtx_instance_manager.h"
#include "rtx_accel_manager.h"
#include "rtx_context.h"
#include "rtx_types.h"
#include "rtx/pass/instance_definitions.h"
#include "../imgui/imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace dxvk {

  // RTX_OPTIONs for the persistent static promotion tier.
  // All declared here (not in rtx_options.h) so the fork footprint on the
  // upstream options file stays at zero LOC. Wrapped in a fork-owned holder
  // class because the RTX_OPTION macro expands to a class-body member
  // declaration (public: inline static RtxOption<T>); it cannot live at
  // namespace scope. Mirrors the RtxForkGlobalTonemap / RtxForkHableFilmic
  // pattern used by rtx_fork_tonemap.h. Discovered by the auto-doc
  // generator via the standard RTX_OPTION macro.
  class RtxForkStaticPromotion {
    RTX_OPTION("rtx", bool, enableStaticGeometryPromotion, false,
      "Master switch for the persistent static geometry promotion tier. When enabled, "
      "stable static world geometry is detected and routed into long-lived merged BLASes "
      "that only rebuild on membership change. Reduces TLAS instance count and per-frame "
      "BLAS rebuild cost for stable scenes.");

    RTX_OPTION("rtx", uint32_t, staticGeometryPromotionFrames, 8,
      "Stability threshold (K). An instance must observe identical geometry / transform / "
      "material for K consecutive frames before it is promoted into a persistent bucket. "
      "Higher = more conservative (fewer false promotes); lower = faster ramp-up after "
      "cell transitions.");

    RTX_OPTION("rtx", uint32_t, maxPrimsInPersistentBLAS, 262144,
      "Maximum primitive count per persistent merged BLAS. Larger than maxPrimsInMergedBLAS "
      "because the persistent rebuild cost is amortized across many frames.");

    RTX_OPTION("rtx", uint32_t, persistentBlasMemoryBudgetMB, 256,
      "LRU eviction trigger. When total persistent BLAS memory exceeds this budget, the "
      "least-recently-rendered bucket is demoted wholesale.");

    RTX_OPTION("rtx", bool, enableSameFrameContentDedup, false,
      "Layer 2: when DrawCallCache::get would allocate a fresh BlasEntry because the only "
      "topological match was already touched this frame, fall through to a cross-bucket "
      "VertexDataHash + material + bones search and share the existing entry on match. "
      "Reduces duplicate BlasEntries for games that submit the same content twice per frame.");

    RTX_OPTION("rtx", float, staticPromotionTransformEpsilon, 1.0e-4f,
      "Per-element transform stability tolerance. Element-wise abs-diff above this threshold "
      "resets transformStableFrames. Default 1e-4 yields sub-pixel positional error if a "
      "within-epsilon drift is absorbed by the persistent BLAS.");
  };

  namespace static_promotion {
    PromotionFrameCounters s_frameCounters;
    const PromotionFrameCounters& getFrameCounters() { return s_frameCounters; }

    // Bumps the stability histogram bucket for a given min-stability value.
    void bumpStabilityHistogram(uint32_t minStability) {
      if (minStability >= 32) { ++s_frameCounters.stabilityHistogram[5]; return; }
      if (minStability >= 16) { ++s_frameCounters.stabilityHistogram[4]; return; }
      if (minStability >= 8)  { ++s_frameCounters.stabilityHistogram[3]; return; }
      if (minStability >= 4)  { ++s_frameCounters.stabilityHistogram[2]; return; }
      if (minStability >= 1)  { ++s_frameCounters.stabilityHistogram[1]; return; }
      ++s_frameCounters.stabilityHistogram[0];
    }
  } // namespace static_promotion

  static static_promotion::StaticPromotionPool s_pool;

  namespace static_promotion {
    StaticPromotionPool& getPool() { return s_pool; }

    PersistentBlasBucket& StaticPromotionPool::addInstance(uint64_t bucketKeyHash, RtInstance* instance, uint32_t currentFrame) {
      auto it = m_instanceToBucket.find(instance);
      if (it != m_instanceToBucket.end()) {
        if (it->second == bucketKeyHash) {
          auto& b = m_buckets[bucketKeyHash];
          b.lastTouchedFrame = currentFrame;
          return b;
        }
        removeInstance(instance);
      }

      auto& bucket = m_buckets[bucketKeyHash];
      bucket.bucketKeyHash = bucketKeyHash;
      bucket.members.push_back(instance);
      bucket.dirty = true;
      bucket.lastTouchedFrame = currentFrame;
      m_instanceToBucket[instance] = bucketKeyHash;
      return bucket;
    }

    void StaticPromotionPool::removeInstance(RtInstance* instance) {
      auto it = m_instanceToBucket.find(instance);
      if (it == m_instanceToBucket.end()) return;
      const uint64_t keyHash = it->second;
      auto bucketIt = m_buckets.find(keyHash);
      if (bucketIt != m_buckets.end()) {
        auto& members = bucketIt->second.members;
        members.erase(std::remove(members.begin(), members.end(), instance), members.end());
        bucketIt->second.dirty = true;
        if (members.empty()) {
          m_totalBytes -= bucketIt->second.blasBytes;
          m_buckets.erase(bucketIt);
        }
      }
      m_instanceToBucket.erase(it);
    }

    PersistentBlasBucket* StaticPromotionPool::findBucketFor(RtInstance* instance) {
      auto it = m_instanceToBucket.find(instance);
      if (it == m_instanceToBucket.end()) return nullptr;
      auto bucketIt = m_buckets.find(it->second);
      return (bucketIt == m_buckets.end()) ? nullptr : &bucketIt->second;
    }

    void StaticPromotionPool::touchAll(uint32_t currentFrame) {
      for (auto& [k, b] : m_buckets) b.lastTouchedFrame = currentFrame;
    }

    void StaticPromotionPool::setBucketBytes(uint64_t bucketKeyHash, uint64_t bytes) {
      auto it = m_buckets.find(bucketKeyHash);
      if (it == m_buckets.end()) return;
      m_totalBytes -= it->second.blasBytes;
      it->second.blasBytes = bytes;
      m_totalBytes += bytes;
    }

    uint32_t StaticPromotionPool::evictLeastRecentlyUsed() {
      if (m_buckets.empty()) return 0;
      auto victim = m_buckets.begin();
      for (auto it = m_buckets.begin(); it != m_buckets.end(); ++it) {
        if (it->second.lastTouchedFrame < victim->second.lastTouchedFrame) {
          victim = it;
        }
      }
      const uint32_t demoted = static_cast<uint32_t>(victim->second.members.size());
      for (RtInstance* inst : victim->second.members) {
        m_instanceToBucket.erase(inst);
      }
      m_totalBytes -= victim->second.blasBytes;
      m_buckets.erase(victim);
      return demoted;
    }

    uint32_t StaticPromotionPool::enforceMemoryBudget(uint64_t budgetBytes) {
      uint32_t evictions = 0;
      while (m_totalBytes > budgetBytes && !m_buckets.empty()) {
        evictLeastRecentlyUsed();
        ++evictions;
      }
      return evictions;
    }
  } // namespace static_promotion

  namespace {
    // Element-wise abs-diff comparison of two VkTransformMatrixKHR matrices under
    // the user-configured epsilon. Returns true if every element is within
    // tolerance, false otherwise.
    bool transformWithinEpsilon(const VkTransformMatrixKHR& a, const VkTransformMatrixKHR& b, float eps) {
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
          if (std::abs(a.matrix[row][col] - b.matrix[row][col]) > eps) {
            return false;
          }
        }
      }
      return true;
    }

    // Computes a stable hash of the bucket-key-relevant fields of an RtInstance.
    // Mirrors BlasBucketKey but reads directly from the live VkInstance and the
    // two boolean flags the bucket key tracks.
    uint64_t hashBucketKeyFields(const VkAccelerationStructureInstanceKHR& vk, bool isSubsurface, bool isUnordered) {
      struct Packed {
        uint32_t sbtOffset;
        uint32_t customIndexFlags;
        uint32_t instanceFlags;
        uint8_t  mask;
        bool     isUnordered;
        bool     isSubsurface;
        uint8_t  pad;
      } p {};
      p.sbtOffset = vk.instanceShaderBindingTableRecordOffset;
      p.customIndexFlags = vk.instanceCustomIndex & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK);
      p.instanceFlags = vk.flags;
      p.mask = static_cast<uint8_t>(vk.mask);
      p.isUnordered = isUnordered;
      p.isSubsurface = isSubsurface;
      return XXH3_64bits(&p, sizeof(p));
    }
  } // namespace

  namespace fork_hooks {

    void tickStabilityCounters(const std::vector<RtInstance*>& instances, uint32_t currentFrame) {
      if (!RtxForkStaticPromotion::enableStaticGeometryPromotion()) {
        return;
      }
      const float eps = RtxForkStaticPromotion::staticPromotionTransformEpsilon();

      // Reset the per-frame histogram and instance total; everything else in
      // s_frameCounters is owned by Tasks 4-5 and stays untouched here.
      std::memset(&static_promotion::s_frameCounters.stabilityHistogram, 0,
                  sizeof(static_promotion::s_frameCounters.stabilityHistogram));
      static_promotion::s_frameCounters.totalInstances = static_cast<uint32_t>(instances.size());

      for (RtInstance* inst : instances) {
        if (!inst) continue;

        // Geometry: bumps when the linked BlasEntry's content was NOT updated this frame.
        BlasEntry* blas = inst->getBlas();
        const bool geometryChanged = blas && (blas->frameLastUpdated == currentFrame);
        if (geometryChanged) {
          inst->m_geometryStableFrames = 0;
        } else {
          ++inst->m_geometryStableFrames;
        }

        // Transform: element-wise compare against last frame.
        const VkTransformMatrixKHR& cur = inst->getVkInstance().transform;
        if (transformWithinEpsilon(cur, inst->m_prevFrameTransform, eps)) {
          ++inst->m_transformStableFrames;
        } else {
          inst->m_transformStableFrames = 0;
        }
        inst->m_prevFrameTransform = cur;

        // Material/SBT/mask: compare hashed bucket key.
        const uint64_t curHash = hashBucketKeyFields(
          inst->getVkInstance(),
          inst->isSubsurface(),
          inst->usesUnorderedApproximations());
        if (curHash == inst->m_prevFrameBucketKeyHash) {
          ++inst->m_materialStableFrames;
        } else {
          inst->m_materialStableFrames = 0;
        }
        inst->m_prevFrameBucketKeyHash = curHash;

        // Bucket the minimum of the three counters into the histogram.
        const uint32_t minStability = std::min({
          inst->m_geometryStableFrames,
          inst->m_transformStableFrames,
          inst->m_materialStableFrames
        });
        static_promotion::bumpStabilityHistogram(minStability);
      }
    }

    bool tryRouteToPersistentBucket(AccelManager& /*mgr*/, RtInstance* /*instance*/, uint32_t /*currentFrame*/) {
      return false;
    }

    void touchPersistentBlasesForFastSkip(AccelManager& /*mgr*/, uint32_t /*currentFrame*/) {}

    void onInstanceRemoved(RtInstance* /*instance*/) {}

    void showStaticPromotionPanel() {
      if (!ImGui::CollapsingHeader("Static Geometry Promotion", ImGuiTreeNodeFlags_None)) {
        return;
      }

      // RTX_OPTION knobs (RemixGui::* per the tonemap fork pattern).
      RemixGui::Checkbox("Enable static geometry promotion",
        &RtxForkStaticPromotion::enableStaticGeometryPromotionObject());
      RemixGui::DragInt("Stability threshold (K frames)",
        &RtxForkStaticPromotion::staticGeometryPromotionFramesObject(), 1.0f, 1, 64);
      RemixGui::DragInt("Max prims per persistent BLAS",
        &RtxForkStaticPromotion::maxPrimsInPersistentBLASObject(), 1024.0f, 4096, 1048576);
      RemixGui::DragInt("Memory budget (MB)",
        &RtxForkStaticPromotion::persistentBlasMemoryBudgetMBObject(), 4.0f, 16, 2048);
      RemixGui::DragFloat("Transform epsilon",
        &RtxForkStaticPromotion::staticPromotionTransformEpsilonObject(), 1e-5f, 1e-6f, 1e-2f, "%.6f");
      RemixGui::Checkbox("Enable same-frame content dedup",
        &RtxForkStaticPromotion::enableSameFrameContentDedupObject());

      ImGui::Separator();

      const auto& fc = static_promotion::getFrameCounters();
      ImGui::Text("Instances this frame:        %u", fc.totalInstances);
      ImGui::Text("Promoted this frame:         %u", fc.promotedThisFrame);
      ImGui::Text("Demoted this frame:          %u", fc.demotedThisFrame);
      ImGui::Text("BLAS builds this frame:      %u", fc.totalBlasBuilds);
      ImGui::Text("LRU evictions this frame:    %u", fc.lruEvictionsThisFrame);
      ImGui::Spacing();
      ImGui::Text("Persistent buckets:          %u (%u members)",
                  fc.persistentBucketCount, fc.persistentBucketMembers);
      ImGui::Text("Persistent BLAS memory:      %llu MB",
                  (unsigned long long)(fc.persistentBlasBytes / (1024 * 1024)));
      ImGui::Spacing();
      ImGui::Text("TLAS instances:");
      ImGui::Text("  Persistent:                %u", fc.tlasPersistent);
      ImGui::Text("  Ephemeral merged:          %u", fc.tlasMergedEphemeral);
      ImGui::Text("  Dynamic (own BLAS):        %u", fc.tlasDynamic);
      ImGui::Text("  Point instancer:           %u", fc.tlasPointInstancer);

      ImGui::Spacing();
      ImGui::Text("Stability histogram (frames stable):");
      const char* bandLabels[] = { "0", "1-3", "4-7", "8-15", "16-31", "32+" };
      for (int b = 0; b < 6; ++b) {
        ImGui::Text("  %-6s : %u", bandLabels[b], fc.stabilityHistogram[b]);
      }
    }

    void writeStaticPromotionDebugView(DxvkContext& /*ctx*/) {
      // Placeholder — actual per-pixel BLAS-source colorization will land in a
      // follow-up commit after Task 5's persistent routing exists to read from.
      // The enum value is reachable from the menu so users can see it listed.
    }

  } // namespace fork_hooks
} // namespace dxvk
