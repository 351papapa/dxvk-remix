#pragma once

#include <memory>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <remix/remix_retained.h>
#include "rtx_frontend_interface.h"
#include "rtx_option.h"
#include "rtx_retained_static_batch.h"
#include "rtx_retained_world_batch.h"
#include "rtx_retained_compat_policy.h"
#include "rtx_scene_manager.h"

namespace dxvk {

// The versioned fixed-width host catalog. Rendering belongs to RetainedScene.
class RetainedCompat {
public:
  RTX_OPTION("rtx.scene", bool, aggregateLegacyRetainedGeometry, true,
    "Combine compatible legacy retained surfaces within and across unordered instances. Disable to render each surface independently. Both modes preserve replacements and visibility.");

  explicit RetainedCompat(SceneManager& scene) : m_scene(scene) { }
  ~RetainedCompat();
  void upsert(uint64_t epoch, std::vector<remixapi_RetainedInstance>&& records,
              std::vector<frontend::PreparedInstance>&& states);
  void remove(uint64_t epoch, const std::vector<uint64_t>& ids);
  void clear(uint64_t epoch);
  void commit(const Rc<DxvkContext>& ctx, uint64_t epoch);
  void drawOrdered(const Rc<DxvkContext>& ctx, uint64_t epoch, const std::vector<uint64_t>& ids);
  void drawImmediate(const Rc<DxvkContext>& ctx, std::unique_ptr<ExternalDrawState> state);
  void finishFrame(const Rc<DxvkContext>& ctx);
  void destroyMesh(remixapi_MeshHandle mesh);
  void invalidateAsset(uint64_t asset);
  // Called before scene upload as well as at commit, including changes made after a host commit.
  void refreshBatches(const Rc<DxvkContext>& ctx, bool wetnessRulesChanged = false);
  remixapi_RetainedStatus status() const { return m_status; }
  bool hasLegacy(const ExternalDrawState& state) const;

private:
  struct Entry {
    remixapi_RetainedInstance record {};
    std::shared_ptr<const ExternalDrawState> state;
    retained_compat::PublishedSourceState<ExternalDrawState, remixapi_RetainedInstance> published;
    uint64_t publishedRefreshRevision = 0;
    std::vector<uint64_t> children;
    std::vector<uint64_t> assets;
    std::vector<std::vector<ExternalDrawState>> worldParts;
    bool everMoved = false;
    bool worldCandidate = false;
    bool worldVisibilityQueued = false;
    uint32_t sourceCount = 0;
    bool dirty = true;
    bool desiredVisible = true;
    bool visible = false;
    uint32_t orderedFrame = ~0u;
  };
  SceneManager& m_scene;
  std::unordered_map<uint64_t, Entry> m_entries;
  std::unordered_multimap<uint64_t, uint64_t> m_assetUsers;
  std::vector<uint64_t> m_dirty;
  std::vector<uint64_t> m_visibilityDirty;
  std::vector<uint64_t> m_ordered;
  // Multi-surface unordered parents only; no catalog walk in an ordinary stable frame.
  std::unordered_set<uint64_t> m_batchCandidates;
  RetainedStaticBatchCache m_batchCache;
  struct WorldView {
    uint64_t handle = 0, revision = 0;
    bool built = false;
  };
  RetainedWorldBatcher m_worldBatcher;
  std::unordered_map<uint64_t, WorldView> m_worldViews;
  std::unordered_set<uint64_t> m_worldOwnedChildren;
  bool m_worldDirty = true;
  retained_compat::WorldBatchUpdateState m_worldUpdates;
  retained_compat::WorldBatchAdmissionState m_worldAdmission;
  enum class WorldReason : size_t { NewCandidate, Remove, FirstMove, Description, ChildBuild,
    AssetGlobal, AssetIndexed, Texture, SceneClear, Options, Wetness, Identity, QueueFailure, Count };
  std::array<uint64_t, size_t(WorldReason::Count)> m_worldReasons {};
  std::unordered_map<uint64_t, uint64_t> m_worldDetach;
  std::unordered_set<uint64_t> m_publishedRefresh;
  bool m_worldGlobalPending = false;
  uint64_t m_worldRefreshRevision = 0;
  uint64_t m_worldDetachedChildren = 0, m_worldGlobalRestores = 0;
  std::vector<uint64_t> m_worldVisibilityDirty;
  uint64_t m_worldFullCalls = 0, m_worldFastCalls = 0;
  uint64_t m_worldStructureEvents = 0, m_worldVisibilityEvents = 0;
  uint64_t m_worldBuilds = 0, m_worldFailures = 0;
  struct WorldCpuStats {
    uint64_t calls = 0;
    double snapshotTotal = 0, prepareTotal = 0, publishTotal = 0;
    double snapshotMax = 0, prepareMax = 0, publishMax = 0;
  };
  WorldCpuStats m_worldFullCpu, m_worldFastCpu;
  uint32_t m_worldFullLogFrame = 0, m_worldFastLogFrame = 0;
  bool m_identityToolsInactive = false;
  bool m_geometryAggregation = true;
  uint64_t m_fusedBuilds = 0;
  uint64_t m_sourceChecks = 0;
  uint64_t m_pairChecks = 0;
  struct OrderedStats {
    uint64_t calls = 0, ids = 0, cleanReuses = 0, rebuilds = 0, failures = 0;
  } m_orderedStats;
  uint32_t m_batchRejectSamples = 0;
  RetainedBatchRejectCounts m_sourceRejects {};
  RetainedBatchRejectCounts m_pairRejects {};
  std::vector<std::pair<uint32_t, uint64_t>> m_immediate;
  remixapi_RetainedStatus m_status {};
  uint64_t m_nextHandle = uint64_t(1) << 60;
  uint64_t m_assetEpoch = ~0ull;
  uint32_t m_textureGeneration = ~0u;
  uint64_t m_clearEpoch = ~0ull;
  uint64_t m_options = ~0ull;
  uint32_t m_commitFrame = ~0u;
  uint32_t m_visibleUnorderedCount = 0;
  bool m_active = false;
  bool checkEpoch(uint64_t epoch);
  void fail();
  void eraseChildren(Entry& entry);
  bool prepare(const Rc<DxvkContext>& ctx, const ExternalDrawState& source,
               std::vector<ExternalDrawState>* pStates);
  bool build(const Rc<DxvkContext>& ctx, Entry& entry, bool refreshPublished = false);
  void fuseOriginals(const Rc<DxvkContext>& ctx, uint64_t id, uint32_t flags, std::vector<ExternalDrawState>& states);
  void rejectBatchSample(uint64_t parent, const ExternalDrawState& state, const char* phase, RetainedBatchReason reason);
  void invalidate();
  void indexAssets(Entry& entry);
  void unindexAssets(Entry& entry);
  void visible(Entry& entry, bool value);
  void requestWorldAdmission(WorldReason reason);
  void changeWorldEntry(Entry& entry, WorldReason reason) noexcept;
  void invalidateWorld(WorldReason reason) noexcept;
  void syncPublishedTransform(Entry& entry);
  void queueWorldVisibility(Entry& entry) noexcept;
  void clearWorldVisibility();
  bool publishWorldVisibility(const Rc<DxvkContext>& ctx, std::vector<RetainedWorldBatcher::Plan>& plans);
  void hideWorldBatches();
  void restoreWorldChildren(bool clearCache);
  void reconcileWorldBatches(const Rc<DxvkContext>& ctx);
};

}
