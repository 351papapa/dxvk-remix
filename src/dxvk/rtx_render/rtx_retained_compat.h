#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <remix/remix_retained.h>
#include "rtx_frontend_interface.h"
#include "rtx_scene_manager.h"

namespace dxvk {

// The versioned fixed-width host catalog. Rendering belongs to RetainedScene.
class RetainedCompat {
public:
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
  remixapi_RetainedStatus status() const { return m_status; }
  bool hasLegacy(const ExternalDrawState& state) const;

private:
  struct Entry {
    remixapi_RetainedInstance record {};
    std::unique_ptr<ExternalDrawState> state;
    std::vector<uint64_t> children;
    std::vector<uint64_t> assets;
    bool dirty = true;
    bool visible = false;
    uint32_t orderedFrame = ~0u;
  };
  SceneManager& m_scene;
  std::unordered_map<uint64_t, Entry> m_entries;
  std::unordered_multimap<uint64_t, uint64_t> m_assetUsers;
  std::vector<uint64_t> m_dirty;
  std::vector<uint64_t> m_ordered;
  std::vector<std::pair<uint32_t, uint64_t>> m_immediate;
  remixapi_RetainedStatus m_status {};
  uint64_t m_nextHandle = uint64_t(1) << 60;
  uint64_t m_assetEpoch = ~0ull;
  uint64_t m_assetGeneration = ~0ull;
  uint32_t m_textureGeneration = ~0u;
  uint64_t m_clearEpoch = ~0ull;
  uint64_t m_options = ~0ull;
  uint32_t m_commitFrame = ~0u;
  uint32_t m_unorderedCount = 0;
  bool m_active = false;
  bool checkEpoch(uint64_t epoch);
  void fail();
  void eraseChildren(Entry& entry);
  bool prepare(const Rc<DxvkContext>& ctx, const ExternalDrawState& source,
               std::vector<ExternalDrawState>& states);
  bool build(const Rc<DxvkContext>& ctx, Entry& entry);
  void invalidate();
  void indexAssets(Entry& entry);
  void unindexAssets(Entry& entry);
  void visible(Entry& entry, bool value);
};

}
