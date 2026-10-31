// Fixed-width catalog compatibility, adapted from the local rebased9-19 runtime.
// The direct scene owns all persistent instances and runs GPU instancing unchanged.
#include "rtx_retained_compat.h"

#include <algorithm>
#include <remix/remix_legacy_surface_validation.h>
#include "dxvk_device.h"
#include "rtx_asset_replacer.h"
#include "rtx_context.h"
#include "rtx_fork_hooks.h"
#include "rtx_options.h"
#include "rtx_option_manager.h"
#include "rtx_retained_scene.h"
#include "rtx_texture_manager.h"

namespace dxvk {

RetainedCompat::~RetainedCompat() = default;

void RetainedCompat::fail() {
  m_status.lastError = REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  // A failed group must not leave an already-built prefix visible.
  for (auto& [id, entry] : m_entries) {
    visible(entry, false);
  }
  m_active = false;
}

bool RetainedCompat::checkEpoch(uint64_t epoch) {
  if (!epoch || epoch != m_status.epoch) {
    Logger::err(str::format("[RetainedCompat] epoch mismatch: received ", epoch, ", current ", m_status.epoch));
    fail();
    return false;
  }
  return true;
}

void RetainedCompat::eraseChildren(Entry& entry) {
  for (uint64_t handle : entry.children) {
    m_scene.retainedScene().destroy(handle);
  }
  entry.children.clear();
  entry.visible = false;
}

void RetainedCompat::visible(Entry& entry, bool value) {
  if (entry.visible == value) {
    return;
  }
  for (uint64_t handle : entry.children) {
    m_scene.retainedScene().setVisible(handle, value);
  }
  entry.visible = value;
}

void RetainedCompat::unindexAssets(Entry& entry) {
  for (uint64_t asset : entry.assets) {
    auto range = m_assetUsers.equal_range(asset);
    for (auto it = range.first; it != range.second;) {
      if (it->second == entry.record.id) {
        it = m_assetUsers.erase(it);
      } else {
        ++it;
      }
    }
  }
  entry.assets.clear();
}

void RetainedCompat::indexAssets(Entry& entry) {
  unindexAssets(entry);
  entry.assets.push_back(reinterpret_cast<uint64_t>(entry.state->mesh));
  if (auto surfaces = m_scene.getAssetReplacer()->accessExternalMesh(entry.state->mesh)) {
    for (const auto& surface : *surfaces) {
      entry.assets.push_back(reinterpret_cast<uint64_t>(surface.externalMaterial));
    }
  }
  std::sort(entry.assets.begin(), entry.assets.end());
  entry.assets.erase(std::unique(entry.assets.begin(), entry.assets.end()), entry.assets.end());
  for (uint64_t asset : entry.assets) {
    m_assetUsers.emplace(asset, entry.record.id);
  }
}

void RetainedCompat::upsert(uint64_t epoch, std::vector<remixapi_RetainedInstance>&& records,
                           std::vector<frontend::PreparedInstance>&& states) {
  if (!checkEpoch(epoch) || records.size() != states.size()) {
    return;
  }
  // Every state is validated before any record is changed.
  for (const auto& state : states) {
    if (!state) {
      fail();
      return;
    }
  }
  for (size_t i = 0; i < records.size(); ++i) {
    const auto& record = records[i];
    auto [found, inserted] = m_entries.try_emplace(record.id);
    Entry& entry = found->second;
    unindexAssets(entry);
    if (!inserted && !(entry.record.flags & REMIXAPI_RETAINED_ORDERED)) {
      --m_unorderedCount;
    }
    if (!(record.flags & REMIXAPI_RETAINED_ORDERED)) {
      ++m_unorderedCount;
    }
    if (inserted || ((entry.record.flags ^ record.flags) & REMIXAPI_RETAINED_ORDERED)) {
      eraseChildren(entry);
      m_ordered.erase(std::remove(m_ordered.begin(), m_ordered.end(), record.id), m_ordered.end());
      entry.orderedFrame = ~0u;
    }
    entry.record = record;
    entry.state = std::move(states[i].m_state);
    indexAssets(entry);
    entry.dirty = true;
    m_dirty.push_back(record.id);
  }
  m_status.instanceCount = static_cast<uint32_t>(m_entries.size());
}

void RetainedCompat::remove(uint64_t epoch, const std::vector<uint64_t>& ids) {
  if (!checkEpoch(epoch)) {
    return;
  }
  for (uint64_t id : ids) {
    auto found = m_entries.find(id);
    if (found == m_entries.end()) {
      continue;
    }
    if (!(found->second.record.flags & REMIXAPI_RETAINED_ORDERED)) {
      --m_unorderedCount;
    }
    unindexAssets(found->second);
    eraseChildren(found->second);
    m_entries.erase(found);
  }
  m_status.instanceCount = static_cast<uint32_t>(m_entries.size());
}

void RetainedCompat::clear(uint64_t epoch) {
  if (!epoch || epoch <= m_status.epoch) {
    fail();
    return;
  }
  for (auto& [id, entry] : m_entries) {
    eraseChildren(entry);
  }
  m_entries.clear();
  m_assetUsers.clear();
  m_dirty.clear();
  m_ordered.clear();
  m_unorderedCount = 0;
  m_commitFrame = ~0u;
  m_active = false;
  m_status = {};
  m_status.epoch = epoch;
}

bool RetainedCompat::hasLegacy(const ExternalDrawState& state) const {
  auto surfaces = m_scene.getAssetReplacer()->accessExternalMesh(state.mesh);
  return surfaces && !surfaces->empty() && surfaces->front().legacyDrawHash;
}

bool RetainedCompat::prepare(const Rc<DxvkContext>& ctx, const ExternalDrawState& source,
                             std::vector<ExternalDrawState>& states) {
  RTX_OPTION_INVALIDATION_SCOPE(RtxOptionFlags::InvalidatesDrawcallTranslation);
  auto& assets = *m_scene.getAssetReplacer();
  const auto surfaces = assets.accessExternalMesh(source.mesh);
  if (!surfaces || surfaces->empty()) {
    Logger::err(str::format("[RetainedCompat] mesh unavailable: ", source.mesh));
    return false;
  }
  const uint64_t rule = uint64_t(RtxOptions::geometryAssetHashRule().raw()) |
    (uint64_t(RtxOptions::geometryHashGenerationRule().raw()) << 32);
  frontend::SceneInterface scene(*static_cast<RtxContext*>(ctx.ptr()));
  for (const auto& geometry : *surfaces) {
    const MaterialData* material = assets.accessExternalMaterial(geometry.externalMaterial);
    if (geometry.externalMaterial && !material) {
      Logger::err(str::format("[RetainedCompat] material unavailable: ", geometry.externalMaterial, ", mesh ", source.mesh));
      return false;
    }
    ExternalDrawState state = source;
    if (geometry.legacyDrawHash) {
      if (geometry.legacyRuleFingerprint != rule || !material || material->getType() != MaterialDataType::Opaque) {
        Logger::err(str::format("[RetainedCompat] incompatible legacy surface: mesh ", source.mesh, ", captured rule ", geometry.legacyRuleFingerprint, ", current rule ", rule, ", material present ", material != nullptr));
        return false;
      }
      RasterGeometry original = geometry;
      original.cullMode = state.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
      state.inlineSubmeshes = std::make_shared<const std::vector<RasterGeometry>>(1, original);
      state.drawCall.overrideGeometryData(std::shared_ptr<const RasterGeometry>(state.inlineSubmeshes, &state.inlineSubmeshes->front()));
      auto& legacy = state.drawCall.modifyMaterialData();
      const auto& opaque = material->getOpaqueMaterialData();
      legacy.setColorTexture(0, opaque.getAlbedoOpacityTexture(), nullptr, 0);
      legacy.setHashOverride(geometry.legacyMaterialHash);
      const XXH64_hash_t textureHash = legacy.getColorTexture().getImageHash();
      // Keep the captured texture-list categories identical to native D3D9, including hash zero.
      auto& categories = state.drawCall.modifyCategoryFlags();
      const auto setIf = [&](InstanceCategories category, bool enabled) {
        if (enabled) {
          categories.set(category);
        }
      };
      setIf(InstanceCategories::WorldUI, RtxOptions::worldSpaceUiTextures().count(textureHash) != 0);
      setIf(InstanceCategories::WorldMatte, RtxOptions::worldSpaceUiBackgroundTextures().count(textureHash) != 0);
  
      setIf(InstanceCategories::Ignore, RtxOptions::ignoreTextures().count(textureHash) != 0);
      setIf(InstanceCategories::IgnoreLights, RtxOptions::ignoreLights().count(textureHash) != 0);
      setIf(InstanceCategories::IgnoreAntiCulling, RtxOptions::antiCullingTextures().count(textureHash) != 0);
      setIf(InstanceCategories::IgnoreMotionBlur, RtxOptions::motionBlurMaskOutTextures().count(textureHash) != 0);
      setIf(InstanceCategories::IgnoreOpacityMicromap, RtxOptions::opacityMicromapIgnoreTextures().count(textureHash) != 0);
      setIf(InstanceCategories::IgnoreAlphaChannel, RtxOptions::ignoreAlphaOnTextures().count(textureHash) != 0);
      setIf(InstanceCategories::IgnoreBakedLighting, RtxOptions::ignoreBakedLightingTextures().count(textureHash) != 0);
  
      setIf(InstanceCategories::Hidden, RtxOptions::hideInstanceTextures().count(textureHash) != 0);
  
      setIf(InstanceCategories::Particle, RtxOptions::particleTextures().count(textureHash) != 0);
      setIf(InstanceCategories::Beam, RtxOptions::beamTextures().count(textureHash) != 0);
  
      setIf(InstanceCategories::DecalStatic, RtxOptions::decalTextures().count(textureHash) != 0);
      setIf(InstanceCategories::DecalDynamic, RtxOptions::dynamicDecalTextures().count(textureHash) != 0);
      setIf(InstanceCategories::DecalSingleOffset, RtxOptions::singleOffsetDecalTextures().count(textureHash) != 0);
      setIf(InstanceCategories::DecalNoOffset, RtxOptions::nonOffsetDecalTextures().count(textureHash) != 0);
  
      setIf(InstanceCategories::AnimatedWater, RtxOptions::animatedWaterTextures().count(textureHash) != 0);
  
      setIf(InstanceCategories::ThirdPersonPlayerModel, RtxOptions::playerModelTextures().count(textureHash) != 0);
      setIf(InstanceCategories::ThirdPersonPlayerBody, RtxOptions::playerModelBodyTextures().count(textureHash) != 0);
  
      setIf(InstanceCategories::Terrain, RtxOptions::terrainTextures().count(textureHash) != 0);
      setIf(InstanceCategories::Sky, RtxOptions::skyBoxTextures().count(textureHash) != 0);
  
      setIf(InstanceCategories::ParticleEmitter, RtxOptions::particleEmitterTextures().count(textureHash) != 0);
      setIf(InstanceCategories::HairCards, RtxOptions::hairCardTextures().count(textureHash) != 0);
      setIf(InstanceCategories::SmoothNormals, RtxOptions::smoothNormalsTextures().count(textureHash) != 0);
      setIf(InstanceCategories::Sky, RtxOptions::skyBoxGeometries().count(geometry.legacyGeometryHash) != 0);
      state.categories = state.drawCall.getCategoryFlags();
      LegacyMaterialData lookupMaterial = legacy;
      // Texture replacements use the original game texture key; capture keeps the observed draw material key.
      lookupMaterial.setHashOverride(textureHash);
      state.resolvedMaterial = std::make_shared<const MaterialData>(scene.resolveMaterial(lookupMaterial));
      // Authored API sampling is the only sampler information in the old capture contract.
      if (!assets.getReplacementMaterial(textureHash)) {
        MaterialData resolved = *state.resolvedMaterial;
        if (resolved.getType() == MaterialDataType::Opaque) {
          auto& resolvedOpaque = resolved.getOpaqueMaterialData();
          resolvedOpaque.setSamplerOverride(nullptr);
          resolvedOpaque.setFilterMode(opaque.getFilterMode());
          resolvedOpaque.setWrapModeU(opaque.getWrapModeU());
          resolvedOpaque.setWrapModeV(opaque.getWrapModeV());
        }
        state.resolvedMaterial = std::make_shared<const MaterialData>(std::move(resolved));
      }
      state.mesh = reinterpret_cast<remixapi_MeshHandle>(geometry.hashes[HashComponents::VertexPosition]);
      state.frontendResolved = true;
      state.assetKeys = frontend::AssetKeys {};
      state.assetKeys->mesh = geometry.legacyDrawHash;
      state.drawCall.wetnessMeshHash = geometry.legacyDrawHash;
      states.push_back(std::move(state));
    } else {
      // A normal API mesh stays a single description, including its GPU transform buffer.
      states.push_back(source);
      return true;
    }
  }
  return true;
}

bool RetainedCompat::build(const Rc<DxvkContext>& ctx, Entry& entry) {
  std::vector<ExternalDrawState> states;
  if (!prepare(ctx, *entry.state, states)) {
    eraseChildren(entry);
    fail();
    return false;
  }
  indexAssets(entry);
  auto& retained = m_scene.retainedScene();
  while (entry.children.size() > states.size()) {
    retained.destroy(entry.children.back());
    entry.children.pop_back();
  }
  while (entry.children.size() < states.size()) {
    entry.children.push_back(++m_nextHandle);
  }
  for (size_t i = 0; i < states.size(); ++i) {
    // A previous missing CommitFrame may have hidden this direct-scene entry.
    retained.setVisible(entry.children[i], true);
    if (!retained.buildNow(ctx, entry.children[i], std::move(states[i]), false)) {
      Logger::err(str::format("[RetainedCompat] build rejected: record ", entry.record.id, ", mesh ", entry.state->mesh, ", frame ", m_scene.device()->getCurrentFrameId()));
      eraseChildren(entry);
      fail();
      return false;
    }
  }
  entry.dirty = false;
  entry.visible = true;
  return true;
}

void RetainedCompat::invalidate() {
  auto& assets = *m_scene.getAssetReplacer();
  const uint64_t assetEpoch = assets.getExternalAssetGlobalEpoch();
  const uint64_t clearEpoch = m_scene.clearEpoch();
  const uint32_t textureGeneration = m_scene.device()->getCommon()->getTextureManager().getTextureCacheGeneration();
  const uint64_t options = RtxOptionManager::valueChangeGeneration();
  const bool all = assetEpoch != m_assetEpoch || clearEpoch != m_clearEpoch ||
    textureGeneration != m_textureGeneration ||
    (options != m_options && RtxOptionManager::isDrawcallTranslationInvalid());
  m_assetEpoch = assetEpoch;
  m_clearEpoch = clearEpoch;
  m_textureGeneration = textureGeneration;
  m_options = options;
  const auto dirty = [&](uint64_t id, Entry& entry) {
    if (!entry.dirty) {
      entry.dirty = true;
      m_dirty.push_back(id);
    }
  };
  if (all) {
    for (auto& [id, entry] : m_entries) {
      dirty(id, entry);
    }
  }
  m_scene.consumeExternalAssetChanges();
}

void RetainedCompat::invalidateAsset(uint64_t asset) {
  auto range = m_assetUsers.equal_range(asset);
  for (auto it = range.first; it != range.second; ++it) {
    auto found = m_entries.find(it->second);
    if (found != m_entries.end() && !found->second.dirty) {
      found->second.dirty = true;
      m_dirty.push_back(found->first);
    }
  }
}

void RetainedCompat::commit(const Rc<DxvkContext>& ctx, uint64_t epoch) {
  if (!checkEpoch(epoch) || m_status.lastError) {
    return;
  }
  const uint32_t frame = m_scene.device()->getCurrentFrameId();
  if (m_commitFrame == frame) {
    return;
  }
  invalidate();
  for (uint64_t id : m_dirty) {
    auto found = m_entries.find(id);
    if (found != m_entries.end() && found->second.dirty && !(found->second.record.flags & REMIXAPI_RETAINED_ORDERED)) {
      if (!build(ctx, found->second)) {
        return;
      }
    }
  }
  m_dirty.clear();
  if (!m_active) {
    for (auto& [id, entry] : m_entries) {
      if (!(entry.record.flags & REMIXAPI_RETAINED_ORDERED)) {
        visible(entry, true);
      }
    }
  }
  m_active = true;
  m_commitFrame = frame;
  ++m_status.commitCount;
  m_status.submittedCount += m_unorderedCount;
  m_status.lastCommitFrame = frame;
}

void RetainedCompat::drawOrdered(const Rc<DxvkContext>& ctx, uint64_t epoch, const std::vector<uint64_t>& ids) {
  if (!checkEpoch(epoch) || m_status.lastError) {
    return;
  }
  invalidate();
  // Reject the whole group before touching order or visibility.
  for (uint64_t id : ids) {
    auto found = m_entries.find(id);
    std::vector<ExternalDrawState> checked;
    if (found == m_entries.end() || !(found->second.record.flags & REMIXAPI_RETAINED_ORDERED) ||
        !prepare(ctx, *found->second.state, checked)) {
      Logger::err(str::format("[RetainedCompat] ordered group rejected: record ", id, ", present ", found != m_entries.end(), ", epoch ", epoch));
      fail();
      return;
    }
  }
  const uint32_t frame = m_scene.device()->getCurrentFrameId();
  for (uint64_t id : ids) {
    Entry& entry = m_entries.at(id);
    if (entry.dirty && !build(ctx, entry)) {
      return;
    }
    visible(entry, true);
    for (uint64_t handle : entry.children) {
      m_scene.retainedScene().setSortOrder(handle, m_scene.nextDecalSortOrder());
      m_scene.retainedScene().markDrawn(ctx, handle);
    }
    if (entry.orderedFrame == ~0u) {
      m_ordered.push_back(id);
    }
    entry.orderedFrame = frame;
    ++m_status.submittedCount;
  }
}

void RetainedCompat::drawImmediate(const Rc<DxvkContext>& ctx, std::unique_ptr<ExternalDrawState> state) {
  std::vector<ExternalDrawState> states;
  if (!prepare(ctx, *state, states)) {
    return;
  }
  const uint32_t frame = m_scene.device()->getCurrentFrameId();
  for (auto& prepared : states) {
    const uint64_t handle = ++m_nextHandle;
    m_scene.retainedScene().buildNow(ctx, handle, std::move(prepared), false);
    m_scene.retainedScene().setSortOrder(handle, m_scene.nextDecalSortOrder());
    m_immediate.emplace_back(frame, handle);
  }
}

void RetainedCompat::finishFrame(const Rc<DxvkContext>& ctx) {
  const uint32_t frame = m_scene.device()->getCurrentFrameId();
  // Normal successive commits only touch this scalar. Visibility changes are events.
  if (m_active && (m_commitFrame != frame || m_status.lastError)) {
    for (auto& [id, entry] : m_entries) {
      if (!(entry.record.flags & REMIXAPI_RETAINED_ORDERED)) {
        visible(entry, false);
      }
    }
    m_active = false;
  }
  m_ordered.erase(std::remove_if(m_ordered.begin(), m_ordered.end(), [&](uint64_t id) {
    auto found = m_entries.find(id);
    if (found == m_entries.end()) {
      return true;
    }
    if (found->second.orderedFrame != frame || m_status.lastError) {
      visible(found->second, false);
      found->second.orderedFrame = ~0u;
      return true;
    }
    return false;
  }), m_ordered.end());
  m_immediate.erase(std::remove_if(m_immediate.begin(), m_immediate.end(), [&](const auto& item) {
    if (item.first == frame) {
      return false;
    }
    m_scene.retainedScene().destroy(item.second);
    return true;
  }), m_immediate.end());
}

void RetainedCompat::destroyMesh(remixapi_MeshHandle mesh) {
  std::vector<uint64_t> ids;
  for (const auto& [id, entry] : m_entries) {
    if (entry.state->mesh == mesh) {
      ids.push_back(id);
    }
  }
  if (!ids.empty()) {
    remove(m_status.epoch, ids);
  }
}

}
