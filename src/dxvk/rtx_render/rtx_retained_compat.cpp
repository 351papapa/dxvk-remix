// Fixed-width catalog compatibility, adapted from the local rebased9-19 runtime.
// The direct scene owns all persistent instances and runs GPU instancing unchanged.
#include "rtx_retained_compat.h"
#include "rtx_retained_compat_policy.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <remix/remix_legacy_surface_validation.h>
#include <remix/remix_retained_validation.h>
#include "dxvk_device.h"
#include "rtx_asset_replacer.h"
#include "rtx_context.h"
#include "rtx_fork_hooks.h"
#include "rtx_options.h"
#include "rtx_option_manager.h"
#include "rtx_retained_scene.h"
#include "rtx_retained_static_batch.h"
#include "rtx_game_capturer.h"
#include "rtx_debug_view.h"
#include "rtx_texture_manager.h"

namespace dxvk {

RetainedCompat::~RetainedCompat() = default;

void RetainedCompat::fail() {
  m_status.lastError = REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  // A failed group must not leave an already-built prefix visible.
  for (auto& [id, entry] : m_entries) {
    visible(entry, false);
  }
  hideWorldBatches();
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
  changeWorldEntry(entry, WorldReason::Remove);
  for (uint64_t handle : entry.children) {
    m_scene.retainedScene().destroy(handle);
  }
  entry.children.clear();
  entry.worldParts.clear();
  entry.worldCandidate = false;
  entry.visible = false;
}

void RetainedCompat::visible(Entry& entry, bool value) {
  retained_compat::applyVisibility(entry, value, [&](bool target) {
    queueWorldVisibility(entry);
    for (uint64_t handle : entry.children) {
      m_scene.retainedScene().setVisible(handle, target && !m_worldOwnedChildren.count(handle));
    }
  });
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
  const auto add = [&](const std::shared_ptr<const ExternalDrawState>& source) {
    if (!source) {
      return;
    }
    entry.assets.push_back(reinterpret_cast<uint64_t>(source->mesh));
    if (auto surfaces = m_scene.getAssetReplacer()->accessExternalMesh(source->mesh)) {
      for (const auto& surface : *surfaces) {
        entry.assets.push_back(reinterpret_cast<uint64_t>(surface.externalMaterial));
      }
    }
  };
  add(entry.state);
  if (entry.published.source != entry.state) {
    add(entry.published.source);
  }
  std::sort(entry.assets.begin(), entry.assets.end());
  entry.assets.erase(std::unique(entry.assets.begin(), entry.assets.end()), entry.assets.end());
  for (uint64_t asset : entry.assets) {
    m_assetUsers.emplace(asset, entry.record.id);
  }
}

void RetainedCompat::syncPublishedTransform(Entry& entry) {
  if (!entry.published.source) {
    return;
  }
  entry.published.updateApplied(entry.state, [](ExternalDrawState& published, const ExternalDrawState& current) {
    auto& transforms = published.drawCall.modifyTransformData();
    transforms.objectToWorld = current.drawCall.getTransformData().objectToWorld;
    transforms.objectToView = transforms.worldToView * transforms.objectToWorld;
  });
  entry.published.record.transform = entry.record.transform;
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
    // A placement that has moved uses its canonical children for the rest of this record's lifetime.
    // This prevents moving props from repeatedly rebuilding static cell geometry.
    if (!inserted && !entry.everMoved && !remixapi_retained::sameTransform(entry.record, record)) {
      changeWorldEntry(entry, WorldReason::FirstMove);
      entry.everMoved = true;
      entry.worldParts.clear();
      entry.worldCandidate = false;
    }
    if (!inserted && !(record.flags & REMIXAPI_RETAINED_ORDERED) &&
        remixapi_retained::sameDescription(entry.record, record)) {
      const bool moved = !remixapi_retained::sameTransform(entry.record, record);
      const bool desired = !(record.flags & REMIXAPI_RETAINED_HIDDEN);
      if (entry.desiredVisible != desired) {
        m_visibleUnorderedCount = desired ? m_visibleUnorderedCount + 1 : m_visibleUnorderedCount - 1;
        entry.desiredVisible = desired;
        m_visibilityDirty.push_back(record.id);
        visible(entry, m_active && m_commitFrame == m_scene.device()->getCurrentFrameId());
      }
      entry.record = record;
      entry.state = std::move(states[i].m_state);
      if (moved) {
        syncPublishedTransform(entry);
        const Matrix4& transform = entry.state->drawCall.getTransformData().objectToWorld;
        for (uint64_t handle : entry.children) {
          m_scene.retainedScene().setTransform(handle, transform, true);
        }
      }
      continue;
    }
    changeWorldEntry(entry, WorldReason::Description);
    entry.published.stage();
    unindexAssets(entry);
    if (!inserted && remixapi_retained::visibleUnordered(entry.record)) {
      --m_visibleUnorderedCount;
    }
    if (remixapi_retained::visibleUnordered(record)) {
      ++m_visibleUnorderedCount;
    }
    if (inserted || ((entry.record.flags ^ record.flags) & REMIXAPI_RETAINED_ORDERED)) {
      eraseChildren(entry);
      m_ordered.erase(std::remove(m_ordered.begin(), m_ordered.end(), record.id), m_ordered.end());
      entry.orderedFrame = ~0u;
    }
    entry.record = record;
    entry.desiredVisible = !(record.flags & REMIXAPI_RETAINED_HIDDEN);
    if (!entry.desiredVisible) {
      visible(entry, false);
    }
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
    if (remixapi_retained::visibleUnordered(found->second.record)) {
      --m_visibleUnorderedCount;
    }
    unindexAssets(found->second);
    m_batchCandidates.erase(id);
    m_publishedRefresh.erase(id);
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
  restoreWorldChildren(true);
  for (auto& [id, entry] : m_entries) {
    eraseChildren(entry);
  }
  m_entries.clear();
  m_assetUsers.clear();
  m_dirty.clear();
  m_visibilityDirty.clear();
  m_ordered.clear();
  m_batchCandidates.clear();
  m_batchCache.clear();
  m_fusedBuilds = 0;
  m_worldBuilds = m_worldFailures = 0;
  m_worldFullCalls = m_worldFastCalls = 0;
  m_worldFullCpu = m_worldFastCpu = {};
  m_worldFullLogFrame = m_worldFastLogFrame = 0;
  m_worldStructureEvents = m_worldVisibilityEvents = 0;
  m_worldUpdates = {};
  m_worldAdmission = {};
  m_worldDetach.clear();
  m_publishedRefresh.clear();
  m_worldGlobalPending = false;
  m_worldRefreshRevision = 0;
  m_worldDetachedChildren = m_worldGlobalRestores = 0;
  m_worldReasons = {};
  m_worldVisibilityDirty.clear();
  m_worldDirty = true;
  m_sourceChecks = m_pairChecks = 0;
  m_orderedStats = {};
  m_batchRejectSamples = 0;
  m_sourceRejects = {};
  m_pairRejects = {};
  m_visibleUnorderedCount = 0;
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
                             std::vector<ExternalDrawState>* pStates) {
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
    const auto check = retained_compat::checkSurface(geometry.externalMaterial != nullptr, material != nullptr,
      geometry.legacyDrawHash != 0, geometry.legacyRuleFingerprint, rule, material && material->getType() == MaterialDataType::Opaque);
    if (check == retained_compat::SurfaceCheck::MissingMaterial) {
      Logger::err(str::format("[RetainedCompat] material unavailable: ", geometry.externalMaterial, ", mesh ", source.mesh));
      return false;
    }
    if (check == retained_compat::SurfaceCheck::IncompatibleLegacy) {
      Logger::err(str::format("[RetainedCompat] incompatible legacy surface: mesh ", source.mesh, ", captured rule ", geometry.legacyRuleFingerprint, ", current rule ", rule, ", material present ", material != nullptr));
      return false;
    }
    if (!geometry.legacyDrawHash) {
      // A normal API mesh stays a single description, including its GPU transform buffer.
      if (pStates) {
        pStates->push_back(source);
        pStates = nullptr;
      }
      continue;
    }
    // Ordered replay validates live dependencies every call. Only a build needs material translation.
    if (pStates) {
      ExternalDrawState state = source;
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
      pStates->push_back(std::move(state));
    }
  }
  return true;
}

void RetainedCompat::rejectBatchSample(uint64_t parent, const ExternalDrawState& state,
                                      const char* phase, RetainedBatchReason reason) {
  if (m_batchRejectSamples >= 16) {
    return;
  }
  ++m_batchRejectSamples;
  Logger::info(str::format("[RetainedBatchReject] parent=", parent,
    " source=", state.assetKeys.has_value() ? state.assetKeys->mesh : 0,
    " phase=", phase, " reason=", RetainedStaticBatch::reasonName(reason)));
}

void RetainedCompat::fuseOriginals(const Rc<DxvkContext>& ctx, uint64_t id, uint32_t flags, std::vector<ExternalDrawState>& states) {
  if (states.size() < 2 || (flags & REMIXAPI_RETAINED_ORDERED)) {
    m_batchCandidates.erase(id);
    return;
  }
  m_batchCandidates.insert(id);
  if (!m_identityToolsInactive) {
    return;
  }
  // Only a disposable rendering view is merged. The API mesh, all original identities and
  // the dependency index stay intact, so a newly installed replacement resolves per source.
  try {
    std::vector<ExternalDrawState> rendered;
    std::vector<bool> used(states.size(), false);
    rendered.reserve(states.size());
    const auto original = [&](const ExternalDrawState& state) {
      ++m_sourceChecks;
      RetainedBatchReason reason = RetainedBatchReason::None;
      uint32_t matchedKey = 0;  // Required output storage, including a successful replacement match.
      if (!state.frontendResolved || !state.assetKeys.has_value()) {
        reason = RetainedBatchReason::Source;
      } else if (m_scene.findMeshReplacements(*state.assetKeys, &matchedKey)) {
        reason = RetainedBatchReason::Replacement;
      } else {
        reason = RetainedStaticBatch::rejectionReason(state, state);
      }
      if (reason != RetainedBatchReason::None) {
        ++m_sourceRejects[size_t(reason)];
        rejectBatchSample(id, state, "source", reason);
      }
      return reason == RetainedBatchReason::None;
    };
    for (size_t i = 0; i < states.size(); ++i) {
      if (used[i]) {
        continue;
      }
      used[i] = true;
      std::vector<size_t> members {i};
      if (original(states[i])) {
        const auto rule = m_scene.getWetnessMeshRule(states[i].drawCall.wetnessMeshHash);
        for (size_t j = i + 1; j < states.size() && members.size() < 64; ++j) {
          if (used[j] || !original(states[j])) {
            continue;
          }
          ++m_pairChecks;
          const RetainedBatchReason reason = m_scene.getWetnessMeshRule(states[j].drawCall.wetnessMeshHash) != rule
            ? RetainedBatchReason::Wetness : RetainedStaticBatch::rejectionReason(states[i], states[j]);
          if (reason != RetainedBatchReason::None) {
            ++m_pairRejects[size_t(reason)];
            rejectBatchSample(id, states[j], "pair", reason);
            continue;
          }
          members.push_back(j);
        }
      }
      ExternalDrawState merged;
      bool fused = false;
      if (members.size() > 1) {
        std::vector<ExternalDrawState> sources;
        sources.reserve(members.size());
        for (size_t member : members) {
          sources.push_back(states[member]);
        }
        RetainedStaticBatchProof proof;
        proof.parent = id;
        proof.apiOwnedOriginals = proof.noMeshReplacements = proof.sameWetnessRule = proof.identityToolsInactive = true;
        fused = RetainedStaticBatch::build(ctx, sources, proof, m_batchCache, merged);
      }
      if (fused) {
        for (size_t member : members) {
          used[member] = true;
        }
        rendered.push_back(std::move(merged));
      } else {
        rendered.push_back(states[i]);
      }
    }
    if (rendered.size() < states.size()) {
      ++m_fusedBuilds;
      states.swap(rendered);
    }
  } catch (const std::exception&) {
    // Fusion is optional. Failure keeps the complete original list, never a successful prefix.
  }
}

void RetainedCompat::refreshBatches(const Rc<DxvkContext>& ctx, bool wetnessRulesChanged) {
  // This path also runs before upload, after a host may have already committed this frame.
  // Asset callbacks only enqueue dirtiness; consuming their taken batch cannot recurse here.
  invalidate();
  auto& common = *m_scene.device()->getCommon();
  auto& debug = common.metaDebugView();
  const uint32_t frame = m_scene.device()->getCurrentFrameId();
  const bool inactive = !common.capturer()->requiresOriginalSurfaceIdentities() &&
    !debug.ObjectPicking.containsRequests() && !debug.Highlighting.active(frame) &&
    !common.getResources().getRaytracingOutput().m_primaryObjectPicking.isValid();
  if (wetnessRulesChanged) {
    m_batchCache.clear();
    invalidateWorld(WorldReason::Wetness);
  }
  if (inactive != m_identityToolsInactive) {
    invalidateWorld(WorldReason::Identity);
  }
  m_identityToolsInactive = inactive;
  reconcileWorldBatches(ctx);
}

bool RetainedCompat::build(const Rc<DxvkContext>& ctx, Entry& entry, bool refreshPublished) {
  const auto source = refreshPublished ? entry.published.source : entry.state;
  if (!source) {
    return true;
  }
  const auto& record = refreshPublished ? entry.published.record : entry.record;
  const bool effectiveVisible = entry.visible;
  std::vector<ExternalDrawState> states;
  if (!prepare(ctx, *source, &states)) {
    eraseChildren(entry);
    fail();
    return false;
  }
  entry.sourceCount = static_cast<uint32_t>(states.size());
  fuseOriginals(ctx, record.id, record.flags, states);
  changeWorldEntry(entry, WorldReason::ChildBuild);
  auto& retained = m_scene.retainedScene();
  while (entry.children.size() > states.size()) {
    retained.destroy(entry.children.back());
    entry.children.pop_back();
  }
  while (entry.children.size() < states.size()) {
    entry.children.push_back(++m_nextHandle);
  }
  const bool wasWorldCandidate = entry.worldCandidate;
  entry.worldCandidate = false;
  entry.worldParts.clear();
  entry.worldParts.resize(states.size());
  for (size_t i = 0; i < states.size(); ++i) {
    if (!entry.everMoved && !(record.flags & REMIXAPI_RETAINED_ORDERED)) {
      // This optional view retains no authority over the canonical replacement description.
      try { retained.resolveStaticParts(states[i], entry.worldParts[i]); }
      catch (const std::exception&) { entry.worldParts[i].clear(); }
      entry.worldCandidate = entry.worldCandidate || !entry.worldParts[i].empty();
    }
    if (!retained.buildNow(ctx, entry.children[i], std::move(states[i]), false)) {
      Logger::err(str::format("[RetainedCompat] build rejected: record ", entry.record.id, ", mesh ", entry.state->mesh, ", frame ", m_scene.device()->getCurrentFrameId()));
      eraseChildren(entry);
      fail();
      return false;
    }
  }
  if (!wasWorldCandidate && entry.worldCandidate) {
    requestWorldAdmission(WorldReason::NewCandidate);
  }
  if (refreshPublished) {
    for (uint64_t handle : entry.children) {
      retained.setVisible(handle, effectiveVisible && !m_worldOwnedChildren.count(handle));
    }
    entry.dirty = entry.published.pending;
  } else {
    retained_compat::finishBuild(entry, [&](bool target) {
      for (uint64_t handle : entry.children) {
        retained.setVisible(handle, target && !m_worldOwnedChildren.count(handle));
      }
    });
    entry.published.publish(entry.state, entry.record);
    entry.dirty = false;
  }
  entry.publishedRefreshRevision = m_worldRefreshRevision;
  m_publishedRefresh.erase(entry.record.id);
  indexAssets(entry);
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
  if (assetEpoch != m_assetEpoch) {
    invalidateWorld(WorldReason::AssetGlobal);
  }
  if (clearEpoch != m_clearEpoch) {
    invalidateWorld(WorldReason::SceneClear);
  }
  if (textureGeneration != m_textureGeneration) {
    invalidateWorld(WorldReason::Texture);
  }
  if (options != m_options && RtxOptionManager::isDrawcallTranslationInvalid()) {
    invalidateWorld(WorldReason::Options);
  }
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
    m_batchCache.clear();
    for (auto& [id, entry] : m_entries) {
      dirty(id, entry);
    }
  }
  m_scene.consumeExternalAssetChanges();
}

void RetainedCompat::invalidateAsset(uint64_t asset) {
  auto range = m_assetUsers.equal_range(asset);
  if (range.first == range.second) {
    return;  // New/unrelated API assets have no users and cannot invalidate this geometry cache.
  }
  m_batchCache.clear();
  for (auto it = range.first; it != range.second; ++it) {
    auto found = m_entries.find(it->second);
    if (found == m_entries.end()) {
      continue;
    }
    auto& entry = found->second;
    changeWorldEntry(entry, WorldReason::AssetIndexed);
    // Keep unrelated placements and their stable batches untouched by a single-handle update.
    if (entry.published.source) {
      try {
        m_publishedRefresh.insert(found->first);
      } catch (...) {
        invalidateWorld(WorldReason::QueueFailure);
      }
    }
    if (!entry.dirty) {
      entry.dirty = true;
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
  refreshBatches(ctx);
  for (uint64_t id : m_dirty) {
    auto found = m_entries.find(id);
    if (found != m_entries.end() && found->second.dirty && !(found->second.record.flags & REMIXAPI_RETAINED_ORDERED)) {
      if (!build(ctx, found->second)) {
        return;
      }
    }
  }
  m_dirty.clear();
  for (uint64_t id : m_visibilityDirty) {
    auto found = m_entries.find(id);
    if (found != m_entries.end() && !(found->second.record.flags & REMIXAPI_RETAINED_ORDERED)) {
      visible(found->second, true);
    }
  }
  m_visibilityDirty.clear();
  if (!m_active) {
    for (auto& [id, entry] : m_entries) {
      if (!(entry.record.flags & REMIXAPI_RETAINED_ORDERED)) {
        visible(entry, true);
      }
    }
  }
  m_active = true;
  m_commitFrame = frame;
  reconcileWorldBatches(ctx);
  ++m_status.commitCount;
  m_status.submittedCount += m_visibleUnorderedCount;
  m_status.lastCommitFrame = frame;
  if (frame % 240 == 0 && m_orderedStats.calls) {
    Logger::info(str::format("[RetainedOrdered] calls=", m_orderedStats.calls, " ids=", m_orderedStats.ids,
      " cleanReuses=", m_orderedStats.cleanReuses, " rebuilds=", m_orderedStats.rebuilds,
      " failures=", m_orderedStats.failures));
  }
  if (frame % 240 == 0 && !m_batchCandidates.empty()) {
    uint64_t sources = 0, children = 0;
    for (uint64_t id : m_batchCandidates) {
      const auto found = m_entries.find(id);
      if (found != m_entries.end()) {
        sources += found->second.sourceCount;
        children += found->second.children.size();
      }
    }
    Logger::info(str::format("[RetainedBatch] parents=", m_batchCandidates.size(),
      " sources=", sources, " children=", children, " fusedBuilds=", m_fusedBuilds,
      " identityToolsInactive=", m_identityToolsInactive,
      " cacheEntries=", m_batchCache.size(), " cacheBytes=", m_batchCache.bytes(),
      " cacheHits=", m_batchCache.hits(), " cacheMisses=", m_batchCache.misses(),
      " cacheCreates=", m_batchCache.creates(), " cacheEvictions=", m_batchCache.evictions(),
      " sourceChecks=", m_sourceChecks, " pairChecks=", m_pairChecks,
      " buildAttempts=", m_batchCache.buildAttempts()));
    std::string gates;
    const auto append = [&](const char* phase, const RetainedBatchRejectCounts& counts) {
      for (size_t i = 1; i < counts.size(); ++i) {
        if (counts[i]) {
          gates += str::format(" ", phase, ".", RetainedStaticBatch::reasonName(RetainedBatchReason(i)), "=", counts[i]);
        }
      }
    };
    append("source", m_sourceRejects);
    append("pair", m_pairRejects);
    append("build", m_batchCache.buildRejects());
    Logger::info(str::format("[RetainedBatchGates]", gates));
  }
}

void RetainedCompat::drawOrdered(const Rc<DxvkContext>& ctx, uint64_t epoch, const std::vector<uint64_t>& ids) {
  ++m_orderedStats.calls;
  m_orderedStats.ids += ids.size();
  if (!checkEpoch(epoch) || m_status.lastError) {
    ++m_orderedStats.failures;
    return;
  }
  invalidate();
  const uint32_t frame = m_scene.device()->getCurrentFrameId();
  const auto validate = [&](uint64_t id) {
    auto found = m_entries.find(id);
    if (found == m_entries.end() || !(found->second.record.flags & REMIXAPI_RETAINED_ORDERED) ||
        !prepare(ctx, *found->second.state, nullptr)) {
      Logger::err(str::format("[RetainedCompat] ordered group rejected: record ", id, ", present ", found != m_entries.end(), ", epoch ", epoch));
      fail();
      return false;
    }
    return true;
  };
  const auto draw = [&](uint64_t id) {
    Entry& entry = m_entries.at(id);
    if (entry.dirty) {
      ++m_orderedStats.rebuilds;
      if (!build(ctx, entry)) {
        return false;
      }
    } else {
      ++m_orderedStats.cleanReuses;
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
    return true;
  };
  if (!retained_compat::drawValidatedGroup(ids, validate, draw)) {
    ++m_orderedStats.failures;
  }
}

void RetainedCompat::drawImmediate(const Rc<DxvkContext>& ctx, std::unique_ptr<ExternalDrawState> state) {
  std::vector<ExternalDrawState> states;
  if (!prepare(ctx, *state, &states)) {
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
    hideWorldBatches();
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

void RetainedCompat::requestWorldAdmission(WorldReason reason) {
  ++m_worldStructureEvents;
  ++m_worldReasons[size_t(reason)];
  m_worldAdmission.request();
  m_worldDirty = true;
}

void RetainedCompat::invalidateWorld(WorldReason reason) noexcept {
  if (!m_worldGlobalPending) {
    ++m_worldRefreshRevision;
  }
  m_worldGlobalPending = true;
  m_worldUpdates.invalidateStructure();
  requestWorldAdmission(reason);
}

void RetainedCompat::changeWorldEntry(Entry& entry, WorldReason reason) noexcept {
  if (!entry.worldCandidate && entry.children.empty()) {
    return;
  }
  bool relevant = entry.worldCandidate;
  try {
    for (uint64_t handle : entry.children) {
      if (m_worldOwnedChildren.count(handle)) {
        relevant = true;
        m_worldDetach.emplace(handle, entry.record.id);
      }
    }
  } catch (...) {
    invalidateWorld(WorldReason::QueueFailure);
    return;
  }
  if (relevant) {
    requestWorldAdmission(reason);
  }
}

void RetainedCompat::queueWorldVisibility(Entry& entry) noexcept {
  if (!entry.worldCandidate) {
    return;
  }
  ++m_worldVisibilityEvents;
  m_worldDirty = true;
  if (entry.worldVisibilityQueued) {
    return;
  }
  try {
    m_worldVisibilityDirty.push_back(entry.record.id);
    entry.worldVisibilityQueued = true;
  } catch (...) {
    invalidateWorld(WorldReason::QueueFailure);
  }
}

void RetainedCompat::clearWorldVisibility() {
  for (uint64_t id : m_worldVisibilityDirty) {
    const auto found = m_entries.find(id);
    if (found != m_entries.end()) {
      found->second.worldVisibilityQueued = false;
    }
  }
  m_worldVisibilityDirty.clear();
}

bool RetainedCompat::publishWorldVisibility(const Rc<DxvkContext>& ctx, std::vector<RetainedWorldBatcher::Plan>& plans) {
  auto& retained = m_scene.retainedScene();
  for (auto& plan : plans) {
    const auto found = m_worldViews.find(plan.id);
    if (found == m_worldViews.end() || !plan.revision || plan.revision < found->second.revision) {
      return false;
    }
    auto& view = found->second;
    if (plan.visible && (!view.built || view.revision != plan.revision || !retained.submeshInstance(view.handle, 0))) {
      // Release-policy entries must be visible before an immediate build can allocate their scene resources.
      retained.setVisible(view.handle, true);
      if (!retained.buildNow(ctx, view.handle, std::move(plan.renderingState), false)) {
        return false;
      }
      view.built = true;
      view.revision = plan.revision;
      ++m_worldBuilds;
    }
  }
  for (const auto& plan : plans) {
    auto& view = m_worldViews.at(plan.id);
    if (view.built) {
      retained.setVisible(view.handle, plan.visible);
      if (!plan.visible && RtxOptions::retainedHiddenPolicy() == 0) {
        view.built = false;
      }
    }
  }
  return true;
}

void RetainedCompat::hideWorldBatches() {
  for (auto& [id, view] : m_worldViews) {
    if (view.built) {
      m_scene.retainedScene().setVisible(view.handle, false);
      if (RtxOptions::retainedHiddenPolicy() == 0) {
        view.built = false;
      }
    }
  }
  m_worldUpdates.hide();
  m_worldDirty = true;
}

void RetainedCompat::restoreWorldChildren(bool clearCache) {
  auto& retained = m_scene.retainedScene();
  for (const auto& [id, view] : m_worldViews) retained.destroy(view.handle);
  m_worldViews.clear();
  m_worldOwnedChildren.clear();
  for (auto& [id, entry] : m_entries) {
    for (uint64_t handle : entry.children) retained.setVisible(handle, entry.visible);
  }
  if (clearCache) m_worldBatcher.clear();
  m_worldUpdates.lostPublication();
  m_worldDetach.clear();
  clearWorldVisibility();
  m_worldDirty = true;
}

void RetainedCompat::reconcileWorldBatches(const Rc<DxvkContext>& ctx) {
  const auto frame = m_scene.device()->getCurrentFrameId();
  if (!m_active || m_commitFrame != frame || m_status.lastError) {
    return;
  }
  using Clock = std::chrono::steady_clock;
  auto& retained = m_scene.retainedScene();
  if (m_worldGlobalPending) {
    const uint64_t revision = m_worldRefreshRevision;
    restoreWorldChildren(true);
    m_worldGlobalPending = false;
    ++m_worldGlobalRestores;
    m_publishedRefresh.clear();
    // Refresh the recipe that actually produced the view, preserving any uncommitted description.
    for (auto& [id, entry] : m_entries) {
      if (!entry.published.source || (entry.published.record.flags & REMIXAPI_RETAINED_ORDERED) ||
          entry.publishedRefreshRevision == revision) {
        continue;
      }
      try {
        if (build(ctx, entry, true)) {
          continue;
        }
      } catch (const std::exception&) {
        // A stale inline material must never become the fallback for a failed global refresh.
        eraseChildren(entry);
        fail();
      }
      ++m_worldFailures;
      m_worldAdmission.cancel();
      m_worldDirty = false;
      return;
    }
  }
  // Indexed API assets re-resolve only actual users, independently of optional batch admission.
  while (!m_publishedRefresh.empty()) {
    const uint64_t id = *m_publishedRefresh.begin();
    m_publishedRefresh.erase(m_publishedRefresh.begin());
    const auto found = m_entries.find(id);
    if (found == m_entries.end() || !found->second.published.source ||
        (found->second.published.record.flags & REMIXAPI_RETAINED_ORDERED)) {
      continue;
    }
    try {
      if (build(ctx, found->second, true)) {
        continue;
      }
    } catch (const std::exception&) {
      eraseChildren(found->second);
      fail();
    }
    ++m_worldFailures;
    m_worldAdmission.cancel();
    m_worldDirty = false;
    return;
  }
  if (!m_identityToolsInactive) {
    if (!m_worldViews.empty()) {
      restoreWorldChildren(false);
    }
    return;
  }
  if (!m_worldDirty && !m_worldAdmission.pending()) {
    return;
  }
  const bool due = m_worldAdmission.due(Clock::now());
  const bool incremental = m_worldUpdates.incremental(m_worldBatcher.membershipRevision());
  const bool hasLocal = !m_worldDetach.empty() || !m_worldVisibilityDirty.empty() || m_worldUpdates.presentationDirty;
  if (!due && !(incremental && hasLocal)) {
    clearWorldVisibility();
    m_worldDirty = false;
    return;
  }
  auto& logFrame = due ? m_worldFullLogFrame : m_worldFastLogFrame;
  const bool sampleCpu = (due ? m_worldFullCalls : m_worldFastCalls) < 16 || frame - logFrame >= 240;
  const auto snapshotStart = Clock::now();
  auto prepareStart = snapshotStart;
  auto publishStart = snapshotStart;
  const auto milliseconds = [](auto first, auto last) {
    return std::chrono::duration<double, std::milli>(last - first).count();
  };
  const auto recordCpu = [&](WorldCpuStats& stats, Clock::time_point publishEnd) {
    const double snapshotMs = milliseconds(snapshotStart, prepareStart);
    const double prepareMs = milliseconds(prepareStart, publishStart);
    const double publishMs = milliseconds(publishStart, publishEnd);
    ++stats.calls;
    stats.snapshotTotal += snapshotMs;
    stats.prepareTotal += prepareMs;
    stats.publishTotal += publishMs;
    stats.snapshotMax = std::max(stats.snapshotMax, snapshotMs);
    stats.prepareMax = std::max(stats.prepareMax, prepareMs);
    stats.publishMax = std::max(stats.publishMax, publishMs);
  };
  const auto logFailure = [&](const char* stage) noexcept {
    if (!sampleCpu || (m_worldFailures > 4 && frame - logFrame < 240)) return;
    logFrame = frame;
    try {
      Logger::err(str::format("[WorldBatchFailure] stage=", stage, " failures=", m_worldFailures,
        " pendingAdmission=", m_worldAdmission.pending(), " elapsedMs=", milliseconds(snapshotStart, Clock::now())));
    } catch (...) { }
  };
  try {
    // A due full transaction supersedes local masks. Do not rebuild one plan twice in a frame.
    if (incremental && hasLocal && !due) {
      ++m_worldFastCalls;
      const bool republishAll = m_worldUpdates.presentationDirty;
      std::vector<RetainedWorldBatcher::VisibilityInput> changes;
      for (uint64_t id : m_worldVisibilityDirty) {
        const auto found = m_entries.find(id);
        if (found == m_entries.end() || !found->second.worldCandidate) {
          continue;
        }
        for (uint64_t handle : found->second.children) {
          changes.push_back({handle, found->second.visible});
        }
      }
      std::vector<uint64_t> detached;
      detached.reserve(m_worldDetach.size());
      for (const auto& [handle, parent] : m_worldDetach) {
        detached.push_back(handle);
      }
      std::vector<RetainedWorldBatcher::Plan> plans;
      prepareStart = Clock::now();
      const bool prepared = detached.empty()
        ? m_worldBatcher.updateVisibility(ctx, m_worldUpdates.revision, changes, republishAll, plans)
        : m_worldBatcher.detachChildren(ctx, m_worldUpdates.revision, detached, changes, republishAll, plans);
      publishStart = Clock::now();
      if (!prepared || !publishWorldVisibility(ctx, plans)) {
        ++m_worldFailures;
        restoreWorldChildren(false);
        m_worldAdmission.cancel();
        m_worldDirty = false;
        logFailure(prepared ? "local-publish" : "local-prepare");
        return;
      }
      // The old parts are absent before canonical ownership is released.
      for (const auto& [handle, parent] : m_worldDetach) {
        m_worldOwnedChildren.erase(handle);
        const auto found = m_entries.find(parent);
        if (found != m_entries.end() &&
            std::find(found->second.children.begin(), found->second.children.end(), handle) != found->second.children.end()) {
          retained.setVisible(handle, found->second.visible);
        }
      }
      m_worldDetachedChildren += detached.size();
      m_worldDetach.clear();
      m_worldUpdates.presented();
      clearWorldVisibility();
      m_worldDirty = false;
      const auto publishEnd = Clock::now();
      recordCpu(m_worldFastCpu, publishEnd);
      if (sampleCpu) {
        logFrame = frame;
        const auto& stats = m_worldBatcher.stats();
        Logger::info(str::format("[WorldBatchFast] changes=", changes.size(), " plans=", plans.size(),
          " republishAll=", republishAll, " fullCalls=", m_worldFullCalls, " fastCalls=", m_worldFastCalls,
          " structureEvents=", m_worldStructureEvents, " visibilityEvents=", m_worldVisibilityEvents,
          " pendingAdmission=", m_worldAdmission.pending(), " detachedChildren=", m_worldDetachedChildren,
          " globalRestores=", m_worldGlobalRestores,
          " causeNew=", m_worldReasons[size_t(WorldReason::NewCandidate)], " causeRemove=", m_worldReasons[size_t(WorldReason::Remove)],
          " causeMove=", m_worldReasons[size_t(WorldReason::FirstMove)], " causeDescription=", m_worldReasons[size_t(WorldReason::Description)],
          " causeBuild=", m_worldReasons[size_t(WorldReason::ChildBuild)], " causeAssetGlobal=", m_worldReasons[size_t(WorldReason::AssetGlobal)],
          " causeAssetIndexed=", m_worldReasons[size_t(WorldReason::AssetIndexed)], " causeTexture=", m_worldReasons[size_t(WorldReason::Texture)],
          " causeClear=", m_worldReasons[size_t(WorldReason::SceneClear)], " causeOptions=", m_worldReasons[size_t(WorldReason::Options)],
          " causeWetness=", m_worldReasons[size_t(WorldReason::Wetness)], " causeIdentity=", m_worldReasons[size_t(WorldReason::Identity)],
          " causeQueueFailure=", m_worldReasons[size_t(WorldReason::QueueFailure)],
          " builds=", m_worldBuilds, " failures=", m_worldFailures,
          " vertexBuilds=", stats.vertexBuilds, " indexBuilds=", stats.indexBuilds, " vertexReuses=", stats.vertexReuses,
          " vertexBuildAttempts=", stats.vertexBuildAttempts, " indexBuildAttempts=", stats.indexBuildAttempts,
          " indexAllocationAttempts=", stats.indexAllocationAttempts, " indexAllocations=", stats.indexAllocations,
          " indexBytesUploaded=", stats.indexBytesUploaded, " indexTrianglesUploaded=", stats.indexTrianglesUploaded,
          " revealReuse=", stats.revealReuse, " allHiddenNoAllocation=", stats.allHiddenNoAllocation,
          " vertexAllocationAttempts=", stats.vertexAllocationAttempts, " vertexAllocations=", stats.vertexAllocations,
          " vertexBytesUploaded=", stats.vertexBytesUploaded,
          " lastFixedChunks=", stats.lastFixedChunks, " lastFixedBytes=", stats.lastFixedBytes,
          " lastResidentParts=", stats.lastResidentParts, " lastActiveParts=", stats.lastActiveParts,
          " lastDeltaCandidateChildren=", stats.lastDeltaCandidateChildren, " lastDeltaCandidateParts=", stats.lastDeltaCandidateParts,
          " lastDeltaAdmittedChildren=", stats.lastDeltaAdmittedChildren, " lastDeltaAdmittedParts=", stats.lastDeltaAdmittedParts,
          " lastDeadChunks=", stats.lastDeadChunks,
          " visibilityUpdates=", stats.visibilityUpdates, " visibilityCommits=", stats.visibilityCommits,
          " detachUpdates=", stats.detachUpdates, " detachCommits=", stats.detachCommits, " detachFailures=", stats.detachFailures,
          " residentBatches=", stats.batches, " bytes=", stats.bytes,
          " fullCpuCalls=", m_worldFullCpu.calls,
          " fullSnapshotMsTotal=", m_worldFullCpu.snapshotTotal, " fullSnapshotMsMax=", m_worldFullCpu.snapshotMax,
          " fullPrepareMsTotal=", m_worldFullCpu.prepareTotal, " fullPrepareMsMax=", m_worldFullCpu.prepareMax,
          " fullPublishMsTotal=", m_worldFullCpu.publishTotal, " fullPublishMsMax=", m_worldFullCpu.publishMax,
          " fastCpuCalls=", m_worldFastCpu.calls,
          " fastSnapshotMsTotal=", m_worldFastCpu.snapshotTotal, " fastSnapshotMsMax=", m_worldFastCpu.snapshotMax,
          " fastPrepareMsTotal=", m_worldFastCpu.prepareTotal, " fastPrepareMsMax=", m_worldFastCpu.prepareMax,
          " fastPublishMsTotal=", m_worldFastCpu.publishTotal, " fastPublishMsMax=", m_worldFastCpu.publishMax,
          " snapshotMs=", milliseconds(snapshotStart, prepareStart), " prepareMs=", milliseconds(prepareStart, publishStart),
          " publishMs=", milliseconds(publishStart, publishEnd)));
      }
    }
    if (!due) {
      return;
    }
    m_worldAdmission.begin(Clock::now());
    ++m_worldFullCalls;
    std::vector<RetainedWorldBatcher::ParentInput> inputs;
    std::unordered_map<uint64_t, std::vector<uint8_t>> coverage;
    std::unordered_set<uint64_t> visibleInputs;
    for (const auto& [id, entry] : m_entries) {
      if (!entry.worldCandidate || entry.everMoved || (entry.record.flags & REMIXAPI_RETAINED_ORDERED) || entry.dirty ||
          entry.worldParts.size() != entry.children.size()) continue;
      for (size_t i = 0; i < entry.children.size(); ++i) {
        if (entry.worldParts[i].empty()) continue;
        // The synchronous preparation borrows descriptions. New cached sources take their own strong copies.
        RetainedWorldBatcher::ParentInput input {entry.children[i], 0,
          uint64_t(m_scene.getWetnessMeshRule(entry.worldParts[i].front().drawCall.wetnessMeshHash)),
          entry.visible, entry.worldParts[i]};
        if (input.visible) visibleInputs.insert(input.id);
        coverage.emplace(input.id, std::vector<uint8_t>(input.parts.size(), 0));
        inputs.push_back(std::move(input));
      }
    }
    // A due full admission must retire queued old slots even if their source description still compares equal.
    std::vector<uint64_t> pendingDetaches;
    pendingDetaches.reserve(m_worldDetach.size());
    for (const auto& [handle, parent] : m_worldDetach) {
      pendingDetaches.push_back(handle);
    }
    std::vector<RetainedWorldBatcher::Plan> plans;
    prepareStart = Clock::now();
    const bool prepared = m_worldBatcher.prepare(ctx, inputs, 64.f, pendingDetaches, plans);
    publishStart = Clock::now();
    if (!prepared) {
      ++m_worldFailures;
      restoreWorldChildren(false);
      m_worldDirty = false;
      logFailure("prepare");
      return;
    }
    std::unordered_set<uint64_t> owners, planIds;
    // Independently validate the planner's all-parts contract before suppressing any canonical child.
    bool valid = true;
    for (const auto& plan : plans) {
      valid = valid && plan.id && plan.revision && !plan.members.empty() && planIds.insert(plan.id).second;
      bool anyVisible = false;
      for (const auto& member : plan.members) {
        auto found = coverage.find(member.id);
        if (found == coverage.end() || member.part >= found->second.size() || found->second[member.part]) {
          valid = false;
          break;
        }
        found->second[member.part] = 1;
        anyVisible = anyVisible || visibleInputs.count(member.id) != 0;
        owners.insert(member.id);
      }
      valid = valid && plan.visible == anyVisible;
    }
    for (uint64_t owner : owners) {
      const auto& parts = coverage.at(owner);
      valid = valid && std::all_of(parts.begin(), parts.end(), [](uint8_t value) { return value != 0; });
    }
    if (!valid) {
      ++m_worldFailures;
      restoreWorldChildren(true);
      m_worldDirty = false;
      Logger::err("[WorldBatch] rejected incomplete or duplicate source ownership");
      logFailure("coverage");
      return;
    }
    // Prepare the full scene views first. Failure restores every canonical source, including a built prefix.
    // This scene-thread transaction completes before upload, so existing views need no temporary hide/show.
    for (auto& plan : plans) {
      auto [found, inserted] = m_worldViews.try_emplace(plan.id);
      auto& view = found->second;
      if (inserted) view.handle = ++m_nextHandle;
      if (plan.visible && (!view.built || view.revision != plan.revision || !retained.submeshInstance(view.handle, 0))) {
        retained.setVisible(view.handle, true);
        if (!retained.buildNow(ctx, view.handle, std::move(plan.renderingState), false)) {
          ++m_worldFailures;
          restoreWorldChildren(false);
          m_worldDirty = false;
          logFailure("publish");
          return;
        }
        view.built = true;
        view.revision = plan.revision;
        ++m_worldBuilds;
      }
    }
    for (auto it = m_worldViews.begin(); it != m_worldViews.end();) {
      if (!planIds.count(it->first)) {
        retained.destroy(it->second.handle);
        it = m_worldViews.erase(it);
      } else ++it;
    }
    m_worldOwnedChildren.swap(owners);
    for (auto& [id, entry] : m_entries) {
      for (uint64_t handle : entry.children) retained.setVisible(handle, entry.visible && !m_worldOwnedChildren.count(handle));
    }
    for (const auto& plan : plans) {
      auto& view = m_worldViews.at(plan.id);
      if (view.built) {
        retained.setVisible(view.handle, plan.visible);
        if (!plan.visible && RtxOptions::retainedHiddenPolicy() == 0) {
          view.built = false;
        }
      }
    }
    m_worldUpdates.prepared(m_worldBatcher.membershipRevision());
    m_worldDetach.clear();
    clearWorldVisibility();
    m_worldDirty = false;
    uint64_t visibleParts = 0, visibleChildren = 0, visibleBatches = 0;
    for (const auto& input : inputs) {
      if (input.visible && m_worldOwnedChildren.count(input.id)) {
        ++visibleChildren;
        visibleParts += input.parts.size();
      }
    }
    for (const auto& plan : plans) visibleBatches += plan.visible;
    const auto publishEnd = Clock::now();
    recordCpu(m_worldFullCpu, publishEnd);
    const auto& stats = m_worldBatcher.stats();
    // Diagnostic logging cannot turn an already committed publication into a failure.
    try {
      Logger::info(str::format("[WorldBatchPublish] frame=", frame, " prepareAttempt=", stats.prepares,
        " success=compatPublish fullCpuCalls=", m_worldFullCpu.calls,
        " inputs=", inputs.size(), " ownedChildren=", m_worldOwnedChildren.size(),
        " plans=", plans.size(), " visibleParts=", visibleParts, " visibleBatches=", visibleBatches,
        " snapshotMs=", milliseconds(snapshotStart, prepareStart),
        " prepareMs=", milliseconds(prepareStart, publishStart),
        " publishMs=", milliseconds(publishStart, publishEnd)));
    } catch (...) {
    }
    if (sampleCpu) {
      logFrame = frame;
      Logger::info(str::format("[WorldBatch] mode=full fullCalls=", m_worldFullCalls, " fastCalls=", m_worldFastCalls,
      " structureEvents=", m_worldStructureEvents, " visibilityEvents=", m_worldVisibilityEvents,
      " pendingAdmission=", m_worldAdmission.pending(), " detachedChildren=", m_worldDetachedChildren,
      " globalRestores=", m_worldGlobalRestores,
      " causeNew=", m_worldReasons[size_t(WorldReason::NewCandidate)], " causeRemove=", m_worldReasons[size_t(WorldReason::Remove)],
      " causeMove=", m_worldReasons[size_t(WorldReason::FirstMove)], " causeDescription=", m_worldReasons[size_t(WorldReason::Description)],
      " causeBuild=", m_worldReasons[size_t(WorldReason::ChildBuild)], " causeAssetGlobal=", m_worldReasons[size_t(WorldReason::AssetGlobal)],
      " causeAssetIndexed=", m_worldReasons[size_t(WorldReason::AssetIndexed)], " causeTexture=", m_worldReasons[size_t(WorldReason::Texture)],
      " causeClear=", m_worldReasons[size_t(WorldReason::SceneClear)], " causeOptions=", m_worldReasons[size_t(WorldReason::Options)],
      " causeWetness=", m_worldReasons[size_t(WorldReason::Wetness)], " causeIdentity=", m_worldReasons[size_t(WorldReason::Identity)],
      " causeQueueFailure=", m_worldReasons[size_t(WorldReason::QueueFailure)],
      " inputs=", inputs.size(), " ownedChildren=", m_worldOwnedChildren.size(),
      " batches=", plans.size(), " visibleChildren=", visibleChildren, " visibleParts=", visibleParts,
      " visibleBatches=", visibleBatches, " builds=", m_worldBuilds, " failures=", m_worldFailures,
      " vertexBuilds=", stats.vertexBuilds, " indexBuilds=", stats.indexBuilds,
      " vertexReuses=", stats.vertexReuses, " bytes=", stats.bytes,
      " vertexBuildAttempts=", stats.vertexBuildAttempts, " indexBuildAttempts=", stats.indexBuildAttempts,
      " indexAllocationAttempts=", stats.indexAllocationAttempts, " indexAllocations=", stats.indexAllocations,
      " indexBytesUploaded=", stats.indexBytesUploaded, " indexTrianglesUploaded=", stats.indexTrianglesUploaded,
      " revealReuse=", stats.revealReuse, " allHiddenNoAllocation=", stats.allHiddenNoAllocation,
      " vertexAllocationAttempts=", stats.vertexAllocationAttempts, " vertexAllocations=", stats.vertexAllocations,
      " vertexBytesUploaded=", stats.vertexBytesUploaded,
      " lastFixedChunks=", stats.lastFixedChunks, " lastFixedBytes=", stats.lastFixedBytes,
      " lastResidentParts=", stats.lastResidentParts, " lastActiveParts=", stats.lastActiveParts,
      " lastDeltaCandidateChildren=", stats.lastDeltaCandidateChildren, " lastDeltaCandidateParts=", stats.lastDeltaCandidateParts,
      " lastDeltaAdmittedChildren=", stats.lastDeltaAdmittedChildren, " lastDeltaAdmittedParts=", stats.lastDeltaAdmittedParts,
      " lastDeadChunks=", stats.lastDeadChunks,
      " budgetRejectedInputs=", stats.preflightRejectedInputs, " budgetRejectedBatches=", stats.preflightRejectedBatches,
      " budgetRejectedComponents=", stats.preflightRejectedComponents,
      " projectedBatches=", stats.lastProjectedBatches, " projectedBytes=", stats.lastProjectedBytes,
      " selectedBatches=", stats.lastSelectedBatches, " selectedBytes=", stats.lastSelectedBytes,
      " fullCpuCalls=", m_worldFullCpu.calls,
      " fullSnapshotMsTotal=", m_worldFullCpu.snapshotTotal, " fullSnapshotMsMax=", m_worldFullCpu.snapshotMax,
      " fullPrepareMsTotal=", m_worldFullCpu.prepareTotal, " fullPrepareMsMax=", m_worldFullCpu.prepareMax,
      " fullPublishMsTotal=", m_worldFullCpu.publishTotal, " fullPublishMsMax=", m_worldFullCpu.publishMax,
      " fastCpuCalls=", m_worldFastCpu.calls,
      " fastSnapshotMsTotal=", m_worldFastCpu.snapshotTotal, " fastSnapshotMsMax=", m_worldFastCpu.snapshotMax,
      " fastPrepareMsTotal=", m_worldFastCpu.prepareTotal, " fastPrepareMsMax=", m_worldFastCpu.prepareMax,
      " fastPublishMsTotal=", m_worldFastCpu.publishTotal, " fastPublishMsMax=", m_worldFastCpu.publishMax,
      " snapshotMs=", milliseconds(snapshotStart, prepareStart),
      " prepareMs=", milliseconds(prepareStart, publishStart),
      " publishMs=", milliseconds(publishStart, publishEnd)));
    }
  } catch (const std::exception&) {
    ++m_worldFailures;
    restoreWorldChildren(false);
    m_worldAdmission.cancel();
    m_worldDirty = false;
    logFailure("exception");
  }
}

void RetainedCompat::destroyMesh(remixapi_MeshHandle mesh) {
  m_batchCache.clear();
  std::vector<uint64_t> ids;
  for (auto& [id, entry] : m_entries) {
    const auto action = retained_compat::sourceDestruction(entry.state && entry.state->mesh == mesh,
      entry.published.source && entry.published.source->mesh == mesh);
    if (action == retained_compat::SourceDestruction::RemoveEntry) {
      ids.push_back(id);
    } else if (action == retained_compat::SourceDestruction::ForgetPublished) {
      // The pending replacement remains staged for the next host commit.
      eraseChildren(entry);
      entry.published.forget();
      if (!entry.dirty) {
        entry.dirty = true;
        m_dirty.push_back(id);
      }
      indexAssets(entry);
    }
  }
  if (!ids.empty()) {
    remove(m_status.epoch, ids);
  }
}

}
