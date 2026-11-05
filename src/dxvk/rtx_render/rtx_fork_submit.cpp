// The fork_hooks functions of the SceneManager::submitExternalDraw path (see rtx_fork_hooks.h).
// externalDrawObjectPicking accesses SceneManager::m_drawCallMeta, so SceneManager names it a friend.

#include "rtx_fork_hooks.h"

#include "rtx_asset_replacer.h"   // AssetReplacer, AssetReplacement
#include "rtx_options.h"          // RtxOptions::*, fast_unordered_set, InstanceCategories
#include "rtx_scene_manager.h"    // SceneManager, DrawCallMetaInfo

#include "dxvk_device.h"          // DxvkDevice::getCommon()->getResources()

namespace dxvk {
namespace fork_hooks {

  // Returns the USD mesh or light replacement bucket keyed on the API mesh handle hash, as the D3D9 draw path does.
  // The caller then calls drawReplacements and returns.
  // Returns null when there is no replacement.
  std::shared_ptr<const ReplacementBucket> externalDrawMeshReplacement(
      AssetReplacer& replacer, XXH64_hash_t meshHash) {
    // A host key (then its aliases) replaces the handle as the lookup key.
    if (!replacer.assetResolver().empty()) {
      if (const AssetResolver::AssetKey* key = replacer.assetResolver().meshKey(meshHash)) {
        return AssetResolver::resolve(*key, [&](uint64_t k) { return replacer.getReplacementsForMesh(k); });
      }
    }
    return replacer.getReplacementsForMesh(meshHash);
  }

  // Points the caller's material at its USD replacement (getReplacementMaterial) if there is one.
  std::shared_ptr<MaterialData> externalDrawMaterialReplacement(
      AssetReplacer& replacer, const MaterialData*& material) {
    // Replacement materials are handed out by shared_ptr, so a hot reload cannot free one under a live draw.
    // The owner goes back to the caller, so `material` stays valid for the rest of the submesh iteration.
    std::shared_ptr<MaterialData> pReplacementMaterial;
    const AssetResolver::AssetKey* key = nullptr;
    if (!replacer.assetResolver().empty()) {
      key = replacer.assetResolver().materialKey(material->getHash());
    }
    if (key != nullptr) {
      pReplacementMaterial = AssetResolver::resolve(*key, [&](uint64_t k) { return replacer.getReplacementMaterial(k); });
    } else {
      pReplacementMaterial = replacer.getReplacementMaterial(material->getHash());
    }
    if (pReplacementMaterial != nullptr) {
      material = pReplacementMaterial.get();
    }
    return pReplacementMaterial;
  }

  // Applies the texture-based instance categories to an API draw. The albedo texture hash from the material's
  // opaque data goes to outTextureHash (object picking uses it too) and is looked up in every category set.
  void externalDrawTextureCategories(
      const MaterialData* material,
      SceneDraw& drawCall,
      XXH64_hash_t& outTextureHash) {
    // The albedo texture hash is what the D3D9 path's setupCategoriesForTexture() keys on.
    if (material->getType() == MaterialDataType::Opaque) {
      const auto& opaqueMat = material->getOpaqueMaterialData();
      if (opaqueMat.getAlbedoOpacityTexture().isValid()) {
        outTextureHash = opaqueMat.getAlbedoOpacityTexture().getImageHash();
      }
    }

    if (outTextureHash != 0 && outTextureHash != kEmptyHash) {
      auto applyCategory = [&](const fast_unordered_set& hashSet, InstanceCategories cat) {
        if (hashSet.find(outTextureHash) != hashSet.end()) {
          drawCall.setCategory(cat, true);
        }
      };

      applyCategory(RtxOptions::skyBoxTextures(), InstanceCategories::Sky);
      applyCategory(RtxOptions::ignoreTextures(), InstanceCategories::Ignore);
      applyCategory(RtxOptions::worldSpaceUiTextures(), InstanceCategories::WorldUI);
      applyCategory(RtxOptions::worldSpaceUiBackgroundTextures(), InstanceCategories::WorldMatte);
      applyCategory(RtxOptions::particleTextures(), InstanceCategories::Particle);
      applyCategory(RtxOptions::beamTextures(), InstanceCategories::Beam);
      applyCategory(RtxOptions::decalTextures(), InstanceCategories::DecalStatic);
      applyCategory(RtxOptions::terrainTextures(), InstanceCategories::Terrain);
      applyCategory(RtxOptions::animatedWaterTextures(), InstanceCategories::AnimatedWater);
      applyCategory(RtxOptions::ignoreLights(), InstanceCategories::IgnoreLights);
      applyCategory(RtxOptions::antiCullingTextures(), InstanceCategories::IgnoreAntiCulling);
      applyCategory(RtxOptions::motionBlurMaskOutTextures(), InstanceCategories::IgnoreMotionBlur);
      applyCategory(RtxOptions::hideInstanceTextures(), InstanceCategories::Hidden);
    }
  }

  // Stores the draw's texture hash in SceneManager::m_drawCallMeta while object picking is active, as the D3D9
  // path does in processDrawCallState.
  void externalDrawObjectPicking(
      DxvkDevice& device,
      SceneDraw& drawCall,
      XXH64_hash_t textureHash,
      SceneManager& scene) {
    // API draws supply their own drawCallID (remixapi_InstanceInfoObjectPickingEXT), which keys the entry.
    const bool objectPickingActive = device.getCommon()->getResources().getRaytracingOutput()
      .m_primaryObjectPicking.isValid();
    if (objectPickingActive && drawCall.drawCallID != 0 &&
        textureHash != 0 && textureHash != kEmptyHash) {
      auto meta = SceneManager::DrawCallMetaInfo {};
      meta.legacyTextureHash = textureHash;

      std::lock_guard lock { scene.m_drawCallMeta.mutex };
      auto [iter, isNew] = scene.m_drawCallMeta.infos[scene.m_drawCallMeta.ticker].emplace(drawCall.drawCallID, meta);
      ONCE_IF_FALSE(isNew, Logger::warn(
        "Found multiple API draw calls with the same \'objectPickingValue\'. "
        "Some objects might not be available through object picking"));
    }
  }

} // namespace fork_hooks
} // namespace dxvk
