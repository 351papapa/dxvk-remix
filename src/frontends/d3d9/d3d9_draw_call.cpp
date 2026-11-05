/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include "d3d9_draw_call.h"

#include <algorithm>
#include <optional>

#include "../../dxvk/rtx_render/rtx_options.h"
#include "../../tracy/Tracy.hpp"
#include "../../util/util_fast_cache.h"

namespace dxvk {

  bool DrawCallState::finalizePendingFutures(const frontend::CameraQuery* skinningCamera) {
    ZoneScopedN(__FUNCTION__);
    // Geometry hashes are vital, and cannot be disabled, so its important we get valid data (hence the return type)
    const bool valid = finalizeGeometryHashes();
    if (valid) {
      // Bounding boxes (if enabled) will be finalized here, default is FLT_MAX bounds
      finalizeGeometryBoundingBox();

      // Skinning processing will be finalized here, if object requires skinning
      finalizeSkinningData(skinningCamera);

      // Update any categories that require geometry hash
      setupCategoriesForGeometry();

      return true;
    }

    return false;
  }

  bool DrawCallState::finalizeGeometryHashes() {
    if (!futureGeometryHashes.valid()) {
      return false;
    }

    geometryData.hashes = futureGeometryHashes.get();

    if (geometryData.hashes[HashComponents::VertexPosition] == kEmptyHash) {
      throw DxvkError("Position hash should never be empty");
    }

    return true;
  }

  void DrawCallState::finalizeGeometryBoundingBox() {
    if (futureBoundingBox.valid())
      geometryData.boundingBox = futureBoundingBox.get();
  }

  void DrawCallState::finalizeSkinningData(const frontend::CameraQuery* skinningCamera) {
    if (futureSkinningData.valid()) {
      skinningData = futureSkinningData.get();

      assert(geometryData.blendWeightBuffer.defined());
      assert(skinningData.numBonesPerVertex <= 4);

      if (skinningCamera != nullptr) {
        const auto fusedMode = RtxOptions::fusedWorldViewMode();
        if (likely(fusedMode == FusedWorldViewMode::None)) {
          transformData.objectToView = transformData.worldToView;
          // Do not bother when transform is fused. Camera matrices are identity and so is worldToView.
        }
        transformData.objectToWorld = skinningCamera->preciseViewToWorld * transformData.objectToView;
        transformData.worldToView = skinningCamera->preciseWorldToView;
      } else {
        ONCE(Logger::warn("[RTX-Compatibility-Warn] Cannot decompose the matrices for a skinned mesh because the camera is not set."));
      }

      // In rare cases when the mesh is skinned but has only one active bone, skip the skinning pass
      // and bake that single bone into the objectToWorld/View matrices.
      if (skinningData.minBoneIndex + 1 == skinningData.numBones) {
        const Matrix4& skinningMatrix = skinningData.pBoneMatrices[skinningData.minBoneIndex];

        transformData.objectToWorld = transformData.objectToWorld * skinningMatrix;
        transformData.objectToView = transformData.objectToView * skinningMatrix;

        skinningData.boneHash = 0;
        skinningData.numBones = 0;
        skinningData.numBonesPerVertex = 0;
      }

      // Store the numBonesPerVertex in the RasterGeometry as well to allow it to be overridden
      geometryData.numBonesPerVertex = skinningData.numBonesPerVertex;
    }
  }

  void DrawCallState::setCategory(InstanceCategories category, bool doSet) {
    if (doSet) {
      categories.set(category);
    }
  }

  void DrawCallState::removeCategory(InstanceCategories category) {
    categories.clr(category);
  }

  CategoryFlags DrawCallState::computeTextureListCategories(const XXH64_hash_t& textureHash) {
    // The texture-list part of setupCategoriesForTexture(): depends only on the texture hash and the
    // option values, so rtx.textureCategoryCache can cache it per texture hash.
    CategoryFlags result = 0;
    const auto setIf = [&result](InstanceCategories category, bool doSet) {
      if (doSet) {
        result.set(category);
      }
    };

    setIf(InstanceCategories::WorldUI, lookupHash(RtxOptions::worldSpaceUiTextures(), textureHash));
    setIf(InstanceCategories::WorldMatte, lookupHash(RtxOptions::worldSpaceUiBackgroundTextures(), textureHash));

    setIf(InstanceCategories::Ignore, lookupHash(RtxOptions::ignoreTextures(), textureHash));
    setIf(InstanceCategories::IgnoreLights, lookupHash(RtxOptions::ignoreLights(), textureHash));
    setIf(InstanceCategories::IgnoreAntiCulling, lookupHash(RtxOptions::antiCullingTextures(), textureHash));
    setIf(InstanceCategories::IgnoreMotionBlur, lookupHash(RtxOptions::motionBlurMaskOutTextures(), textureHash));
    setIf(InstanceCategories::IgnoreOpacityMicromap, lookupHash(RtxOptions::opacityMicromapIgnoreTextures(), textureHash));
    setIf(InstanceCategories::IgnoreAlphaChannel, lookupHash(RtxOptions::ignoreAlphaOnTextures(), textureHash));
    setIf(InstanceCategories::IgnoreBakedLighting, lookupHash(RtxOptions::ignoreBakedLightingTextures(), textureHash));

    setIf(InstanceCategories::Hidden, lookupHash(RtxOptions::hideInstanceTextures(), textureHash));

    setIf(InstanceCategories::Particle, lookupHash(RtxOptions::particleTextures(), textureHash));
    setIf(InstanceCategories::Beam, lookupHash(RtxOptions::beamTextures(), textureHash));

    setIf(InstanceCategories::DecalStatic, lookupHash(RtxOptions::decalTextures(), textureHash));
    setIf(InstanceCategories::DecalDynamic, lookupHash(RtxOptions::dynamicDecalTextures(), textureHash));
    setIf(InstanceCategories::DecalSingleOffset, lookupHash(RtxOptions::singleOffsetDecalTextures(), textureHash));
    setIf(InstanceCategories::DecalNoOffset, lookupHash(RtxOptions::nonOffsetDecalTextures(), textureHash));

    setIf(InstanceCategories::AnimatedWater, lookupHash(RtxOptions::animatedWaterTextures(), textureHash));

    setIf(InstanceCategories::ThirdPersonPlayerModel, lookupHash(RtxOptions::playerModelTextures(), textureHash));
    setIf(InstanceCategories::ThirdPersonPlayerBody, lookupHash(RtxOptions::playerModelBodyTextures(), textureHash));

    setIf(InstanceCategories::Terrain, lookupHash(RtxOptions::terrainTextures(), textureHash));
    setIf(InstanceCategories::Sky, lookupHash(RtxOptions::skyBoxTextures(), textureHash));

    setIf(InstanceCategories::ParticleEmitter, lookupHash(RtxOptions::particleEmitterTextures(), textureHash));
    setIf(InstanceCategories::HairCards, lookupHash(RtxOptions::hairCardTextures(), textureHash));
    return result;
  }

  void DrawCallState::setupCategoriesForTexture(const CategoryFlags* pCachedTextureListCategories) {
    // TODO (REMIX-231): It would probably be much more efficient to use a map of texture hash to category flags, rather
    //                   than doing N lookups per texture hash for each category.
    //                   (rtx.textureCategoryCache: the D3D9 layer passes the cached result for this texture.)
    const XXH64_hash_t& textureHash = materialData.getColorTexture().getImageHash();

    // setCategory() only ever sets bits, so OR-ing the list result in is the same as setting each bit in turn.
    categories.set(pCachedTextureListCategories != nullptr ? *pCachedTextureListCategories
                                                           : computeTextureListCategories(textureHash));
    setCategory(InstanceCategories::IgnoreOpacityMicromap, isUsingRaytracedRenderTarget);
  }

  void DrawCallState::setupCategoriesForGeometry() {
    const XXH64_hash_t assetReplacementHash = getHash(RtxOptions::geometryAssetHashRule());
    setCategory(InstanceCategories::Sky, lookupHash(RtxOptions::skyBoxGeometries(), assetReplacementHash));
  }

  static std::optional<Vector3> makeCameraPosition(const Matrix4& worldToView,
                                                   bool zWrite,
                                                   bool alphaBlend,
                                                   bool hasSkinning) {
    if (hasSkinning) {
      return std::nullopt;
    }
    // particles
    if (!zWrite && alphaBlend) {
      return std::nullopt;
    }
    // identity matrix
    if (isIdentityExact(worldToView)) {
      return std::nullopt;
    }

#define USE_TRUE_CAMERA_POSITION_FOR_COMPARISON 0

#if USE_TRUE_CAMERA_POSITION_FOR_COMPARISON
    return (inverse(worldToView))[3].xyz();
#else
    // as we compare the cameras relatively and don't need precise camera position:
    // just return a position-like vector, to avoid calculating heavy matrix inverse operation
    return worldToView[3].xyz();
#endif
  }

  static bool areCamerasClose(const Vector3& a, const Vector3& b) {
    const float distanceThreshold = RtxOptions::skyAutoDetectUniqueCameraDistance();
    return lengthSqr(a - b) < distanceThreshold * distanceThreshold;
  }

  static bool checkSkyAutoDetect(bool depthTestEnable,
                          const std::optional<Vector3>& newCameraPos,
                          uint32_t prevFrameSeenCamerasCount,
                          const std::vector<Vector3>& seenCameraPositions) {

    if (RtxOptions::skyAutoDetect() != SkyAutoDetectMode::CameraPositionAndDepthFlags &&
        RtxOptions::skyAutoDetect() != SkyAutoDetectMode::CameraPosition) {
      return false;
    }
    const bool withDepthFlags = (RtxOptions::skyAutoDetect() == SkyAutoDetectMode::CameraPositionAndDepthFlags);


    const bool searchingForSkyCamera             = (seenCameraPositions.size() == 0);
    const bool skyFoundAndSearchingForMainCamera = (seenCameraPositions.size() == 1);
    const bool skyAndMainCameraFound             = (seenCameraPositions.size() >= 2);

    if (skyAndMainCameraFound) {
      // assume that subsequent draw calls can not be sky
      return false;
    }

    if (searchingForSkyCamera) {
      if (withDepthFlags) {
        // no depth test: frame starts with a sky
        // depth test: frame starts with a world, not a sky
        return !depthTestEnable;
      }
      // assume the first camera to be sky
      return true;
    }

    {
      // corner case: if there was no sky camera at all, fallback, but this would also
      // involve a one-frame (preceding to the current one) being rasterized (like a flicker)
      if (prevFrameSeenCamerasCount < 2) {
        if (withDepthFlags) {
          // no depth test: sky
          // depth test: world
          return !depthTestEnable;
        }
        // assume no sky
        return false;
      }
    }

    if (skyFoundAndSearchingForMainCamera) {
      // if draw call doesn't have a camera position
      if (!newCameraPos) {
        // it can't contain main camera, so assume that it's still a sky
        return true;
      }

      // if same as the existing sky camera
      if (areCamerasClose(seenCameraPositions[0], *newCameraPos)) {
        // still sky
        return true;
      }

      // found a new unique camera, which should be a main camera
      return false;
    }

    assert(0);
    return false;
  }

  enum class SkyDetectionSource {
    None,
    Explicit,   // minZ, texHash, geoHash, dcIdThreshold
    AutoDetect  // checkSkyAutoDetect
  };

  static SkyDetectionSource shouldBakeSky(const DrawCallState& drawCallState,
                     bool hasSkinning,
                     uint32_t prevFrameSeenCamerasCount,
                     std::vector<Vector3>& seenCameraPositions) {           
    const auto drawCallCameraPos =
      drawCallState.isDrawingToRaytracedRenderTarget
        ? std::optional<Vector3>{}
        : makeCameraPosition(
            drawCallState.getTransformData().worldToView,
            drawCallState.zWriteEnable,
            drawCallState.getMaterialData().blendMode.enableBlending,
            hasSkinning);

    auto l_addIfUnique = [&seenCameraPositions](const std::optional<Vector3>& newCameraPos) {
      if (!newCameraPos) {
        return;
      }
      for (const Vector3& seen : seenCameraPositions) {
        if (areCamerasClose(seen, *newCameraPos)) {
          return;
        }
      }
      seenCameraPositions.push_back(*newCameraPos);
    };
    l_addIfUnique(drawCallCameraPos);


    if (drawCallState.minZ >= RtxOptions::skyMinZThreshold()) {
      return SkyDetectionSource::Explicit;
    }

    // NOTE: we use color texture hash for sky detection, however the replacement is hashed with
    // the whole legacy material hash (which, as of 12/9/2022, equals to color texture hash). Adding a check just in case.
    assert(drawCallState.getMaterialData().getColorTexture().getImageHash() == drawCallState.getMaterialData().getHash() && "Texture or material hash method changed!");

    if (drawCallState.getMaterialData().usesTexture()) {
      if (lookupHash(RtxOptions::skyBoxTextures(), drawCallState.getMaterialData().getHash())) {
        return SkyDetectionSource::Explicit;
      }
    } else {
      if (drawCallState.drawCallID < RtxOptions::skyDrawcallIdThreshold()) {
        return SkyDetectionSource::Explicit;
      }
    }

    // don't track camera positions for Raytraced Render Targets, as they are a different camera position from main view
    const static auto renderTargetCameraPositions = std::vector<Vector3>{};

    if (checkSkyAutoDetect(drawCallState.zEnable,
                           drawCallCameraPos,
                           prevFrameSeenCamerasCount,
                           drawCallState.isDrawingToRaytracedRenderTarget ? renderTargetCameraPositions : seenCameraPositions)) {
      return SkyDetectionSource::AutoDetect;
    }

    return SkyDetectionSource::None;
  }

  static bool shouldBakeTerrain(const DrawCallState& drawCallState) {
    if (!frontend::terrainBakeActive())
      return false;

    return lookupHash(RtxOptions::terrainTextures(), drawCallState.getMaterialData().getHash());
  }

  void DrawCallState::trackSkyCameraPosition(const Matrix4& worldToView, bool zWriteEnable, bool alphaBlend, bool hasSkinning,
                                             bool drawingToRaytracedRenderTarget, std::vector<Vector3>& seenCameraPositions) {
    // As shouldBakeSky: no camera position for render-to-texture draws, then add it unless one close to it is known.
    if (drawingToRaytracedRenderTarget) {
      return;
    }
    const std::optional<Vector3> position = makeCameraPosition(worldToView, zWriteEnable, alphaBlend, hasSkinning);
    if (!position) {
      return;
    }
    for (const Vector3& seen : seenCameraPositions) {
      if (areCamerasClose(seen, *position)) {
        return;
      }
    }
    seenCameraPositions.push_back(*position);
  }

  bool DrawCallState::predictSkyAutoDetect(const Matrix4& worldToView, bool zWriteEnable, bool alphaBlend, bool hasSkinning,
                                           bool drawingToRaytracedRenderTarget, bool zEnable,
                                           uint32_t prevFrameSeenCamerasCount,
                                           const std::vector<Vector3>& seenCameraPositions) {
    // As shouldBakeSky: render-to-texture draws are judged against no cameras, others after their own camera is added.
    if (drawingToRaytracedRenderTarget) {
      static const std::vector<Vector3> kNoCameras;
      return checkSkyAutoDetect(zEnable, std::nullopt, prevFrameSeenCamerasCount, kNoCameras);
    }
    const std::optional<Vector3> position = makeCameraPosition(worldToView, zWriteEnable, alphaBlend, hasSkinning);
    if (seenCameraPositions.size() >= 2) {
      // Adding a camera cannot change the result once two are known.
      return checkSkyAutoDetect(zEnable, position, prevFrameSeenCamerasCount, seenCameraPositions);
    }
    std::vector<Vector3> withDraw(seenCameraPositions);
    trackSkyCameraPosition(worldToView, zWriteEnable, alphaBlend, hasSkinning, false, withDraw);
    return checkSkyAutoDetect(zEnable, position, prevFrameSeenCamerasCount, withDraw);
  }

  void DrawCallState::setupCategoriesForHeuristics(uint32_t prevFrameSeenCamerasCount,
                                                   std::vector<Vector3>& seenCameraPositions) {
    const SkyDetectionSource skySource = shouldBakeSky(*this,
                                                       futureSkinningData.valid(),
                                                       prevFrameSeenCamerasCount,
                                                       seenCameraPositions);
    setCategory(InstanceCategories::Sky, skySource != SkyDetectionSource::None);
    skyAutoDetected = (skySource == SkyDetectionSource::AutoDetect);

    setCategory(InstanceCategories::Terrain, shouldBakeTerrain(*this));
  }

} // namespace dxvk
