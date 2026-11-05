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
#pragma once

// The D3D9 frontend's record of one D3D9 draw. The app half (D3D9Rtx)
// fills it and queues it for the CS half, which finalizes it and describes it to the scene. It holds the pending
// geometry hashes, bounds and skinning of the geometry workers, the D3D9 state the frontend reads and the D3D9
// category and sky heuristics. The scene gets none of it: the frontend describes the draw (SceneInstanceDesc).
// The API-only build compiles it too, since D3D9Rtx holds the records outside the frontend's guards.

#include <vector>

#include "../../dxso/dxso_common.h"
#include "../../dxvk/rtx_render/rtx_frontend_interface.h"
#include "../../dxvk/rtx_render/rtx_materials.h"
#include "../../dxvk/rtx_render/rtx_scene_types.h"
#include "../../util/util_threadpool.h"
#include "../translator/camera_classifier.h"

namespace dxvk {
  struct D3D9Rtx;

  struct DrawCallState {
    // A DrawCallState owns its pending futures, so it moves and is never copied.
    DrawCallState() = default;
    DrawCallState(DrawCallState&&) = default;
    DrawCallState& operator=(DrawCallState&&) = default;
    DrawCallState(const DrawCallState&) = delete;
    DrawCallState& operator=(const DrawCallState&) = delete;

    // The asset hashes, as SceneDraw::getHash and getHashLegacy compute them. They use the original material.
    const XXH64_hash_t getHash(const HashRule& rule) const {
      return geometryData.getHashForRule(rule) ^ materialData.getHash();
    }

    const XXH64_hash_t getHashLegacy(const HashRule& rule) const {
      return geometryData.getHashForRuleLegacy(rule) ^ materialData.getHash();
    }

    const RasterGeometry& getGeometryData() const {
      return geometryData;
    }

    RasterGeometry& modifyGeometryData() {
      return geometryData;
    }

    const LegacyMaterialData& getMaterialData() const {
      return materialData;
    }

    LegacyMaterialData& modifyMaterialData() {
      return materialData;
    }

    const DrawCallTransforms& getTransformData() const {
      return transformData;
    }

    DrawCallTransforms& modifyTransformData() {
      return transformData;
    }

    const SkinningData& getSkinningState() const {
      return skinningData;
    }

    const CategoryFlags getCategoryFlags() const {
      return categories;
    }

    CategoryFlags& modifyCategoryFlags() {
      return categories;
    }

    template<typename... InstanceCategories>
    bool testCategoryFlags(InstanceCategories... cat) const { return categories.any(cat...); }

    bool hasTextureCoordinates() const {
      return geometryData.texcoordBuffer.defined() || transformData.texgenMode != TexGenMode::None;
    }

    const FogState& getFogState() const {
      return fogState;
    }

    // Resolves the geometry hashes, bounds and skinning. skinningCamera decomposes fixed-function skinned matrices, and
    // is null when that camera is not set up.
    // Returns false when the geometry hashes failed.
    bool finalizePendingFutures(const frontend::CameraQuery* skinningCamera);
    bool geometryHashesPending() const {
      return futureGeometryHashes.valid();
    }
    bool skinningPending() const {
      return futureSkinningData.valid();
    }

    // The draw's camera class
    translator::CameraClass cameraType = translator::CameraClass::Unknown;

    // Fixed-function vertex blending is on (no vertex shader, D3DRS_VERTEXBLEND set), whatever bones the skinning keeps.
    bool vertexBlend = false;

    // The view the app thread decomposes this draw's fixed-function skinning with: the frame camera of the draw's class
    // as it was at the draw (DrawInputs and the skinning match the draw with the same view). Unset before any camera.
    bool hasSkinningView = false;
    bool skinningViewThisFrame = false;  // set up this frame, not the last
    Matrix4 skinningWorldToView;

    // Uses a programmable vertex shader
    bool usesVertexShader = false;

    uint32_t drawCallID = 0;
    uint32_t wetnessTag = 0;

    // Snapshot of the caller's capture correlation at draw submission, before asynchronous execution.
    uint32_t captureSession = 0;
    uint32_t captureToken = 0;

    bool isUsingRaytracedRenderTarget = false;

    bool stencilEnabled = false;

    // Uses a programmable pixel shader
    bool usesPixelShader = false;

    // Contains valid values only if usesVertex/PixelShader is set
    DxsoProgramInfo programmableVertexShaderInfo;
    DxsoProgramInfo programmablePixelShaderInfo;

    float minZ = 0.0f;
    float maxZ = 1.0f;

    bool zWriteEnable = false;
    bool zEnable = false;

    bool isDrawingToRaytracedRenderTarget = false;

    // Set when the Sky category was assigned by skyAutoDetect heuristic
    // (as opposed to explicit methods like skyBoxTextures/skyBoxGeometries/skyMinZThreshold).
    // The D3D9 frontend's sky rules may then reproject the draw instead of rasterizing it into the sky probe,
    // since it may be world geometry that should go through reprojection instead.
    bool skyAutoDetected = false;

    // D3D9 draw facts the D3D9 frontend reads: a description of such a draw is deforming (its geometry changes between
    // descriptions).
    // vertexCaptured: positions come from vertex shader capture. dynamicSource: the position stream is a
    // D3DUSAGE_DYNAMIC vertex buffer or user-pointer (DrawPrimitiveUP) data.
    bool vertexCaptured = false;
    bool dynamicSource = false;

    // pCachedTextureListCategories: computeTextureListCategories() of the color texture hash, when the
    // caller has it cached (rtx.textureCategoryCache); nullptr = look the lists up now.
    void setupCategoriesForTexture(const CategoryFlags* pCachedTextureListCategories = nullptr);
    static CategoryFlags computeTextureListCategories(const XXH64_hash_t& textureHash);
    void setupCategoriesForGeometry();
    void setupCategoriesForHeuristics(uint32_t prevFrameSeenCamerasCount,
                                      std::vector<Vector3>& seenCameraPositions);
    // The camera-position step of the sky heuristic (setupCategoriesForHeuristics records every draw's camera in
    // seenCameraPositions; the next frame's auto-detection counts them), for a draw D3D9 v2 skips without preparing it.
    // Same rule as in shouldBakeSky.
    static void trackSkyCameraPosition(const Matrix4& worldToView, bool zWriteEnable, bool alphaBlend, bool hasSkinning,
                                       bool drawingToRaytracedRenderTarget, std::vector<Vector3>& seenCameraPositions);
    // The camera-position sky heuristic's result for a draw that is not sky by an explicit rule, as
    // setupCategoriesForHeuristics would derive it now, without recording the draw's camera. D3D9 v2 compares it with a
    // skipped draw's stored result.
    static bool predictSkyAutoDetect(const Matrix4& worldToView, bool zWriteEnable, bool alphaBlend, bool hasSkinning,
                                     bool drawingToRaytracedRenderTarget, bool zEnable, uint32_t prevFrameSeenCamerasCount,
                                     const std::vector<Vector3>& seenCameraPositions);

  private:
    friend struct D3D9Rtx;

    // NOTE: 'setCategory' can only add a category, it will not unset a bit
    void setCategory(InstanceCategories category, bool set);
    void removeCategory(InstanceCategories category);

    bool finalizeGeometryHashes();
    void finalizeGeometryBoundingBox();
    void finalizeSkinningData(const frontend::CameraQuery* skinningCamera);

    RasterGeometry geometryData;
    LegacyMaterialData materialData;
    DrawCallTransforms transformData;
    SkinningData skinningData;
    CategoryFlags categories = 0;
    FogState fogState;

    Future<GeometryHashes> futureGeometryHashes;
    Future<AxisAlignedBoundingBox> futureBoundingBox;
    Future<SkinningData> futureSkinningData;
  };

} // namespace dxvk
