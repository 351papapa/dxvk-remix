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
#include "d3d9_camera.h"

#include <cmath>

#include "../../util/log/log.h"
#include "../../util/util_once.h"

namespace dxvk {
  namespace frontends {
    namespace d3d9 {

      translator::CameraClass classifyDrawCamera(RtxContext& ctx, const DrawCallState& state) {
        // The render thread's rules keep only the last projection's decomposition.
        thread_local translator::CameraRules rules;

        const DrawCallTransforms& transforms = state.getTransformData();
        translator::CameraInputs camera;
        camera.worldToView = transforms.worldToView;
        camera.viewToProjection = transforms.viewToProjection;
        camera.objectToWorld = transforms.objectToWorld;
        camera.objectToView = transforms.objectToView;
        camera.maxZ = state.maxZ;
        camera.sky = state.testCategoryFlags(InstanceCategories::Sky);
        camera.drawingToRaytracedRenderTarget = state.isDrawingToRaytracedRenderTarget;
        camera.skinned = state.vertexBlend;

        // A draw is compared with the Main camera of this frame, which an earlier draw, the app thread's camera feed
        // or an API camera may have set up.
        frontend::SceneInterface scene { ctx };
        const uint32_t frame = scene.frameIndex();
        const frontend::CameraFov mainFov = scene.cameraFov(frontend::CameraView::Main);
        translator::MainCamera main;
        main.set = mainFov.lastSetupFrame == frame;
        main.known = mainFov.lastSetupFrame != kInvalidFrameIndex;
        main.fov = mainFov.fov;

        bool updatesCamera = false;
        const translator::CameraClass cameraClass = rules.classify(camera, main, updatesCamera, true);
        if (!updatesCamera) {
          return cameraClass;
        }

        const frontend::CameraView view = translator::toView(cameraClass);
        if (frame > 0) {
          const frontend::CameraFov last = scene.cameraFov(view);
          if (last.lastSetupFrame == frame - 1 &&
              !(std::abs(rules.fov() - last.fov) < translator::CameraRules::kFovToleranceRadians)) {
            ONCE(Logger::warn("[RTX] CameraManager: FOV of a camera changed between frames"));
          }
        }
        CameraSource source;
        source.drawCallId = state.drawCallID;
        source.vertexCount = state.getGeometryData().vertexCount;
        source.primitiveCount = state.getGeometryData().calculatePrimitiveCount();
        source.categories = state.getCategoryFlags().raw();
        source.textureHash = state.getMaterialData().getColorTexture().getImageHash();
        source.skinned = camera.skinned;
        source.programmableVS = state.usesVertexShader;
        source.vertexCapture = state.vertexCaptured;
        source.renderThread = true;
        applyDrawCamera(ctx, view, camera.worldToView, camera.viewToProjection, source);
        return cameraClass;
      }

      namespace {
        // The draws that set up the Main camera in the last frames, by frame.
        struct MainCameraRecord {
          uint32_t frame = kInvalidFrameIndex;
          CameraSource source;
          Matrix4 worldToView;
          Matrix4 viewToProjection;
        };
        constexpr uint32_t kMainCameraRecords = 8;
        MainCameraRecord g_mainCameraRecords[kMainCameraRecords];
        uint32_t g_loggedKeptCut = kInvalidFrameIndex;

        const MainCameraRecord* findRecord(uint32_t frame) {
          const MainCameraRecord& record = g_mainCameraRecords[frame % kMainCameraRecords];
          return record.frame == frame ? &record : nullptr;
        }

        std::string describe(const MainCameraRecord* record) {
          if (record == nullptr) {
            return "no D3D9 draw (an API camera, or none)";
          }
          const CameraSource& s = record->source;
          DecomposeProjectionParams projection {};
          decomposeProjection(record->viewToProjection, projection);
          const Vector3 position = inverseAffine(record->worldToView)[3].xyz();
          return str::format("pos=(", position.x, ", ", position.y, ", ", position.z, ") draw #", s.drawCallId,
                             s.renderThread ? " (render thread)" : s.early ? " (app thread, early skip)" : " (app thread)",
                             " skinned=", s.skinned ? 1 : 0, " programmableVS=", s.programmableVS ? 1 : 0,
                             " vertexCapture=", s.vertexCapture ? 1 : 0, " categories=0x", std::hex, s.categories,
                             " texture=0x", s.textureHash, std::dec, " vertices=", s.vertexCount, " primitives=", s.primitiveCount,
                             " fovY=", projection.fov * (180.f / 3.14159265f), " near=", projection.nearPlane, " far=", projection.farPlane);
        }
      }

      void applyDrawCamera(RtxContext& ctx, frontend::CameraView view, const Matrix4& worldToView, const Matrix4& viewToProjection,
                           const CameraSource& source) {
        frontend::SceneInterface scene { ctx };
        const uint32_t frame = scene.frameIndex();
        const bool mainSetBefore = view == frontend::CameraView::Main &&
                                   scene.cameraFov(frontend::CameraView::Main).lastSetupFrame == frame;
        scene.applyDrawCamera(view, worldToView, viewToProjection);
        if (view == frontend::CameraView::Main && !mainSetBefore &&
            scene.cameraFov(frontend::CameraView::Main).lastSetupFrame == frame) {
          MainCameraRecord& record = g_mainCameraRecords[frame % kMainCameraRecords];
          record.frame = frame;
          record.source = source;
          record.worldToView = worldToView;
          record.viewToProjection = viewToProjection;
        }
      }

      void logKeptCameraCut(RtxContext& ctx) {
        const frontend::CameraQuery main = frontend::SceneInterface { ctx }.camera(frontend::CameraView::Main);
        if (main.keptCutFrame == kInvalidFrameIndex || main.keptCutFrame == g_loggedKeptCut) {
          return;
        }
        g_loggedKeptCut = main.keptCutFrame;
        Logger::info(str::format("[D3D9v2] camera cut kept at frame ", main.keptCutFrame, ": Main camera set up by ",
                                 describe(findRecord(main.keptCutFrame)), " | the frame before: ",
                                 describe(findRecord(main.keptCutFrame - 1))));
      }

    } // namespace d3d9
  } // namespace frontends
} // namespace dxvk
