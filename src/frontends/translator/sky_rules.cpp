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
#include "sky_rules.h"

#include <cstring>

#include "MathLib/MathLib.h"

#include "../../dxvk/rtx_render/rtx_frontend_interface.h"
#include "../../dxvk/rtx_render/rtx_options.h"

namespace dxvk {
  namespace translator {

    namespace {
      // The projection with its near and far planes replaced.
      Matrix4d overrideNearFarPlanes(const Matrix4d& modifiedViewToProj, float nearPlane, float farPlane) {
        // MathLib works in single precision.
        Matrix4 floatModifiedViewToProj{ modifiedViewToProj };

        // The memcpy below relies on the two layouts having the same size.
        static_assert(sizeof(float4x4) == sizeof(floatModifiedViewToProj));

        uint32_t flags;
        float    cameraParams[PROJ_NUM];
        DecomposeProjection(
          NDC_D3D,
          NDC_D3D,
          *reinterpret_cast<float4x4*>(&floatModifiedViewToProj),
          &flags,
          cameraParams,
          nullptr,
          nullptr,
          nullptr,
          nullptr);

        float4x4 newProjection;
        newProjection.SetupByAngles(
          cameraParams[PROJ_ANGLEMINX],
          cameraParams[PROJ_ANGLEMAXX],
          cameraParams[PROJ_ANGLEMINY],
          cameraParams[PROJ_ANGLEMAXY],
          nearPlane,
          farPlane,
          flags);
        std::memcpy(&floatModifiedViewToProj, &newProjection, sizeof(float4x4));

        return Matrix4d{ floatModifiedViewToProj };
      }
    }

    bool isSkyQuad(const SkyDraw& draw) {
      if (draw.blending) {
        return false;
      }
      if (draw.indexCount == 0) {
        return draw.vertexCount <= 6;
      }
      return draw.indexCount <= 6;
    }

    bool skyDrawRasterizes(const SkyDraw& draw) {
      if (!RtxOptions::skyReprojectToMainCameraSpace()) {
        return true;
      }
      // rtx.skyForceAutoDetectedToReproject: a draw only auto-detected as sky may be world geometry that shares the
      // sky camera's position, which the matte would hide, so it is reprojected.
      if (RtxOptions::skyForceAutoDetectedToReproject() && draw.autoDetected) {
        return false;
      }
      return isSkyQuad(draw);
    }

    bool reprojectSkyToMain(SkyTransforms& transforms, const frontend::CameraQuery& main, uint32_t frame,
                            bool allowStaleMainCamera) {
      const uint32_t lastUpdate = main.lastSetupFrame;
      if (lastUpdate != frame && !(allowStaleMainCamera && lastUpdate + 1 == frame)) {
        return false;
      }

      const float scale = RtxOptions::skyReprojectScale();
      const Matrix4d scaleMatrix{
        scale, 0,     0,     0,
        0,     scale, 0,     0,
        0,     0,     scale, 0,
        0,     0,     0,     1,
      };
      const Matrix4d skyViewToProjection = overrideNearFarPlanes(
        transforms.viewToProjection,
        main.nearPlane,
        main.farPlane);
      const Matrix4d skyViewToMainWorld =
        main.preciseViewToWorld *
        (main.preciseProjectionToView * skyViewToProjection) *
        scaleMatrix;

      transforms.objectToWorld    = skyViewToMainWorld * transforms.worldToView * transforms.objectToWorld;
      transforms.worldToView      = main.renderWorldToView;
      transforms.viewToProjection = main.preciseViewToProjection;
      return true;
    }

  } // namespace translator
} // namespace dxvk
