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

// Translator toolkit: what a frontend does with the draws of the game's sky camera. A sky draw is either rasterized
// into the backend's sky matte and probe (a sky quad), or it is a 3D-skybox object, which is moved into the main
// camera's world and ray traced.

#include <cstdint>

#include "../../util/util_matrix.h"

namespace dxvk {
  namespace frontend {
    struct CameraQuery;
  }

  namespace translator {

    // What the sky rules read from a sky-camera draw.
    struct SkyDraw {
      bool blending = false;
      uint32_t vertexCount = 0;
      // 0 when the draw is not indexed.
      uint32_t indexCount = 0;
      // The sky category came from rtx.skyAutoDetect, not from a texture list or the draw order.
      bool autoDetected = false;
    };

    // True for a sky quad: an unblended draw of at most six vertices or indices.
    bool isSkyQuad(const SkyDraw& draw);

    // True when a sky-camera draw is rasterized into the sky matte and probe.
    // False for a 3D-skybox draw.
    bool skyDrawRasterizes(const SkyDraw& draw);

    // The transforms of a 3D-skybox draw that the reprojection reads and writes.
    struct SkyTransforms {
      Matrix4 objectToWorld;
      Matrix4 worldToView;
      Matrix4 viewToProjection;
    };

    // Moves a 3D-skybox draw from the sky camera's space into the main camera's world, scaled by
    // rtx.skyReprojectScale. main is the Main camera query of the current frame.
    // Returns false, with the transforms unchanged, when the main camera was not set up this frame, or in the last
    // frame with allowStaleMainCamera.
    bool reprojectSkyToMain(SkyTransforms& transforms, const frontend::CameraQuery& main, uint32_t frame,
                            bool allowStaleMainCamera);

  } // namespace translator
} // namespace dxvk
