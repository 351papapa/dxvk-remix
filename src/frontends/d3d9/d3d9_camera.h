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

// The D3D9 frontend's cameras on the render thread. Its draws carry their cameras: each draw prepared for the scene is
// classified by the translator's camera rules and sets its camera up through the scene interface.

#include "../../dxvk/rtx_render/rtx_frontend_interface.h"
#include "../translator/camera_classifier.h"
#include "d3d9_draw_call.h"

namespace dxvk {
  class RtxContext;

  namespace frontends {
    namespace d3d9 {

      // Classifies a draw's camera and sets that camera up from the draw's matrices, which are read as they are, after
      // rtx.fusedWorldViewMode's fix-up. The frame's Main camera is the backend's, whoever set it up.
      // Returns CameraClass::Unknown when the matrices describe no camera, or Sky for a Sky-category draw. Such a draw
      // sets up no camera.
      translator::CameraClass classifyDrawCamera(RtxContext& ctx, const DrawCallState& state);

      // The draw a camera is set up from, as logKeptCameraCut reports it.
      struct CameraSource {
        uint32_t drawCallId = 0;
        uint32_t vertexCount = 0;
        uint32_t primitiveCount = 0;
        uint32_t categories = 0;
        uint64_t textureHash = 0;
        bool skinned = false;
        bool programmableVS = false;
        bool vertexCapture = false;
        bool early = false;         // the app thread's feed for a draw its early change test skipped
        bool renderThread = false;  // set up by the render thread's own preparation, not the app thread's feed
      };

      // Render thread. Sets a camera up from a draw's matrices and records the draw when it sets up the frame's Main
      // camera.
      void applyDrawCamera(RtxContext& ctx, frontend::CameraView view, const Matrix4& worldToView, const Matrix4& viewToProjection,
                           const CameraSource& source);

      // Render thread. Logs a camera cut the scene kept since the last call, with the draws that set up the Main camera
      // in the cut frame and in the frame before it.
      void logKeptCameraCut(RtxContext& ctx);

    } // namespace d3d9
  } // namespace frontends
} // namespace dxvk
