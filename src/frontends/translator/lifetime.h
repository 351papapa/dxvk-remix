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

// Translator toolkit: how long an object the game stopped drawing stays in the scene. The policy is the backend's
// (rtx_translator_lifetime.h), which the backend's own ageing shares, so a frontend ages its objects as the backend
// ages replacement instances and game lights. This header feeds the policy from the scene interface.

#include "../../dxvk/rtx_render/rtx_frontend_interface.h"
#include "../../dxvk/rtx_render/translator/rtx_translator_lifetime.h"

namespace dxvk {
  namespace translator {

    // The inputs of one collection, from the Main camera query. scene must outlive the returned frame.
    inline LifetimeFrame frontendLifetime(const frontend::CameraQuery& main, const frontend::SceneInterface& scene,
                                          bool forceCollect) {
      LifetimeFrame frame = LifetimeFrame::fromOptions(forceCollect);
      frame.intersectsFrustum = [](void* source, const AxisAlignedBoundingBox& bounds, const Matrix4& objectToWorld) {
        return static_cast<const frontend::SceneInterface*>(source)->intersectsFrustum(frontend::CameraView::Main, bounds,
                                                                                      objectToWorld);
      };
      frame.frustumSource = const_cast<frontend::SceneInterface*>(&scene);
      frame.antiCullingSupported = main.antiCullingSupported;
      frame.keptCutFrame = main.keptCutFrame;
      return frame;
    }

  } // namespace translator
} // namespace dxvk
