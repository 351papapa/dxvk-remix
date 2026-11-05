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
#include "rtx_translator_lifetime.h"

#include "../rtx_camera.h"
#include "../rtx_intersection_test.h"

namespace dxvk {
  namespace translator {

    bool aabbIntersectsFrustum(RtCamera& camera, const AxisAlignedBoundingBox& aabb, const Matrix4& objectToWorld) {
      const Matrix4 objectToView = camera.getWorldToView(false) * objectToWorld;
      if (RtxOptions::AntiCulling::Object::enableHighPrecisionAntiCulling()) {
        return boundingBoxIntersectsFrustumSAT(
          camera, aabb.minPos, aabb.maxPos, objectToView,
          RtxOptions::AntiCulling::Object::enableInfinityFarFrustum());
      }
      return boundingBoxIntersectsFrustum(camera.getFrustum(), aabb.minPos, aabb.maxPos, objectToView);
    }

    LifetimeFrame LifetimeFrame::read(RtCamera& camera, bool antiCullingSupported, bool forceCollect) {
      LifetimeFrame frame = fromOptions(forceCollect);
      frame.intersectsFrustum = [](void* source, const AxisAlignedBoundingBox& bounds, const Matrix4& objectToWorld) {
        return aabbIntersectsFrustum(*static_cast<RtCamera*>(source), bounds, objectToWorld);
      };
      frame.frustumSource = &camera;
      frame.antiCullingSupported = antiCullingSupported;
      return frame;
    }

  } // namespace translator
} // namespace dxvk
