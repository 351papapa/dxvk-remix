/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_context.h"

#include "dxvk_device.h"
#include "rtx_resources.h"
#include "rtx_scene_manager.h"

namespace dxvk {

  static Matrix4 makeViewMatrixForCubePlane(uint32_t plane, const Vector3& cameraPosition) {
    assert(plane >= 0 && plane < 6);

    constexpr static Vector3 targets[]{
      Vector3{+1,  0,  0},
      Vector3{-1,  0,  0},
      Vector3{ 0, +1,  0},
      Vector3{ 0, -1,  0},
      Vector3{ 0,  0, +1},
      Vector3{ 0,  0, -1},
    };
    constexpr static Vector3 ups[]{
      Vector3{ 0, +1,  0},
      Vector3{ 0, +1,  0},
      Vector3{ 0,  0, -1},
      Vector3{ 0,  0, +1},
      Vector3{ 0, +1,  0},
      Vector3{ 0, +1,  0},
    };
    const static Vector3 axisZ[]{
      targets[0],
      targets[1],
      targets[2],
      targets[3],
      targets[4],
      targets[5],
    };
    const static Vector3 axisX[]{
      cross(ups[0], axisZ[0]),
      cross(ups[1], axisZ[1]),
      cross(ups[2], axisZ[2]),
      cross(ups[3], axisZ[3]),
      cross(ups[4], axisZ[4]),
      cross(ups[5], axisZ[5]),
    };
    const static Vector3 axisY[]{
      cross(axisZ[0], axisX[0]),
      cross(axisZ[1], axisX[1]),
      cross(axisZ[2], axisX[2]),
      cross(axisZ[3], axisX[3]),
      cross(axisZ[4], axisX[4]),
      cross(axisZ[5], axisX[5]),
    };
    assert(isApproxNormalized(axisX[0], 0.0001f) && isApproxNormalized(axisY[0], 0.0001f) && isApproxNormalized(axisZ[0], 0.0001f));
    assert(isApproxNormalized(axisX[1], 0.0001f) && isApproxNormalized(axisY[1], 0.0001f) && isApproxNormalized(axisZ[1], 0.0001f));
    assert(isApproxNormalized(axisX[2], 0.0001f) && isApproxNormalized(axisY[2], 0.0001f) && isApproxNormalized(axisZ[2], 0.0001f));
    assert(isApproxNormalized(axisX[3], 0.0001f) && isApproxNormalized(axisY[3], 0.0001f) && isApproxNormalized(axisZ[3], 0.0001f));
    assert(isApproxNormalized(axisX[4], 0.0001f) && isApproxNormalized(axisY[4], 0.0001f) && isApproxNormalized(axisZ[4], 0.0001f));
    assert(isApproxNormalized(axisX[5], 0.0001f) && isApproxNormalized(axisY[5], 0.0001f) && isApproxNormalized(axisZ[5], 0.0001f));

    const Vector3 translation{
      dot(axisX[plane], -cameraPosition),
      dot(axisY[plane], -cameraPosition),
      dot(axisZ[plane], -cameraPosition),
    };

    return Matrix4{
      axisX[plane].x, axisY[plane].x, axisZ[plane].x, 0.f,
      axisX[plane].y, axisY[plane].y, axisZ[plane].y, 0.f,
      axisX[plane].z, axisY[plane].z, axisZ[plane].z, 0.f,
      translation.x,  translation.y,  translation.z,  1.f,
    };
  }

} // namespace dxvk

void dxvk::RtxContext::initSkyTargets() {
  // Use game render target format for sky render target views whether it is linear, HDR or sRGB -- to render into the images
  m_skyRtColorFormat = m_state.om.renderTargets.color[0].view->image()->info().format;
  // Use sRGB (or linear for HDR formats) for image and sampling views -- to use in ray tracing
  m_skyColorFormat = TextureUtils::toSRGB(m_skyRtColorFormat);
  if (RtxOptions::skyForceHDR()) {
    m_skyRtColorFormat = m_skyColorFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
  }

  getResourceManager().getCompatibleViewForView(
    getResourceManager().getSkyMatte(this, m_skyColorFormat).view,
    m_skyRtColorFormat);
  initSkyProbe();
}
