/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_camera_manager.h"

#include "dxvk_device.h"
#include "rtx_fork_camera_origin.h"

namespace dxvk {

  CameraManager::CameraManager(DxvkDevice* device) : CommonDeviceObject(device) {
    for (int i = 0; i < CameraType::Count; i++) {
      m_cameras[i].setCameraType(CameraType::Enum(i));
    }
  }

  bool CameraManager::isCameraValid(CameraType::Enum cameraType) const {
    assert(cameraType < CameraType::Enum::Count);
    return accessCamera(*this, cameraType).isValid(m_device->getCurrentFrameId());
  }

  void CameraManager::onFrameEnd() {
    m_lastSetCameraType = CameraType::Unknown;
    m_decompositionCache.clear();
  }

  void CameraManager::applyFrameCamera(CameraType::Enum cameraType, const Matrix4& worldToView, const Matrix4& viewToProjection) {
    const uint32_t frameId = m_device->getCurrentFrameId();
    const DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(viewToProjection);
    auto& camera = getCamera(cameraType);
    auto cameraSequence = RtCameraSequence::getInstance();
    bool shouldUpdateMainCamera = cameraType == CameraType::Main && camera.getLastUpdateFrame() != frameId;
    bool isPlaying = RtCameraSequence::mode() == RtCameraSequence::Mode::Playback;
    bool isBrowsing = RtCameraSequence::mode() == RtCameraSequence::Mode::Browse;
    bool isCameraCut = false;
    if (isPlaying || isBrowsing) {
      if (shouldUpdateMainCamera) {
        RtCamera::RtCameraSetting setting;
        cameraSequence->getRecord(cameraSequence->currentFrame(), setting);
        isCameraCut = camera.updateFromSetting(frameId, setting, 0);

        if (isPlaying) {
          cameraSequence->goToNextFrame();
        }
      }
    } else {
      isCameraCut = camera.update(
        frameId,
        worldToView,
        viewToProjection,
        decomposeProjectionParams.fov,
        decomposeProjectionParams.aspectRatio,
        decomposeProjectionParams.nearPlane,
        decomposeProjectionParams.farPlane,
        decomposeProjectionParams.isLHS
      );
    }


    if (shouldUpdateMainCamera && RtCameraSequence::mode() == RtCameraSequence::Mode::Record) {
      auto& setting = camera.getSetting();
      cameraSequence->addRecord(setting);
    }

    // Register camera cut when there are significant interruptions to the view (like changing level, or opening a menu)
    if (isCameraCut && cameraType == CameraType::Main) {
      m_lastCameraCutFrameId = m_device->getCurrentFrameId();
    }
    m_lastSetCameraType = cameraType;
  }

  bool CameraManager::isCameraCutThisFrame() const {
    return m_lastCameraCutFrameId == m_device->getCurrentFrameId();
  }

  void CameraManager::signalCameraCut() {
    getMainCamera().invalidateViewHistory(m_device->getCurrentFrameId());
  }

  void CameraManager::shiftCameraHistory(CameraType::Enum type, const Vector3& offset) {
    if (type >= CameraType::Count) {
      return;
    }
    RtCamera& camera = getCamera(type);
    if (camera.getLastUpdateFrame() != m_device->getCurrentFrameId() && offset != Vector3(0.0f)) {
      camera.applyArtificialWorldOffset(offset);
    }
  }

  void CameraManager::processExternalCamera(CameraType::Enum type,
                                            const Matrix4& worldToView,
                                            const Matrix4& viewToProjection,
                                            const Vector3* worldOrigin,
                                            const Matrix4d* preciseWorldToView) {
    if (worldOrigin != nullptr && type < CameraType::Count) {
      // Ported from BetaRT (sambow23/dxvk-remix-gmod, commit 11761453a). Shift last frame's camera
      // into this frame's render origin before update() turns it into the previous camera, so an origin move is
      // neither motion nor a camera cut. Only once per frame: update() ignores a second call in the same frame.
      RtCamera& camera = getCamera(type);
      if (camera.getLastUpdateFrame() != m_device->getCurrentFrameId()) {
        const Vector3 previousHistoryOffset = fork_camera_origin::calculatePreviousCameraHistoryOffset(
          m_externalCameraOriginValid[type], m_externalCameraOrigins[type], *worldOrigin);
        if (previousHistoryOffset != Vector3(0.0f)) {
          camera.applyArtificialWorldOffset(previousHistoryOffset);
        }
        m_externalCameraOrigins[type] = *worldOrigin;
        m_externalCameraOriginValid[type] = true;
      }
    }
    updateExternalCamera(type, worldToView, viewToProjection, preciseWorldToView);
  }

  void CameraManager::processExternalCamera(CameraType::Enum type,
                                            const Matrix4& worldToView,
                                            const Matrix4& viewToProjection) {
    updateExternalCamera(type, worldToView, viewToProjection, nullptr);
  }

  void CameraManager::updateExternalCamera(CameraType::Enum type,
                                           const Matrix4& worldToView,
                                           const Matrix4& viewToProjection,
                                           const Matrix4d* preciseWorldToView) {
    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(viewToProjection);

    getCamera(type).update(
      m_device->getCurrentFrameId(),
      worldToView,
      viewToProjection,
      decomposeProjectionParams.fov,
      decomposeProjectionParams.aspectRatio,
      decomposeProjectionParams.nearPlane,
      decomposeProjectionParams.farPlane,
      decomposeProjectionParams.isLHS,
      (uint32_t) RtCamera::UpdateFlag::UpdateNormal,
      preciseWorldToView);
  }

    DecomposeProjectionParams CameraManager::getOrDecomposeProjection(const Matrix4& viewToProjection) {
      XXH64_hash_t projectionHash = XXH64(&viewToProjection, sizeof(viewToProjection), 0);
      auto iter = m_decompositionCache.find(projectionHash);
      if (iter != m_decompositionCache.end()) {
        return iter->second;
      }

      DecomposeProjectionParams decomposeProjectionParams;
      decomposeProjection(viewToProjection, decomposeProjectionParams);
      m_decompositionCache.emplace(projectionHash, decomposeProjectionParams);
      return decomposeProjectionParams;
    }
}  // namespace dxvk
