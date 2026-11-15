// Copyright 2026 Remix Plus contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <mutex>

namespace dxvk {
  constexpr uint32_t kScreenLensDropCapacity = 1024;
  constexpr const char* kScreenLensDropFrameKey = "__lensDrops.frame_v1";
  constexpr const char* kScreenLensDropCapabilityKey = "__lensDrops.capability_v1";
  constexpr const char* kScreenLensDropOrderedCapabilityKey = "__lensDrops.orderedCapability_v1";

  struct ScreenLensDrop {
    uint32_t id = 0;
    uint32_t kind = 0;
    float u = 0, v = 0;
    float radiusX = 0, radiusY = 0;
    float axisX = 0, axisY = 1;
    float opacity = 0;
    uint32_t variant = 0;
  };

  struct ScreenLensDropFrame {
    uint64_t epoch = 0, sequence = 0;
    uint32_t gameFrame = 0, gameTimeMs = 0;
    bool paused = false;
    uint32_t width = 0, height = 0;
    uint32_t kinds = 0;
    std::vector<ScreenLensDrop> drops;
  };

  bool parseScreenLensDropFrame(std::string_view wire, ScreenLensDropFrame& output);

  // Device-owned transaction state. API reception reserves sequence numbers;
  // command-stream commit/consume never reuses a snapshot across render frames.
  class ScreenLensDropState {
  public:
    ScreenLensDropState();
    uint64_t deviceToken() const { return m_deviceToken; }
    bool reserve(const ScreenLensDropFrame& frame);
    void commit(ScreenLensDropFrame&& frame, uint32_t renderFrame, uint64_t sceneEpoch);
    void cancel();
    bool available(uint32_t renderFrame, uint64_t sceneEpoch) const;
    // Screen-space snapshots do not depend on a TLAS. A scene sync can advance its
    // epoch after normal injection; still finish this accepted present, never a later one.
    bool availableForPresent(uint32_t renderFrame) const { return !m_consumed && m_renderFrame == renderFrame; }
    void consume() { m_consumed = true; }
    const ScreenLensDropFrame& frame() const { return m_frame; }

  private:
    const uint64_t m_deviceToken;
    ScreenLensDropFrame m_frame;
    std::mutex m_receptionMutex;
    uint64_t m_reservedEpoch = 0, m_reservedSequence = 0;
    uint32_t m_renderFrame = 0;
    uint64_t m_sceneEpoch = 0;
    bool m_consumed = true;
  };
}
