// Copyright 2026 Remix Plus contributors
// SPDX-License-Identifier: MIT
#include "rtx_screen_lens_drops.h"

#include <cmath>
#include <charconv>
#include <chrono>
#include <algorithm>
#include <array>
#include <utility>

namespace dxvk {
  namespace {
    class WireReader {
    public:
      explicit WireReader(std::string_view wire) : m_remaining(wire) { }
      template<typename T> bool read(T& value) {
        trim();
        const size_t end = m_remaining.find_first_of(" \t\r\n\f\v");
        const auto token = m_remaining.substr(0, end);
        if (token.empty()) { return false; }
        const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
        m_remaining.remove_prefix(token.size());
        return result.ec == std::errc() && result.ptr == token.data() + token.size();
      }
      bool atEnd() { trim(); return m_remaining.empty(); }
    private:
      void trim() {
        const auto start = m_remaining.find_first_not_of(" \t\r\n\f\v");
        m_remaining.remove_prefix(start == std::string_view::npos ? m_remaining.size() : start);
      }
      std::string_view m_remaining;
    };
  }
  bool parseScreenLensDropFrame(std::string_view wire, ScreenLensDropFrame& output) {
    if (wire.empty() || wire.size() > 192 * 1024 || wire.find('\0') != std::string_view::npos) {
      return false;
    }
    WireReader stream(wire);
    ScreenLensDropFrame frame;
    uint32_t version = 0;
    uint32_t paused = 0;
    uint32_t count = 0;
    if (!stream.read(version) || !stream.read(frame.epoch) || !stream.read(frame.sequence)
        || !stream.read(frame.gameFrame) || !stream.read(frame.gameTimeMs)
        || !stream.read(paused) || !stream.read(frame.width) || !stream.read(frame.height)
        || !stream.read(frame.kinds) || !stream.read(count)
        || version != 1 || frame.epoch == 0 || frame.sequence == 0 || paused > 1
        || frame.width == 0 || frame.height == 0 || frame.width > 32768 || frame.height > 32768
        || frame.kinds == 0 || frame.kinds > 3 || count > kScreenLensDropCapacity) {
      return false;
    }
    frame.paused = paused != 0;
    frame.drops.reserve(count);
    std::array<uint32_t, kScreenLensDropCapacity> ids;
    for (uint32_t i = 0; i < count; ++i) {
      ScreenLensDrop drop;
      if (!stream.read(drop.id) || !stream.read(drop.kind) || !stream.read(drop.u) || !stream.read(drop.v)
          || !stream.read(drop.radiusX) || !stream.read(drop.radiusY) || !stream.read(drop.axisX)
          || !stream.read(drop.axisY) || !stream.read(drop.opacity) || !stream.read(drop.variant)) {
        return false;
      }
      const float axisLengthSquared = drop.axisX * drop.axisX + drop.axisY * drop.axisY;
      if ((drop.kind != 1 && drop.kind != 2) || drop.variant > 1
          || !std::isfinite(drop.u) || !std::isfinite(drop.v) || drop.u < -4 || drop.u > 5 || drop.v < -4 || drop.v > 5
          || !std::isfinite(drop.radiusX) || !std::isfinite(drop.radiusY)
          || drop.radiusX <= 0 || drop.radiusY <= 0 || drop.radiusX > 1 || drop.radiusY > 1
          || !std::isfinite(axisLengthSquared) || axisLengthSquared < 0.25f || axisLengthSquared > 2.25f
          || !std::isfinite(drop.opacity) || drop.opacity < 0 || drop.opacity > 1) {
        return false;
      }
      if (!(frame.kinds & (1u << (drop.kind - 1)))) { return false; }
      const float inverseLength = 1.0f / std::sqrt(axisLengthSquared);
      drop.axisX *= inverseLength;
      drop.axisY *= inverseLength;
      frame.drops.push_back(drop);
      ids[i] = drop.id;
    }
    std::sort(ids.begin(), ids.begin() + count);
    if (!stream.atEnd() || std::adjacent_find(ids.begin(), ids.begin() + count) != ids.begin() + count) {
      return false;
    }
    output = std::move(frame);
    return true;
  }

  ScreenLensDropState::ScreenLensDropState()
    : m_deviceToken(uint64_t(std::chrono::steady_clock::now().time_since_epoch().count())) { }

  bool ScreenLensDropState::reserve(const ScreenLensDropFrame& frame) {
    std::lock_guard lock(m_receptionMutex);
    if (frame.epoch == 0 || frame.sequence == 0
        || (frame.epoch == m_reservedEpoch && frame.sequence <= m_reservedSequence)) { return false; }
    m_reservedEpoch = frame.epoch;
    m_reservedSequence = frame.sequence;
    return true;
  }

  void ScreenLensDropState::commit(ScreenLensDropFrame&& frame, uint32_t renderFrame, uint64_t sceneEpoch) {
    m_frame = std::move(frame);
    m_renderFrame = renderFrame;
    m_sceneEpoch = sceneEpoch;
    m_consumed = false;
  }

  void ScreenLensDropState::cancel() {
    m_frame = {};
    m_consumed = true;
  }

  bool ScreenLensDropState::available(uint32_t renderFrame, uint64_t sceneEpoch) const {
    return !m_consumed && m_renderFrame == renderFrame && m_sceneEpoch == sceneEpoch;
  }
}
