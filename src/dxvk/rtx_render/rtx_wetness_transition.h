// SPDX-License-Identifier: MIT
// Local runtime feature source. See CREDITS.md.
#pragma once
#include <algorithm>
#include <cmath>

namespace dxvk {
  class WetnessTransition {
  public:
    float update(bool active, float deltaSeconds, float durationSeconds) {
      if (!std::isfinite(deltaSeconds) || deltaSeconds <= 0.0f) { return weight(); }
      const float duration = std::isfinite(durationSeconds) ? std::max(durationSeconds, 0.1f) : 15.0f;
      const float step = std::min(deltaSeconds, 1.0f) / duration;
      m_progress = std::clamp(m_progress + (active ? step : -step), 0.0f, 1.0f);
      return weight();
    }
    float weight() const { return m_progress * m_progress * (3.0f - 2.0f * m_progress); }
  private:
    float m_progress = 0.0f;
  };
}
