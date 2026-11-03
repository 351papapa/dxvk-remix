// SPDX-License-Identifier: MIT
// Local runtime feature source. See CREDITS.md.
#pragma once

#include <algorithm>
#include <cmath>

namespace dxvk {
  class WetnessRetention {
  public:
    void reset() {
      m_surfaceWetness = 0.0f;
      m_peakWetness = 0.0f;
      m_elapsedSeconds = 0.0;
    }

    float update(float rainfall, float deltaSeconds, float holdSeconds, float drySeconds, bool active) {
      const float rain = std::isfinite(rainfall) ? std::clamp(rainfall, 0.0f, 1.0f) : 0.0f;
      if (!active) {
        reset();
        return rain;
      }

      if (rain > 0.0f) {
        // New rain keeps the water that remains, without restoring a previous rain peak.
        m_surfaceWetness = std::max(m_surfaceWetness, rain);
        m_peakWetness = m_surfaceWetness;
        m_elapsedSeconds = 0.0;
        return m_surfaceWetness;
      }

      if (m_peakWetness <= 0.0f) {
        return 0.0f;
      }

      const float step = std::isfinite(deltaSeconds) ? std::clamp(deltaSeconds, 0.0f, 0.25f) : 0.0f;
      const double hold = std::isfinite(holdSeconds) ? std::max(holdSeconds, 0.0f) : 0.0;
      const double dry = std::isfinite(drySeconds) ? std::max(drySeconds, 0.0f) : 0.0;
      m_elapsedSeconds = std::min(m_elapsedSeconds + step, hold + dry);
      if (m_elapsedSeconds < hold) {
        return m_surfaceWetness;
      }

      const float progress = dry > 0.0 ? float((m_elapsedSeconds - hold) / dry) : 1.0f;
      const float fade = progress * progress * (3.0f - 2.0f * progress);
      m_surfaceWetness = std::min(m_surfaceWetness, m_peakWetness * (1.0f - fade));
      if (progress >= 1.0f) {
        reset();
      }
      return m_surfaceWetness;
    }

  private:
    float m_surfaceWetness = 0.0f;
    float m_peakWetness = 0.0f;
    double m_elapsedSeconds = 0.0;
  };
}
