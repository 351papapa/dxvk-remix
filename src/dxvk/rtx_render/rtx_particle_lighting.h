#pragma once

#include <algorithm>
#include <cmath>

namespace dxvk {

// Only the surface consumer is rescaled. The fog/cache writer and NUMOS radiometry are unchanged.
// A skipped or nearly disabled cache must never produce an unbounded inverse gain.
inline float particleFroxelScale(float consumerGain, float particleGain, bool enabled, bool rawCache = false) {
  constexpr float kMinimumConsumerGain = 1.0e-6f;
  constexpr float kMaximumParticleGain = 32.0f;
  constexpr float kMaximumConsumerScale = 4096.0f;
  if (!enabled || !std::isfinite(consumerGain) || consumerGain <= kMinimumConsumerGain
      || !std::isfinite(particleGain) || particleGain < 0.0f) {
    return 1.0f;
  }
  const float targetGain = std::min(particleGain, kMaximumParticleGain);
  // Translucent particles read pristine evalVolumetricNEE: its result has no consumer gain to undo.
  const float scale = rawCache ? targetGain : targetGain / consumerGain;
  // Treat an excessive inverse as a nearly disabled cache rather than amplifying tiny numerical noise.
  return std::isfinite(scale) && scale <= kMaximumConsumerScale ? scale : 1.0f;
}

} // namespace dxvk
