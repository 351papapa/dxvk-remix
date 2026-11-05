// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string_view>

#include "../../util/util_vector.h"

namespace dxvk {

struct WaterOpticalProperties {
  Vector3 transmittance;
  float distance;
  float materialDistance;
  Vector3 extinction;
};

// Authored transmittance is the existing total extinction, linear at a world-space distance.
// Added neutral particles increase total extinction. A zero particle coefficient preserves the
// authored material exactly; scattering remains a fraction of the resulting neutral extinction.
inline WaterOpticalProperties waterOpticalProperties(Vector3 transmittance, float distance,
  float particleExtinction = 0.0f) {
  WaterOpticalProperties optics;
  optics.distance = std::isfinite(distance) ? std::clamp(distance, 0.1f, 100.0f) : 1.0f;
  for (uint32_t channel = 0; channel < 3; ++channel) {
    const float value = transmittance[channel];
    optics.transmittance[channel] = std::isfinite(value) ? std::clamp(value, 0.0001f, 1.0f) : 1.0f;
    optics.extinction[channel] = -std::log(optics.transmittance[channel]) / optics.distance;
  }
  optics.materialDistance = optics.distance;
  if (std::isfinite(particleExtinction) && particleExtinction > 0.0f) {
    // GPU materials store both values as half floats and clamp distance to at least .001.
    // Re-encode dense water at a shorter distance instead of clipping its total extinction
    // or flushing a subnormal transmission to zero. Keep the authored distance for volume range.
    constexpr float minimumMaterialDistance = 0.001f;
    constexpr float minimumMaterialTransmittance = 0.00006103515625f; // Smallest normal binary16.
    const float maximumOpticalDepth = -std::log(minimumMaterialTransmittance);
    const float maximumBaseExtinction = (std::max)({optics.extinction.x, optics.extinction.y, optics.extinction.z});
    const float maximumParticleExtinction = maximumOpticalDepth / minimumMaterialDistance - maximumBaseExtinction;
    const float particles = (std::min)(particleExtinction, maximumParticleExtinction);
    optics.extinction += Vector3(particles);
    const float maximumExtinction = maximumBaseExtinction + particles;
    optics.materialDistance = (std::max)(minimumMaterialDistance,
      (std::min)(optics.distance, maximumOpticalDepth / maximumExtinction));
    for (uint32_t channel = 0; channel < 3; ++channel) {
      optics.transmittance[channel] = std::exp(-optics.extinction[channel] * optics.materialDistance);
    }
  }
  return optics;
}

inline uint64_t parseWaterMediumHash(std::string_view text) {
  if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
  }
  if (text.empty()) {
    return 0;
  }
  uint64_t hash = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), hash, 16);
  return result.ec == std::errc() && result.ptr == text.data() + text.size() ? hash : 0;
}

struct WaterMediumState {
  bool active = false;
  bool changed = false;
  Vector3 extinction = Vector3(0.0f);
  Vector3 scattering = Vector3(0.0f);
  float anisotropy = 0.0f;
  float attenuationDistance = 1.0f;
};

inline WaterMediumState makeWaterMediumState(bool enabled, uint64_t configuredHash, uint64_t resolvedHash,
  WaterOpticalProperties optics, float scattering, float anisotropy, const WaterMediumState& previous) {
  WaterMediumState state;
  state.active = enabled && configuredHash != 0 && configuredHash == resolvedHash;
  if (state.active) {
    const float scatteringFraction = std::isfinite(scattering) ? std::clamp(scattering, 0.0f, 1.0f) : 0.0f;
    const float minimumExtinction = (std::min)({optics.extinction.x, optics.extinction.y, optics.extinction.z});
    state.extinction = optics.extinction;
    state.scattering = Vector3(minimumExtinction * scatteringFraction);
    state.anisotropy = std::isfinite(anisotropy) ? std::clamp(anisotropy, 0.0f, 0.8f) : 0.0f;
    state.attenuationDistance = optics.distance;
  }
  state.changed = state.active != previous.active || (state.active
    && (state.extinction != previous.extinction || state.scattering != previous.scattering
      || state.anisotropy != previous.anisotropy || state.attenuationDistance != previous.attenuationDistance));
  return state;
}

}
