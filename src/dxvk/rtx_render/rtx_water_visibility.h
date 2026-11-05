// SPDX-License-Identifier: MIT
#pragma once

#include "rtx_water_medium.h"
#include "rtx_water_weather.h"

namespace dxvk {

class NumosWeather;

// Modify only the coefficients sent to the camera volume. The tracked medium
// and the surface material keep their stable optical identity.
inline WaterMediumState applyWaterWeatherParticles(WaterMediumState medium, WaterWeatherState weather,
    float rainy, float stormy, float unitsPerMeter, float scatteringFraction) {
  if (!medium.active || !std::isfinite(unitsPerMeter) || unitsPerMeter <= 0.0f) {
    return medium;
  }
  const auto unitRange = [](float value) {
    return std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f;
  };
  weather.normalize();
  const float perMeter = weather.rain * unitRange(rainy) + weather.storm * unitRange(stormy);
  const float extinction = perMeter / (std::max)(unitsPerMeter, 0.000001f);
  medium.extinction += Vector3(extinction);
  medium.scattering += Vector3(extinction * unitRange(scatteringFraction));
  return medium;
}

WaterMediumState getWeatherCameraWaterMedium(const WaterMediumState& medium, const NumosWeather& weather);

} // namespace dxvk
