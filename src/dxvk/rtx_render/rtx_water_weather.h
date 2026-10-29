// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cmath>
#include <string>

namespace dxvk {
struct WaterWeatherState {
  float rain = 0.0f;
  float storm = 0.0f;
  void normalize() {
    rain = std::isfinite(rain) ? std::clamp(rain, 0.0f, 1.0f) : 0.0f;
    storm = std::isfinite(storm) ? std::clamp(storm, 0.0f, 1.0f) : 0.0f;
    const float sum = rain + storm;
    if (sum > 1.0f) { rain /= sum; storm /= sum; }
  }
  static WaterWeatherState blend(WaterWeatherState a, WaterWeatherState b, float t) {
    a.normalize(); b.normalize();
    t = std::isfinite(t) ? std::clamp(t, 0.0f, 1.0f) : 0.0f;
    return {a.rain + (b.rain-a.rain)*t, a.storm + (b.storm-a.storm)*t};
  }
  float multiplier(float rainy, float stormy) const {
    auto weights = *this; weights.normalize();
    return 1.0f + weights.rain*(rainy-1.0f) + weights.storm*(stormy-1.0f);
  }
  static WaterWeatherState forName(std::string name) {
    for (auto& c : name) { if (c >= 'A' && c <= 'Z') { c += 'a'-'A'; } }
    if (name == "storm" || name == "thunderstorm") { return {0.0f, 1.0f}; }
    if (name == "drizzle" || name == "rainstorm") { return {1.0f, 0.0f}; }
    return {};
  }
};
}
