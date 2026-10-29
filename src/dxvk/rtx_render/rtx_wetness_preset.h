// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cmath>
#include <optional>

#include "rtx/utility/shader_types.h"
#include "rtx/pass/wetness_args.h"

namespace dxvk {
  struct WetnessParameter {
    const char* name;
    float WetnessArgs::* member;
  };

  inline constexpr WetnessParameter kWetnessParameters[] = {
    { "puddleNoiseScale", &WetnessArgs::puddleNoiseScale },
    { "puddleThresholdBase", &WetnessArgs::puddleThresholdBase },
    { "puddleThresholdSpan", &WetnessArgs::puddleThresholdSpan },
    { "puddleAlbedoDarken", &WetnessArgs::puddleAlbedoDarken },
    { "puddleWaveStrength", &WetnessArgs::puddleWaveStrength },
    { "puddleWaveScale", &WetnessArgs::puddleWaveScale },
    { "rippleGridScale", &WetnessArgs::rippleGridScale },
    { "rippleSpawnPeriod", &WetnessArgs::rippleSpawnPeriod },
    { "rippleRadius", &WetnessArgs::rippleRadius },
    { "rippleVehicleGridScale", &WetnessArgs::rippleVehicleGridScale },
    { "rippleVehicleSpawnPeriod", &WetnessArgs::rippleVehicleSpawnPeriod },
    { "rippleVehicleRadius", &WetnessArgs::rippleVehicleRadius },
    { "dropDensity", &WetnessArgs::dropDensity },
    { "dropSize", &WetnessArgs::dropSize },
    { "dropStrength", &WetnessArgs::dropStrength },
    { "dropRoughness", &WetnessArgs::dropRoughness },
    { "streakGridScale", &WetnessArgs::streakGridScale },
    { "streakCycle", &WetnessArgs::streakCycle },
    { "streakLength", &WetnessArgs::streakLength },
    { "streakSize", &WetnessArgs::streakSize },
    { "waterWaveStrength", &WetnessArgs::waterWaveStrength },
    { "waterWaveScale", &WetnessArgs::waterWaveScale },
    { "rainIntensity", &WetnessArgs::weatherWetness },
  };

  struct WetnessPreset {
    std::array<std::optional<float>, std::size(kWetnessParameters)> values;
    std::array<bool, std::size(kWetnessParameters)> relative {};

    void apply(WetnessArgs& args) const {
      for (size_t i = 0; i < values.size(); ++i) {
        if (!values[i].has_value()) {
          continue;
        }
        const auto member = kWetnessParameters[i].member;
        const float v = relative[i] ? args.*member * *values[i] : *values[i];
        if (std::isfinite(v)) {
          args.*member = std::clamp(v, 0.0f, 100.0f);
          if (member == &WetnessArgs::weatherWetness) {
            args.weatherWetness = std::min(args.weatherWetness, 1.0f);
            args.weatherWetnessActive = 65535u;
          }
        }
      }
    }

    static WetnessArgs blend(const WetnessArgs& a, const WetnessArgs& b, float t) {
      t = std::isfinite(t) ? std::clamp(t, 0.0f, 1.0f) : 0.0f;
      WetnessArgs out = b;
      for (const auto& parameter : kWetnessParameters) {
        const auto member = parameter.member;
        out.*member = a.*member + (b.*member - a.*member) * t;
      }
      const float aWeight = float(a.weatherWetnessActive) / 65535.0f * (1.0f - t);
      const float bWeight = float(b.weatherWetnessActive) / 65535.0f * t;
      const float weight = aWeight + bWeight;
      out.weatherWetnessActive = uint32_t(std::round(weight * 65535.0f));
      out.weatherWetness = weight > 0.0f ? (a.weatherWetness * aWeight + b.weatherWetness * bWeight) / weight : 1.0f;
      return out;
    }
  };
}
