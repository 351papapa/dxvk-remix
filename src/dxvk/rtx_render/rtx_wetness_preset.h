// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cmath>
#include <optional>

#include "rtx/utility/shader_types.h"
#include "rtx/pass/wetness_args.h"

namespace dxvk {
  enum class WetnessBlend { Linear, Step };

  struct WetnessParameter {
    const char* name;
    float WetnessArgs::* member;
    WetnessBlend blend = WetnessBlend::Linear;
  };

  inline constexpr WetnessParameter kWetnessParameters[] = {
    { "puddleCellSize", &WetnessArgs::puddleCellSize, WetnessBlend::Step },
    { "puddleCoverage", &WetnessArgs::puddleCoverage, WetnessBlend::Step },
    { "puddleMicroAmplitude", &WetnessArgs::puddleMicroAmplitude, WetnessBlend::Step },
    { "puddleAlbedoDarken", &WetnessArgs::puddleAlbedoDarken },
    { "puddleWaveStrength", &WetnessArgs::puddleWaveStrength, WetnessBlend::Step },
    { "puddleWaveScale", &WetnessArgs::puddleWaveScale, WetnessBlend::Step },
    { "rippleGridScale", &WetnessArgs::rippleGridScale, WetnessBlend::Step },
    { "rippleSpawnPeriod", &WetnessArgs::rippleSpawnPeriod, WetnessBlend::Step },
    { "rippleRadius", &WetnessArgs::rippleRadius, WetnessBlend::Step },
    { "rippleVehicleGridScale", &WetnessArgs::rippleVehicleGridScale, WetnessBlend::Step },
    { "rippleVehicleSpawnPeriod", &WetnessArgs::rippleVehicleSpawnPeriod, WetnessBlend::Step },
    { "rippleVehicleRadius", &WetnessArgs::rippleVehicleRadius, WetnessBlend::Step },
    { "dropDensity", &WetnessArgs::dropDensity },
    { "dropSize", &WetnessArgs::dropSize, WetnessBlend::Step },
    { "dropStrength", &WetnessArgs::dropStrength },
    { "dropRoughness", &WetnessArgs::dropRoughness },
    { "streakGridScale", &WetnessArgs::streakGridScale, WetnessBlend::Step },
    { "streakCycle", &WetnessArgs::streakCycle, WetnessBlend::Step },
    { "streakLength", &WetnessArgs::streakLength, WetnessBlend::Step },
    { "streakSize", &WetnessArgs::streakSize, WetnessBlend::Step },
    { "waterWaveStrength", &WetnessArgs::waterWaveStrength },
    { "waterWaveScale", &WetnessArgs::waterWaveScale, WetnessBlend::Step },
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
      const WetnessArgs defaults;
      WetnessArgs out = b;
      for (const auto& parameter : kWetnessParameters) {
        const auto member = parameter.member;
        const float av = std::isfinite(a.*member) ? std::clamp(a.*member, 0.0f, 100.0f) : defaults.*member;
        const float bv = std::isfinite(b.*member) ? std::clamp(b.*member, 0.0f, 100.0f) : defaults.*member;
        // Keep the incoming noise pattern fixed while its coverage grows from dry.
        switch (parameter.blend) {
        case WetnessBlend::Step: out.*member = bv; break;
        case WetnessBlend::Linear: out.*member = av + (bv - av) * t; break;
        }
        if (t >= 1.0f) { out.*member = bv; }
      }
      const float aWeight = float(std::min(a.weatherWetnessActive, 65535u)) / 65535.0f * (1.0f - t);
      const float bWeight = float(std::min(b.weatherWetnessActive, 65535u)) / 65535.0f * t;
      const float weight = aWeight + bWeight;
      const float aWetness = std::isfinite(a.weatherWetness) ? std::clamp(a.weatherWetness, 0.0f, 1.0f) : 1.0f;
      const float bWetness = std::isfinite(b.weatherWetness) ? std::clamp(b.weatherWetness, 0.0f, 1.0f) : 1.0f;
      out.weatherWetnessActive = uint32_t(std::round(weight * 65535.0f));
      out.weatherWetness = weight > 0.0f ? (aWetness * aWeight + bWetness * bWeight) / weight : 1.0f;
      return out;
    }
  };
}
