// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include "../../util/util_vector.h"

namespace dxvk {
  struct PrecipitationState {
    float precipitationIntensity = 0.0f;
    float precipitationFallSpeed = 8.0f;
    float precipitationWindResponse = 0.04f;
    float precipitationTurbulence = 0.0f;
    float precipitationDrag = 0.0f;
    float precipitationStreak = 1.0f;
    float precipitationDropWidth = 0.35f;
    float precipitationDropLength = 4.0f;
    float precipitationOpacity = 0.6f;
    float precipitationSkyLight = 1.0f;
    Vector3 precipitationColor { 0.75f, 0.80f, 0.90f };
  };

  struct PrecipitationParameter {
    const char* name;
    float PrecipitationState::* member;
    float fallback;
    float minimum;
    float maximum;
  };

  inline constexpr PrecipitationParameter kPrecipitationParameters[] = {
    { "intensity", &PrecipitationState::precipitationIntensity, 0.0f, 0.0f, 1.0f },
    { "fallSpeed", &PrecipitationState::precipitationFallSpeed, 8.0f, 0.05f, 40.0f },
    { "windResponse", &PrecipitationState::precipitationWindResponse, 0.04f, 0.0f, 4.0f },
    { "turbulence", &PrecipitationState::precipitationTurbulence, 0.0f, 0.0f, 8.0f },
    { "drag", &PrecipitationState::precipitationDrag, 0.0f, 0.0f, 10.0f },
    { "streak", &PrecipitationState::precipitationStreak, 1.0f, 0.0f, 8.0f },
    { "dropWidth", &PrecipitationState::precipitationDropWidth, 0.35f, 0.01f, 20.0f },
    { "dropLength", &PrecipitationState::precipitationDropLength, 4.0f, 0.01f, 60.0f },
    { "opacity", &PrecipitationState::precipitationOpacity, 0.6f, 0.0f, 1.0f },
    { "skyLight", &PrecipitationState::precipitationSkyLight, 1.0f, 0.0f, 10.0f },
  };

  struct PrecipitationPreset {
    std::array<std::optional<float>, std::size(kPrecipitationParameters)> values;
    std::array<bool, std::size(kPrecipitationParameters)> relative {};
    std::optional<Vector3> color;
    bool colorRelative = false;

    static void normalize(PrecipitationState& state) {
      for (const auto& parameter : kPrecipitationParameters) {
        float& value = state.*parameter.member;
        value = std::isfinite(value) ? std::clamp(value, parameter.minimum, parameter.maximum) : parameter.fallback;
      }
      const Vector3 fallback { 0.75f, 0.80f, 0.90f };
      for (size_t i = 0; i < 3; ++i) {
        float& value = state.precipitationColor[i];
        value = std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : fallback[i];
      }
    }

    void apply(PrecipitationState& state) const {
      for (size_t i = 0; i < values.size(); ++i) {
        if (values[i]) {
          float& value = state.*kPrecipitationParameters[i].member;
          value = relative[i] ? value * *values[i] : *values[i];
        }
      }
      if (color) {
        for (size_t i = 0; i < 3; ++i) {
          state.precipitationColor[i] = colorRelative ? state.precipitationColor[i] * (*color)[i] : (*color)[i];
        }
      }
      normalize(state);
    }

    void store(PrecipitationState state) {
      normalize(state);
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = state.*kPrecipitationParameters[i].member;
        relative[i] = false;
      }
      color = state.precipitationColor;
      colorRelative = false;
    }

    static PrecipitationState blend(PrecipitationState a, PrecipitationState b, float t) {
      normalize(a);
      normalize(b);
      t = std::isfinite(t) ? std::clamp(t, 0.0f, 1.0f) : 0.0f;
      PrecipitationState result;
      for (const auto& parameter : kPrecipitationParameters) {
        const auto member = parameter.member;
        result.*member = a.*member + (b.*member - a.*member) * t;
      }
      result.precipitationColor = a.precipitationColor + (b.precipitationColor - a.precipitationColor) * t;
      return result;
    }
  };
}
