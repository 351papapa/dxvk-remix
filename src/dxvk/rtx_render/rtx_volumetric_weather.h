// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include "../../util/util_vector.h"

namespace dxvk {
  struct VolumetricWeatherState {
    float transmittanceMeasurementDistanceMeters = 500.0f;
    float fogSunVisibilityGain = 1.0f;
    float volumetricConsumerGain = 0.008f;
    float noiseFieldDensityScale = 1.0f;
    float noiseFieldDensityExponent = 2.0f;
    float noiseFieldInitialFrequencyPerMeter = 8.0f;
    float noiseFieldLacunarity = 2.0f;
    float noiseFieldGain = 0.5f;
    float noiseFieldTimeScale = 0.5f;
    float noiseFieldSubStepSizeMeters = 10.0f;
    float froxelMaxDistanceMeters = 20.0f;
    float fogRemapMaxDistanceMinMeters = 1.0f;
    float fogRemapMaxDistanceMaxMeters = 40.0f;
    float fogRemapTransmittanceMeasurementDistanceMinMeters = 20.0f;
    float fogRemapTransmittanceMeasurementDistanceMaxMeters = 100.0f;
    float fogRemapColorMultiscatteringScale = 0.1f;
    float depthOffset = 0.5f;
    float noiseFieldOctaves = 2.0f;
    float anisotropy = 0.05f;
    Vector3 transmittanceColor { 0.995f, 0.995f, 0.995f };
    Vector3 singleScatteringAlbedo { 0.999f, 0.999f, 0.999f };
    bool enableHeterogeneousFog = false;
    bool enableFogRemap = false;
    bool enableFogColorRemap = false;
    bool enableFogMaxDistanceRemap = true;
    bool enableTranslucentShadows = false;
  };

  struct VolumetricWeatherParameter {
    const char* name;
    float VolumetricWeatherState::* member;
    float fallback;
    float minimum;
    float maximum;
    bool opticalDistance;
  };
  inline constexpr VolumetricWeatherParameter kVolumetricWeatherParameters[] = {
    { "transmittanceMeasurementDistanceMeters", &VolumetricWeatherState::transmittanceMeasurementDistanceMeters, 500.0f, 0.0001f, 1e+08f, true },
    { "fogSunVisibilityGain", &VolumetricWeatherState::fogSunVisibilityGain, 1.0f, 0.0f, 50.0f, false },
    { "volumetricConsumerGain", &VolumetricWeatherState::volumetricConsumerGain, 0.008f, 0.0f, 1.0f, false },
    { "noiseFieldDensityScale", &VolumetricWeatherState::noiseFieldDensityScale, 1.0f, 0.0f, 1e+08f, false },
    { "noiseFieldDensityExponent", &VolumetricWeatherState::noiseFieldDensityExponent, 2.0f, 0.0f, 1e+08f, false },
    { "noiseFieldInitialFrequencyPerMeter", &VolumetricWeatherState::noiseFieldInitialFrequencyPerMeter, 8.0f, 0.0f, 1e+08f, false },
    { "noiseFieldLacunarity", &VolumetricWeatherState::noiseFieldLacunarity, 2.0f, 0.0f, 1e+08f, false },
    { "noiseFieldGain", &VolumetricWeatherState::noiseFieldGain, 0.5f, 0.0f, 1e+08f, false },
    { "noiseFieldTimeScale", &VolumetricWeatherState::noiseFieldTimeScale, 0.5f, 0.0f, 1e+08f, false },
    { "noiseFieldSubStepSizeMeters", &VolumetricWeatherState::noiseFieldSubStepSizeMeters, 10.0f, 0.0001f, 1e+08f, false },
    { "froxelMaxDistanceMeters", &VolumetricWeatherState::froxelMaxDistanceMeters, 20.0f, 0.0001f, 1e+08f, false },
    { "fogRemapMaxDistanceMinMeters", &VolumetricWeatherState::fogRemapMaxDistanceMinMeters, 1.0f, 0.0f, 1e+08f, false },
    { "fogRemapMaxDistanceMaxMeters", &VolumetricWeatherState::fogRemapMaxDistanceMaxMeters, 40.0f, 0.0f, 1e+08f, false },
    { "fogRemapTransmittanceMeasurementDistanceMinMeters", &VolumetricWeatherState::fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f, 0.0001f, 1e+08f, false },
    { "fogRemapTransmittanceMeasurementDistanceMaxMeters", &VolumetricWeatherState::fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f, 0.0001f, 1e+08f, false },
    { "fogRemapColorMultiscatteringScale", &VolumetricWeatherState::fogRemapColorMultiscatteringScale, 0.1f, 0.0f, 1e+08f, false },
    { "depthOffset", &VolumetricWeatherState::depthOffset, 0.5f, 0.0f, 1e+08f, false },
    { "noiseFieldOctaves", &VolumetricWeatherState::noiseFieldOctaves, 2.0f, 1.0f, 8.0f, false },
    { "anisotropy", &VolumetricWeatherState::anisotropy, 0.05f, -1.0f, 1.0f, false },
  };
  struct VolumetricWeatherColor {
    const char* name;
    Vector3 VolumetricWeatherState::* member;
  };
  inline constexpr VolumetricWeatherColor kVolumetricWeatherColors[] = {
    { "transmittanceColor", &VolumetricWeatherState::transmittanceColor },
    { "singleScatteringAlbedo", &VolumetricWeatherState::singleScatteringAlbedo },
  };
  struct VolumetricWeatherSwitch {
    const char* name;
    bool VolumetricWeatherState::* member;
  };
  inline constexpr VolumetricWeatherSwitch kVolumetricWeatherSwitches[] = {
    { "enableHeterogeneousFog", &VolumetricWeatherState::enableHeterogeneousFog },
    { "enableFogRemap", &VolumetricWeatherState::enableFogRemap },
    { "enableFogColorRemap", &VolumetricWeatherState::enableFogColorRemap },
    { "enableFogMaxDistanceRemap", &VolumetricWeatherState::enableFogMaxDistanceRemap },
    { "enableTranslucentShadows", &VolumetricWeatherState::enableTranslucentShadows },
  };

  struct VolumetricWeatherPreset {
    std::array<std::optional<float>, std::size(kVolumetricWeatherParameters)> values;
    std::array<bool, std::size(kVolumetricWeatherParameters)> relative {};
    std::array<std::optional<Vector3>, std::size(kVolumetricWeatherColors)> colors;
    std::array<bool, std::size(kVolumetricWeatherColors)> colorRelative {};
    std::array<std::optional<bool>, std::size(kVolumetricWeatherSwitches)> switches;

    static void normalize(VolumetricWeatherState& state) {
      for (const auto& p : kVolumetricWeatherParameters) {
        float& value = state.*p.member;
        value = std::isfinite(value) ? std::clamp(value, p.minimum, p.maximum) : p.fallback;
      }
      const VolumetricWeatherState fallback;
      for (const auto& p : kVolumetricWeatherColors) {
        for (size_t c = 0; c < 3; ++c) {
          float& value = (state.*p.member)[c];
          value = std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : (fallback.*p.member)[c];
        }
      }
    }

    void apply(VolumetricWeatherState& state) const {
      for (size_t i = 0; i < values.size(); ++i) {
        if (values[i]) {
          float& value = state.*kVolumetricWeatherParameters[i].member;
          value = relative[i] ? value * *values[i] : *values[i];
        }
      }
      for (size_t i = 0; i < colors.size(); ++i) {
        if (colors[i]) {
          auto& value = state.*kVolumetricWeatherColors[i].member;
          for (size_t c = 0; c < 3; ++c) {
            value[c] = colorRelative[i] ? value[c] * (*colors[i])[c] : (*colors[i])[c];
          }
        }
      }
      for (size_t i = 0; i < switches.size(); ++i) {
        if (switches[i]) {
          state.*kVolumetricWeatherSwitches[i].member = *switches[i];
        }
      }
      normalize(state);
    }

    void store(VolumetricWeatherState state) {
      normalize(state);
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = state.*kVolumetricWeatherParameters[i].member;
        relative[i] = false;
      }
      for (size_t i = 0; i < colors.size(); ++i) {
        colors[i] = state.*kVolumetricWeatherColors[i].member;
        colorRelative[i] = false;
      }
      for (size_t i = 0; i < switches.size(); ++i) {
        switches[i] = state.*kVolumetricWeatherSwitches[i].member;
      }
    }

    static VolumetricWeatherState blend(VolumetricWeatherState a, VolumetricWeatherState b, float t) {
      normalize(a);
      normalize(b);
      t = std::isfinite(t) ? std::clamp(t, 0.0f, 1.0f) : 0.0f;
      auto result = b;
      for (const auto& p : kVolumetricWeatherParameters) {
        const float av = a.*p.member;
        const float bv = b.*p.member;
        result.*p.member = p.opticalDistance
          ? 1.0f / std::max((1.0f / av) + ((1.0f / bv) - (1.0f / av)) * t, 1e-4f)
          : av + (bv - av) * t;
      }
      for (const auto& p : kVolumetricWeatherColors) {
        result.*p.member = a.*p.member + (b.*p.member - a.*p.member) * t;
      }
      // Structural switches take the incoming value as soon as a transition starts.
      return result;
    }
  };
}
