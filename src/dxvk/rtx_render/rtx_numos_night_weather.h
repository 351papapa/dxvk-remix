#pragma once

#include <algorithm>
#include <array>
#include <optional>

#include "../../util/util_vector.h"

namespace dxvk {

  struct NumosNightAppearanceState {
    bool nightAppearance = false;
    bool legacyMilkyWay = false;
    float starBrightness = 1.0f;
    float starMagnitudeLimit = 7.5f;
    float milkyWayBrightness = 1.0f;
    float legacyMilkyWayBrightness = 0.05f;
    float legacyMilkyWayDustAmount = 0.6f;
    float nightSkyBrightness = 0.0f;
    Vector3 legacyMilkyWayOuterColor { 0.5f, 0.55f, 0.75f };
    Vector3 legacyMilkyWayCoreColor { 1.0f, 0.85f, 0.55f };
    Vector3 nightSkyColor { 0.06f, 0.12f, 0.32f };

    float catalogueMilkyWayGain = 1.0f;
    float legacyMilkyWayGain = 0.0f;

    void resolveMilkyWayGains() {
      catalogueMilkyWayGain = legacyMilkyWay ? 0.0f : milkyWayBrightness;
      legacyMilkyWayGain = legacyMilkyWay ? legacyMilkyWayBrightness : 0.0f;
    }
  };

  struct NumosNightWeatherFloat {
    const char* name;
    const char* label;
    float NumosNightAppearanceState::* member;
    float minimum;
    float maximum;
  };

  inline constexpr NumosNightWeatherFloat kNumosNightWeatherFloats[] = {
    { "nightSkyBrightness", "Night sky brightness", &NumosNightAppearanceState::nightSkyBrightness, 0.0f, 0.05f },
    { "starBrightness", "Star brightness", &NumosNightAppearanceState::starBrightness, 0.0f, 64.0f },
    { "starMagnitudeLimit", "Star magnitude limit", &NumosNightAppearanceState::starMagnitudeLimit, -2.0f, 7.5f },
    { "milkyWayBrightness", "Catalogue Milky Way brightness", &NumosNightAppearanceState::milkyWayBrightness, 0.0f, 8192.0f },
    { "legacyMilkyWayBrightness", "Legacy Milky Way brightness", &NumosNightAppearanceState::legacyMilkyWayBrightness, 0.0f, 1.0f },
    { "legacyMilkyWayDustAmount", "Legacy Milky Way dust", &NumosNightAppearanceState::legacyMilkyWayDustAmount, 0.0f, 1.0f },
  };

  struct NumosNightWeatherColor {
    const char* name;
    const char* label;
    Vector3 NumosNightAppearanceState::* member;
  };

  inline constexpr NumosNightWeatherColor kNumosNightWeatherColors[] = {
    { "nightSkyColor", "Night sky color", &NumosNightAppearanceState::nightSkyColor },
    { "legacyMilkyWayOuterColor", "Legacy Milky Way outer color", &NumosNightAppearanceState::legacyMilkyWayOuterColor },
    { "legacyMilkyWayCoreColor", "Legacy Milky Way core color", &NumosNightAppearanceState::legacyMilkyWayCoreColor },
  };

  struct NumosNightAppearancePreset {
    std::array<std::optional<float>, std::size(kNumosNightWeatherFloats)> floats;
    std::array<std::optional<Vector3>, std::size(kNumosNightWeatherColors)> colors;
    std::optional<bool> legacyMilkyWay;

    void apply(NumosNightAppearanceState& state) const {
      if (legacyMilkyWay.has_value()) {
        state.legacyMilkyWay = *legacyMilkyWay;
      }
      for (size_t i = 0; i < floats.size(); ++i) {
        if (floats[i]) {
          const auto& p = kNumosNightWeatherFloats[i];
          state.*p.member = std::clamp(*floats[i], p.minimum, p.maximum);
        }
      }
      for (size_t i = 0; i < colors.size(); ++i) {
        if (colors[i]) {
          const auto& p = kNumosNightWeatherColors[i];
          state.*p.member = dxvk::clamp(*colors[i], Vector3(0.0f), Vector3(1.0f));
        }
      }
      state.resolveMilkyWayGains();
    }

    void store(const NumosNightAppearanceState& state, const NumosNightAppearanceState* before = nullptr) {
      if (legacyMilkyWay.has_value() || before == nullptr || state.legacyMilkyWay != before->legacyMilkyWay) {
        legacyMilkyWay = state.legacyMilkyWay;
      }
      for (size_t i = 0; i < floats.size(); ++i) {
        const auto member = kNumosNightWeatherFloats[i].member;
        if (floats[i] || before == nullptr || state.*member != before->*member) {
          floats[i] = state.*member;
        }
      }
      for (size_t i = 0; i < colors.size(); ++i) {
        const auto member = kNumosNightWeatherColors[i].member;
        if (colors[i] || before == nullptr || state.*member != before->*member) {
          colors[i] = state.*member;
        }
      }
    }

    static NumosNightAppearanceState blend(const NumosNightAppearanceState& a, const NumosNightAppearanceState& b, float t) {
      if (t <= 0.0f) {
        return a;
      }
      if (t >= 1.0f) {
        return b;
      }
      auto result = b;
      result.legacyMilkyWay = t < 0.5f ? a.legacyMilkyWay : b.legacyMilkyWay;
      // Blend the contributing gains directly so switching modes never snaps either band on or off.
      result.catalogueMilkyWayGain = a.catalogueMilkyWayGain + (b.catalogueMilkyWayGain - a.catalogueMilkyWayGain) * t;
      result.legacyMilkyWayGain = a.legacyMilkyWayGain + (b.legacyMilkyWayGain - a.legacyMilkyWayGain) * t;
      for (const auto& p : kNumosNightWeatherFloats) {
        result.*p.member = a.*p.member + (b.*p.member - a.*p.member) * t;
      }
      for (const auto& p : kNumosNightWeatherColors) {
        result.*p.member = a.*p.member + (b.*p.member - a.*p.member) * t;
      }
      return result;
    }
  };

}
