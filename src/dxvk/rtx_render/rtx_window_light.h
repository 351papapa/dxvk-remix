// SPDX-License-Identifier: MIT
// Local runtime window lighting. See CREDITS.md.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtx/pass/window_light_args.h"

namespace dxvk {
class WindowLightRules {
public:
  bool configure(const std::string& textureNames) {
    if (m_text == textureNames) {
      return false;
    }
    m_text = textureNames;
    m_textureRules.clear();
    split(textureNames, m_textureRules);
    return true;
  }

  bool matchesTextureName(const std::string* name) const {
    return name != nullptr && matches(m_textureRules, std::string_view(*name));
  }

  bool matchesTextureName(const char* name) const {
    return name != nullptr && matches(m_textureRules, std::string_view(name));
  }

  size_t textureRuleCount() const { return m_textureRules.size(); }

private:
  static void split(const std::string& text, std::vector<std::string>& out) {
    size_t begin = 0;
    while (begin < text.size()) {
      const size_t end = text.find_first_of(";\r\n", begin);
      std::string token = lower(text.substr(begin, end == std::string::npos ? end : end - begin));
      const size_t first = token.find_first_not_of(" \t");
      if (first != std::string::npos) {
        out.push_back(token.substr(first, token.find_last_not_of(" \t") - first + 1));
      }
      if (end == std::string::npos) { break; }
      begin = end + 1;
    }
  }

  static bool matches(const std::vector<std::string>& rules, std::string_view name) {
    if (rules.empty() || name.empty()) { return false; }
    for (const auto& rule : rules) {
      if (rule.empty() || rule.size() > name.size()) { continue; }
      for (size_t i = 0; i + rule.size() <= name.size(); ++i) {
        size_t j = 0;
        for (; j < rule.size(); ++j) {
          char c = name[i + j];
          if (c >= 'A' && c <= 'Z') { c = char(c + ('a' - 'A')); }
          if (c != rule[j]) { break; }
        }
        if (j == rule.size()) { return true; }
      }
    }
    return false;
  }

  static std::string lower(std::string text) {
    for (char& c : text) {
      if (c >= 'A' && c <= 'Z') { c = char(c + ('a' - 'A')); }
    }
    return text;
  }

  std::string m_text;
  std::vector<std::string> m_textureRules;
};

  struct WindowLightGameInputs {
    std::optional<float> hour;
    std::optional<uint32_t> day;
    std::optional<bool> badWeather;
  };

  class WindowLightClock {
  public:
    void update(WindowLightArgs& args, const WindowLightGameInputs& input,
                float ratio, float startHour, float endHour, uint32_t manualSeed) {
      if (input.hour.has_value() && std::isfinite(*input.hour)) {
        m_hour = wrapHour(*input.hour);
      }
      if (input.day.has_value()) {
        m_day = input.day;
      }
      if (input.badWeather.has_value()) {
        m_badWeather = *input.badWeather;
      }
      const float start = wrapHour(startHour);
      const float end = wrapHour(endHour);
      const float length = wrapHour(end - start);
      args.litRatio = std::isfinite(ratio) ? std::clamp(ratio, 0.0f, 1.0f) : 0.0f;
      bool night = true;
      uint32_t nightDay = m_day.value_or(0);
      if (m_hour.has_value() && length > 0.0f) {
        const float elapsed = wrapHour(*m_hour - start);
        night = elapsed < length;
        args.litRatio *= night ? 1.0f - elapsed / length : 0.0f;
        // The night beginning before midnight belongs to the preceding calendar day.
        if (end < start && *m_hour < start) {
          --nightDay;
        }
      }
      args.streetLampOn = night || m_badWeather;
      args.nightSeed = manualSeed ^ (m_day.has_value() ? daySeed(nightDay) : 0u);
    }

  private:
    static float wrapHour(float hour) {
      if (!std::isfinite(hour)) {
        return 0.0f;
      }
      const float wrapped = std::fmod(hour, 24.0f);
      return wrapped < 0.0f ? wrapped + 24.0f : wrapped;
    }
    static uint32_t daySeed(uint32_t day) {
      uint32_t hash = day * 0x9E3779B1u;
      hash = (hash ^ (hash >> 16)) * 0x7FEB352Du;
      hash = (hash ^ (hash >> 15)) * 0x846CA68Bu;
      return hash ^ (hash >> 16);
    }
    std::optional<float> m_hour;
    std::optional<uint32_t> m_day;
    bool m_badWeather = false;
  };

  class WindowLightState {
  public:
    void beginFrame() {
      m_windows.configure(WindowLightOptions::textureNames());
      m_lamps.configure(WindowLightOptions::streetLampTextureNames());
      m_matches.clear();
    }
    uint8_t classify(uint64_t originalHash, uint32_t materialIndex,
                     const std::string* originalName, const char* replacementName) {
      const Key key { originalHash, materialIndex };
      const auto cached = m_matches.find(key);
      if (cached != m_matches.end()) {
        return cached->second;
      }
      uint8_t classification = 0;
      if (WindowLightOptions::streetLampsEnabled()
          && (m_lamps.matchesTextureName(originalName) || m_lamps.matchesTextureName(replacementName))) {
        classification = 2;
      } else if (m_windows.matchesTextureName(originalName) || m_windows.matchesTextureName(replacementName)) {
        classification = 1;
      }
      m_matches.emplace(key, classification);
      return classification;
    }
    void fillShaderParams(WindowLightArgs& args);

  private:
    struct Key {
      uint64_t originalHash;
      uint32_t materialIndex;
      bool operator==(const Key& other) const {
        return originalHash == other.originalHash && materialIndex == other.materialIndex;
      }
    };
    struct KeyHash {
      size_t operator()(const Key& key) const {
        return size_t(key.originalHash ^ (uint64_t(key.materialIndex) * 0x9E3779B185EBCA87ull));
      }
    };
    WindowLightRules m_windows, m_lamps;
    WindowLightClock m_clock;
    std::unordered_map<Key, uint8_t, KeyHash> m_matches;
  };
}
