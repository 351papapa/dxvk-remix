// SPDX-License-Identifier: MIT
// Local runtime window lighting. See CREDITS.md.
#include <charconv>
#include <cmath>

#include "rtx_window_light.h"
#include "rtx_fork_game_state.h"

namespace dxvk {
  void WindowLightState::fillShaderParams(WindowLightArgs& args) {
    args = {};
    args.enabled = WindowLightOptions::enabled();
    if (!args.enabled) {
      return;
    }
    WindowLightGameInputs input;
    auto& store = fork_game_state::GameStateStore::get();
    std::string value;
    if (store.tryGet("__window.hour", value)) {
      float hour = 0;
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), hour);
      if (parsed.ec == std::errc() && parsed.ptr == value.data() + value.size() && std::isfinite(hour)) {
        input.hour = hour;
      }
    }
    if (store.tryGet("__window.day", value)) {
      uint32_t day = 0;
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), day);
      if (parsed.ec == std::errc() && parsed.ptr == value.data() + value.size()) {
        input.day = day;
      }
    }
    if (store.tryGet("__window.badweather", value)) {
      if (value == "1" || value == "true" || value == "True") {
        input.badWeather = true;
      } else if (value == "0" || value == "false" || value == "False") {
        input.badWeather = false;
      }
    }
    m_clock.update(args, input, WindowLightOptions::litRatio(), WindowLightOptions::startHour(),
                   WindowLightOptions::endHour(), WindowLightOptions::nightSeed());
  }
}
