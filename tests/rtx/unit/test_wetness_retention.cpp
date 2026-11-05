// SPDX-License-Identifier: MIT
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "../../../src/dxvk/rtx_render/rtx_wetness_retention.h"

namespace {
  void require(bool value, const char* message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  void requireClose(float actual, float expected, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 0.00001f) {
      std::cerr << message << ": expected " << expected << ", got " << actual << '\n';
      throw std::runtime_error(message);
    }
  }

  float advanceDry(dxvk::WetnessRetention& state, int frames, float step = 0.25f,
                   float hold = 90.0f, float dry = 120.0f) {
    float value = 0.0f;
    for (int i = 0; i < frames; ++i) {
      value = state.update(0.0f, step, hold, dry, true);
    }
    return value;
  }

  void testRainPeakAndDryStart() {
    dxvk::WetnessRetention state;
    requireClose(advanceDry(state, 1200), 0.0f, "Dry weather created water without preceding rain");
    requireClose(state.update(0.2f, 0.25f, 90.0f, 120.0f, true), 0.2f, "Rain did not wet the surface");
    requireClose(state.update(0.8f, 0.25f, 90.0f, 120.0f, true), 0.8f, "Stronger rain did not increase wetness");
    requireClose(state.update(0.3f, 0.25f, 90.0f, 120.0f, true), 0.8f, "Falling rain intensity discarded retained water");
    requireClose(advanceDry(state, 360), 0.8f, "The hold did not start from the rain peak");
  }

  void testHoldAndSmoothDrying() {
    dxvk::WetnessRetention state;
    state.update(1.0f, 0.0f, 90.0f, 120.0f, true);
    requireClose(advanceDry(state, 360), 1.0f, "Water faded before the 90-second hold ended");
    requireClose(advanceDry(state, 120), 0.84375f, "First quarter of drying was not smoothly eased");
    requireClose(advanceDry(state, 120), 0.5f, "Water did not reach half strength halfway through drying");
    requireClose(advanceDry(state, 120), 0.15625f, "Last quarter of drying was not smoothly eased");
    requireClose(advanceDry(state, 120), 0.0f, "Water remained after 120 seconds of drying");
    requireClose(advanceDry(state, 1000), 0.0f, "Dry weather restored an old rain peak");
  }

  void testRainResumesWithoutJump() {
    dxvk::WetnessRetention state;
    state.update(1.0f, 0.0f, 90.0f, 120.0f, true);
    const float remaining = advanceDry(state, 600);
    requireClose(remaining, 0.5f, "Test did not reach partial drying");
    requireClose(state.update(0.1f, 0.25f, 90.0f, 120.0f, true), remaining,
      "Light rain restored the previous peak or removed retained water");
    requireClose(state.update(0.2f, 0.25f, 90.0f, 120.0f, true), remaining,
      "Increasing light rain changed the retained surface discontinuously");
    requireClose(advanceDry(state, 360), remaining, "New rain did not restart the hold at the current level");
    requireClose(advanceDry(state, 240), 0.25f, "The second drying cycle reused the old peak");
    requireClose(state.update(0.75f, 0.25f, 90.0f, 120.0f, true), 0.75f,
      "Stronger returning rain could not raise the surface level");
  }

  void testDisableAndReset() {
    dxvk::WetnessRetention state;
    state.update(1.0f, 0.25f, 90.0f, 120.0f, true);
    requireClose(state.update(0.2f, 0.25f, 90.0f, 120.0f, false), 0.2f,
      "Inactive retention changed the current authored rain");
    requireClose(advanceDry(state, 1), 0.0f, "Re-enabling retention resurrected old water");
    state.update(1.0f, 0.25f, 90.0f, 120.0f, true);
    requireClose(state.update(0.0f, 0.25f, 90.0f, 120.0f, false), 0.0f,
      "Removing authored weather retained stale water");
    requireClose(state.update(0.3f, 0.25f, 90.0f, 120.0f, true), 0.3f,
      "A new authored weather inherited an old peak");
    state.reset();
    requireClose(advanceDry(state, 1), 0.0f, "Explicit reset left retained water");
  }

  void testTimeAndInputValidation() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    dxvk::WetnessRetention state;
    state.update(1.0f, 0.0f, 0.0f, 1.0f, true);
    for (float step : { 0.0f, -1.0f, nan, infinity }) {
      requireClose(state.update(0.0f, step, 0.0f, 1.0f, true), 1.0f,
        "Paused, negative or invalid time advanced drying");
    }
    requireClose(state.update(0.0f, 100.0f, 0.0f, 1.0f, true), 0.84375f,
      "A stalled frame advanced drying by more than 0.25 seconds");
    requireClose(state.update(2.0f, 0.0f, 0.0f, 1.0f, true), 1.0f, "Rain above one escaped the input clamp");
    requireClose(state.update(-1.0f, 0.0f, 0.0f, 0.0f, false), 0.0f, "Negative rain escaped the input clamp");
    for (float rain : { nan, infinity, -infinity }) {
      requireClose(state.update(rain, 0.25f, 0.0f, 0.0f, true), 0.0f,
        "Nonfinite rain created water or escaped into the surface state");
    }
    for (float duration : { 0.0f, -1.0f, nan, infinity }) {
      state.update(1.0f, 0.0f, duration, duration, true);
      requireClose(state.update(0.0f, 0.25f, duration, duration, true), 0.0f,
        "Zero, negative or invalid durations prevented complete drying");
    }
  }

  void testFrameSubdivisionAndBounds() {
    dxvk::WetnessRetention coarse;
    dxvk::WetnessRetention fine;
    coarse.update(0.7f, 0.0f, 2.0f, 4.0f, true);
    fine.update(0.7f, 0.0f, 2.0f, 4.0f, true);
    float previous = 0.7f;
    for (int i = 0; i < 32; ++i) {
      const float current = coarse.update(0.0f, 0.25f, 2.0f, 4.0f, true);
      fine.update(0.0f, 0.125f, 2.0f, 4.0f, true);
      const float subdivided = fine.update(0.0f, 0.125f, 2.0f, 4.0f, true);
      requireClose(current, subdivided, "Drying depended on frame subdivision");
      require(current >= 0.0f && current <= previous, "Drying increased water or left its valid range");
      previous = current;
    }
    requireClose(previous, 0.0f, "Monotonic drying did not finish");
  }

  void testLiveDurationChanges() {
    dxvk::WetnessRetention state;
    state.update(1.0f, 0.0f, 0.0f, 2.0f, true);
    requireClose(advanceDry(state, 4, 0.25f, 0.0f, 2.0f), 0.5f, "Test did not reach partial drying");
    requireClose(state.update(0.0f, 0.25f, 0.0f, 20.0f, true), 0.5f,
      "Extending the dry duration restored evaporated water");
    requireClose(state.update(0.0f, 0.25f, 90.0f, 20.0f, true), 0.5f,
      "Extending the hold restored evaporated water");
    requireClose(state.update(0.0f, 0.25f, 0.0f, 0.0f, true), 0.0f,
      "Removing both durations did not finish drying");
  }
}

int main() {
  try {
    testRainPeakAndDryStart();
    testHoldAndSmoothDrying();
    testRainResumesWithoutJump();
    testDisableAndReset();
    testTimeAndInputValidation();
    testFrameSubdivisionAndBounds();
    testLiveDurationChanges();
    std::cout << "Wetness retention, hold, drying, rain restart and input validation passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return -1;
  }
}
