// SPDX-License-Identifier: MIT
#include <cstdio>
#include <cmath>
#include <limits>
#include "../../../src/dxvk/rtx_render/rtx_water_weather.h"
#include "../../../public/include/remixapi/water_controls.h"
#include "../../../public/include/remixapi/water_gpu_surface.h"
#include "../../../public/include/remixapi/water_gpu_simulation.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/sharc/sharc_binding_indices.h"
#define ACCELERATION_STRUCTURE(slot) slot,
#define STRUCTURED_BUFFER(slot) slot,
#define RW_STRUCTURED_BUFFER(slot) slot,
#define TEXTURE2D(slot) slot,
#define RW_TEXTURE2D(slot) slot,
#define TEXTURE2DARRAY(slot) slot,
#define TEXTURE3D(slot) slot,
#define CONSTANT_BUFFER(slot) slot,
#define SAMPLER3D(slot) slot,
#define SAMPLER(slot) slot,
constexpr unsigned sharedSharcBindings[] = {
  COMMON_RAYTRACING_BINDINGS
  SHARC_BINDING_HASH, SHARC_BINDING_ACCUMULATION, SHARC_BINDING_RESOLVED,
  SHARC_BINDING_DEFERRED_VERTICES
};
#undef ACCELERATION_STRUCTURE
#undef STRUCTURED_BUFFER
#undef RW_STRUCTURED_BUFFER
#undef TEXTURE2D
#undef RW_TEXTURE2D
#undef TEXTURE2DARRAY
#undef TEXTURE3D
#undef CONSTANT_BUFFER
#undef SAMPLER3D
#undef SAMPLER
constexpr bool uniqueSharedBindings() {
  for (unsigned i=0; i<sizeof(sharedSharcBindings)/sizeof(unsigned); ++i) {
    for (unsigned j=0; j<i; ++j) {
      if (sharedSharcBindings[i]==sharedSharcBindings[j]) { return false; }
    }
  }
  return true;
}
static_assert(uniqueSharedBindings(), "Common ray tracing resources must not alias SHARC buffers");
int main() {
  using dxvk::WaterWeatherState;
  int failures = 0;
  auto check = [&](bool ok, const char* message) { if (!ok) { ++failures; std::printf("FAIL: %s\n", message); } };
  auto near = [](float a, float b) { return std::abs(a-b) < 0.00001f; };
  const auto rain = WaterWeatherState::forName("Drizzle"), storm = WaterWeatherState::forName("THUNDERSTORM");
  check(near(rain.multiplier(2,4),2) && near(storm.multiplier(2,4),4), "rain and storm endpoints");
  auto half = WaterWeatherState::blend(rain,storm,.5f);
  check(near(half.multiplier(2,4),3), "rain-storm blend does not double count clear weight");
  auto interrupted = WaterWeatherState::blend(half,{},.5f);
  check(near(interrupted.multiplier(2,4),2), "interrupted transition starts at current wave gain");
  check(near(WaterWeatherState::blend(rain,storm,-1).rain,1) && near(WaterWeatherState::blend(rain,storm,2).storm,1), "clamped transition");
  WaterWeatherState invalid{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()};
  invalid.normalize(); check(near(invalid.multiplier(2,4),1), "invalid persisted weights produce finite calm state");
  WaterWeatherState excess{1,1}; excess.normalize();check(near(excess.rain+excess.storm,1),"normalized mixed weights");
  check(near(WaterWeatherState::blend(rain,storm,std::numeric_limits<float>::quiet_NaN()).rain,1),"invalid blend keeps outgoing state");
  static_assert(sizeof(remixapi::water::Controls)==124);
  static_assert(offsetof(remixapi::water::gpu::Frame,data)==28);
  static_assert(sizeof(remixapi::water::simulation::Vertex)==36);
  std::printf("Water weather and pointer-free wire contracts: %d failures\n",failures);
  return failures ? 1 : 0;
}
