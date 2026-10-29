// SPDX-License-Identifier: MIT
#include <cmath>
#include <cstdio>
#include <limits>
#include "../../../src/dxvk/rtx_render/rtx_precipitation_state.h"
#include "rtx/pass/particles/particle_system_binding_indices.h"
static_assert(sizeof(GpuParticle) == 48, "GPU particle layout must stay compatible");
static_assert(offsetof(GpuParticle, timeToLive) == 44, "Keep full float particle lifetime");
int main() {
  using namespace dxvk;
  int failures = 0;
  auto check = [&](bool ok, const char* label) { if (!ok) { ++failures; std::printf("FAIL %s\n", label); } };
  auto closeEnough = [](float a, float b) { return std::abs(a-b) < 1e-5f; };
  PrecipitationState clear, rain;
  rain.precipitationIntensity = 1;
  rain.precipitationFallSpeed = 12;
  rain.precipitationColor = Vector3(1,0,0);
  auto half = PrecipitationPreset::blend(clear,rain,.5f);
  check(closeEnough(half.precipitationIntensity,.5f) && closeEnough(half.precipitationFallSpeed,10), "mixed rain motion and intensity");
  auto interrupted = PrecipitationPreset::blend(half,clear,.5f);
  check(closeEnough(interrupted.precipitationIntensity,.25f), "interrupted fade uses current look");
  check(closeEnough(PrecipitationPreset::blend(clear,rain,2).precipitationIntensity,1), "overrange endpoint");
  check(closeEnough(PrecipitationPreset::blend(clear,rain,std::numeric_limits<float>::quiet_NaN()).precipitationIntensity,0), "invalid blend factor");
  PrecipitationPreset preset;
  preset.values[0] = .25f;
  preset.values[1] = 2.0f;
  preset.relative[1] = true;
  preset.color = Vector3(.5f,1,.5f);
  preset.colorRelative = true;
  preset.apply(rain);
  check(closeEnough(rain.precipitationIntensity,.25f) && closeEnough(rain.precipitationFallSpeed,24), "absolute and relative overrides");
  check(closeEnough(rain.precipitationColor.x,.5f) && closeEnough(rain.precipitationColor.y,0), "relative color");
  preset.store(rain);
  auto restored = clear;
  preset.apply(restored);
  check(closeEnough(restored.precipitationFallSpeed,24) && !preset.relative[1], "stored look is absolute");
  restored.precipitationFallSpeed = std::numeric_limits<float>::infinity();
  restored.precipitationIntensity = -2;
  restored.precipitationColor = Vector3(2,-1,std::numeric_limits<float>::quiet_NaN());
  PrecipitationPreset::normalize(restored);
  check(closeEnough(restored.precipitationFallSpeed,8) && closeEnough(restored.precipitationIntensity,0), "nonfinite and negative inputs");
  check(closeEnough(restored.precipitationColor.x,1) && closeEnough(restored.precipitationColor.y,0) && closeEnough(restored.precipitationColor.z,.9f), "color bounds");
  RtxParticleSystemDesc desc;
  check(!desc.spawnOnPuddlesOnly && !desc.spawnOnWaterOrPuddles && !desc.spawnOnWetRoads, "contact defaults disabled");
  const auto original = desc.calcHash();
  desc.spawnOnWetRoads = true;
  check(original != desc.calcHash(), "contact mode participates in system identity");
  std::printf("precipitation failures=%d\n", failures);
  return failures ? 1 : 0;
}
