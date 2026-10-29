/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

struct WetnessArgs {

  float puddleNoiseScale = 0.045f;   // world-space noise frequency (larger = smaller patches)
  float puddleThresholdBase = 0.56f; // noise threshold at zero wetness
  float puddleThresholdSpan = 0.4f;  // how much wetness lowers the threshold
  float puddleAlbedoDarken = 0.85f;  // albedo multiplier inside puddles
  float puddleWaveStrength = 0.4f;   // micro-wave normal wobble amplitude inside puddles (0 = flat mirror)
  float puddleWaveScale = 1.0f;      // puddle micro-wave size multiplier (larger = bigger, sparser waves)

  float rippleGridScale = 8.5f;      // cells per world unit
  float rippleSpawnPeriod = 1.5f;    // seconds between drops per cell
  float rippleRadius = 1.0f;         // ring expansion speed multiplier

  float rippleVehicleGridScale = 4.25f;
  float rippleVehicleSpawnPeriod = 3.0f;
  float rippleVehicleRadius = 0.5f;

  float dropDensity = 1.3f;
  float dropSize = 3.0f;
  float dropStrength = 5.0f;
  float dropRoughness = 2.7f;

  float streakGridScale = 12.0f;     // cells per object unit
  float streakCycle = 1.8f;          // seconds per slide run
  float streakLength = 0.4f;         // object units travelled per run
  float streakSize = 1.0f;           // head radius multiplier

  float waterWaveStrength = 0.5f;    // normal-wobble amplitude on water surfaces
  float waterWaveScale = 1.0f;       // wave size multiplier (larger = bigger, sparser chop)

  uint noRainNodeCount = 0;
  uint roadOnly = 0;
  float rainEndRoughnessWeight = 0.0f;
  float rainEndRoughness = 0.75f;
  uint respectNoRainZones = 0;
  uint nonRoadRoughnessOnly = 0;

  uint enabled = 0;

  float contactWetness = 1.0f;

  float weatherWetness = 1.0f;
  uint weatherWetnessActive = 0;

  float shoreOriginX = 0.0f;
  float shoreOriginY = 0.0f;
  float shoreInvSpacing = 0.0f;
  float shoreStrength = 0.0f;
};

#ifndef __cplusplus
float resolveWeatherWetness(uint tag, WetnessArgs wetness) {

  return lerp(float(tag & 0xFFu) * (1.0f / 255.0f), saturate(wetness.weatherWetness),
              saturate(float(wetness.weatherWetnessActive) * (1.0f / 65535.0f)));
}
#endif

#ifdef __cplusplus
static_assert((sizeof(WetnessArgs) & 15) == 0);
static_assert(sizeof(WetnessArgs) == 144); // Shared C++/Slang frame-constant layout.

#include "rtx_option.h"

namespace dxvk {

struct WetnessOptions {
  friend class ImGUI;

  RTX_OPTION("rtx.wetness", bool, enabled, false, "Master switch for the wetness/rain surface system. Off: no wet sheen, puddles, road ripples, vehicle droplets or streaks, and the rain-ending roughness override stops being applied. Weather and rain particles are unaffected.");
  RTX_OPTION("rtx.wetness", bool, respectNoRainZones, false, "Disable wetness inside no-rain cullzones. DFF enable rules override this restriction; rain is still required.");
  RTX_OPTION("rtx.wetness", bool, nonRoadRoughnessOnly, false, "In road-only mode, allow rain-driven roughness on non-road world surfaces without puddles or ripples. Respects cullzones and DFF disable rules; DFF enable rules retain full wetness.");
  RTX_OPTION("rtx.wetness", bool, roadOnly, false, "Restrict world wetness to road-flagged models. Vehicles are unaffected.");
  RTX_OPTION("rtx.wetness", std::string, forceEnableDff, "", "DFF name substrings separated by semicolons. Overrides road and no-rain zone filtering; still requires rain.");
  RTX_OPTION("rtx.wetness", std::string, forceDisableDff, "", "DFF name substrings separated by semicolons. Disables wetness; takes precedence over enable matches.");

  RTX_OPTION("rtx.wetness", bool, rainEndRoughnessEnabled, false, "Blend GLOBAL opaque roughness toward rainEndRoughness while the weather goes from rainy to dry. Off by default: it overrides the authored roughness of every opaque surface and fights the wetness layer's own decay, so materials do not simply return to their normal roughness. Enable only if you want a uniform damp look while the world dries.");
  RTX_OPTION("rtx.wetness", float, rainEndRoughness, 0.75f, "Absolute opaque roughness target while rain is ending.");
  RTX_OPTION("rtx.wetness", float, rainEndTransitionSeconds, 15.0f, "Seconds to blend into and out of the rain-ending roughness override.");

  RTX_OPTION("rtx.wetness", float, puddleNoiseScale, 0.045f, "Puddle noise frequency (world). Larger = smaller puddles.");
  RTX_OPTION("rtx.wetness", float, puddleThresholdBase, 0.56f, "Puddle noise threshold when dry. Lower = more puddles.");
  RTX_OPTION("rtx.wetness", float, puddleThresholdSpan, 0.4f, "How much full wetness lowers the puddle threshold.");
  RTX_OPTION("rtx.wetness", float, puddleAlbedoDarken, 0.85f, "Albedo multiplier inside puddles (water absorbs light).");
  RTX_OPTION("rtx.wetness", float, puddleWaveStrength, 0.4f, "Puddle micro-wave normal wobble amplitude. 0 = dead-flat mirror.");
  RTX_OPTION("rtx.wetness", float, puddleWaveScale, 1.0f, "Puddle micro-wave size multiplier. Larger = bigger, sparser waves.");

  RTX_OPTION("rtx.wetness", float, rippleGridScale, 8.5f, "Road ripple cells per world unit. Higher = more, smaller rings.");
  RTX_OPTION("rtx.wetness", float, rippleSpawnPeriod, 1.5f, "Seconds between raindrops per road cell. Higher = sparser.");
  RTX_OPTION("rtx.wetness", float, rippleRadius, 1.0f, "Road ripple ring expansion speed multiplier.");

  RTX_OPTION("rtx.wetness", float, rippleVehicleGridScale, 4.25f, "Vehicle ripple cells per object unit.");
  RTX_OPTION("rtx.wetness", float, rippleVehicleSpawnPeriod, 3.0f, "Seconds between raindrops per vehicle cell.");
  RTX_OPTION("rtx.wetness", float, rippleVehicleRadius, 0.5f, "Vehicle ripple ring expansion speed multiplier.");

  RTX_OPTION("rtx.wetness", float, dropDensity, 1.3f, "Vehicle droplet density multiplier.");
  RTX_OPTION("rtx.wetness", float, dropSize, 3.0f, "Vehicle droplet size multiplier.");
  RTX_OPTION("rtx.wetness", float, dropStrength, 5.0f, "Vehicle droplet visibility strength.");
  RTX_OPTION("rtx.wetness", float, dropRoughness, 2.7f, "Vehicle droplet roughness contribution.");

  RTX_OPTION("rtx.wetness", float, streakGridScale, 12.0f, "Streak grid cells per object unit.");
  RTX_OPTION("rtx.wetness", float, streakCycle, 1.8f, "Seconds per streak slide run.");
  RTX_OPTION("rtx.wetness", float, streakLength, 0.4f, "Object units a streak travels per run.");
  RTX_OPTION("rtx.wetness", float, streakSize, 1.0f, "Streak head size multiplier.");

  RTX_OPTION("rtx.wetness", float, waterWaveStrength, 0.5f, "Water (AnimatedWater) micro-wave normal wobble amplitude. 0 = off.");
  RTX_OPTION("rtx.wetness", float, waterWaveScale, 1.0f, "Water (AnimatedWater) micro-wave size multiplier. Larger = bigger, sparser chop.");

public:
  static void fillShaderParams(WetnessArgs& args) {
    args.enabled = enabled();
    args.roadOnly = roadOnly();
    args.nonRoadRoughnessOnly = nonRoadRoughnessOnly();
    args.respectNoRainZones = respectNoRainZones();
    args.rainEndRoughness = rainEndRoughness();
    args.puddleNoiseScale = puddleNoiseScale();
    args.puddleThresholdBase = puddleThresholdBase();
    args.puddleThresholdSpan = puddleThresholdSpan();
    args.puddleAlbedoDarken = puddleAlbedoDarken();
    args.puddleWaveStrength = puddleWaveStrength();
    args.puddleWaveScale = puddleWaveScale();

    args.rippleGridScale = rippleGridScale();
    args.rippleSpawnPeriod = rippleSpawnPeriod();
    args.rippleRadius = rippleRadius();

    args.rippleVehicleGridScale = rippleVehicleGridScale();
    args.rippleVehicleSpawnPeriod = rippleVehicleSpawnPeriod();
    args.rippleVehicleRadius = rippleVehicleRadius();

    args.dropDensity = dropDensity();
    args.dropSize = dropSize();
    args.dropStrength = dropStrength();
    args.dropRoughness = dropRoughness();

    args.streakGridScale = streakGridScale();
    args.streakCycle = streakCycle();
    args.streakLength = streakLength();
    args.streakSize = streakSize();
    args.waterWaveStrength = waterWaveStrength();
    args.waterWaveScale = waterWaveScale();
    args.weatherWetness = 1.0f;
    args.weatherWetnessActive = 0u;
    args.shoreOriginX = 0.0f;
    args.shoreOriginY = 0.0f;
    args.shoreInvSpacing = 0.0f;
    args.shoreStrength = 0.0f;
  }
};

} // namespace dxvk
#endif
