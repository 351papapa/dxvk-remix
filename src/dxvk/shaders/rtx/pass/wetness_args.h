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

  float puddleCellSize = 2.8f;       // meters
  float puddleCoverage = 0.7f;
  float puddleMicroAmplitude = 0.03f; // meters
  float puddleAlbedoDarken = 0.25f;
  float puddleWaveStrength = 0.08f;
  float puddleWaveScale = 1.0f;

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

  float puddleDepressionWeight = 0.3f;
  float puddleSlopeLimit = 0.053f;
  float puddleSlopeFeather = 2.4f;
  float puddleEdgeWidth = 0.025f;
  float puddleShoreWidth = 0.02f;
  float puddleShoreDarkening = 0.8f;
  float puddleRoughness = 0.035f;
  float puddleOcclusion = 0.5f;
  float puddleDetailDistance = 400.0f;
  float rainIntensity = 1.0f;
  float rainOcclusionMinHeight = 2.5f;
  float rainOcclusionFeather = 1.5f;
  uint rainMapEnabled = 0;
  uint rainMapResolution = 512;
  float rainMapOriginX = 0.0f;
  float rainMapOriginY = 0.0f;
  float rainMapExtent = 128.0f;
  float rainMapTexel = 0.25f;
  float rainMapTraceTop = 0.0f;
  float rainMapTraceBottom = 0.0f;
  float worldGroundMinRoughness = 0.0f;
  uint pad0 = 0;
  uint pad1 = 0;
  uint pad2 = 0;
};

#ifndef __cplusplus
float resolveWeatherWetness(uint tag, WetnessArgs wetness) {
  const float weatherAmount = (tag >> 24) == 2u ? wetness.weatherWetness : wetness.rainIntensity;
  return lerp(float(tag & 0xFFu) * (1.0f / 255.0f), saturate(weatherAmount),
              saturate(float(wetness.weatherWetnessActive) * (1.0f / 65535.0f)));
}
#endif

#ifdef __cplusplus
static_assert((sizeof(WetnessArgs) & 15) == 0);
static_assert(sizeof(WetnessArgs) == 240); // Shared C++/Slang frame-constant layout.

#include <algorithm>
#include <cmath>
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

  RTX_OPTION("rtx.wetness", float, puddleCellSize, 2.8f, "Ground unevenness cell size in meters. Larger cells form broader puddles.");
  RTX_OPTION("rtx.wetness", float, puddleCoverage, 0.7f, "Water level at full wetness, from 0 to 1. Higher values flood more of the ground unevenness.");
  RTX_OPTION("rtx.wetness", float, puddleMicroAmplitude, 0.03f, "Height amplitude of small ground unevenness in meters.");
  RTX_OPTION("rtx.wetness", float, puddleAlbedoDarken, 0.25f, "Albedo multiplier reached under water at 2 cm depth.");
  RTX_OPTION("rtx.wetness", float, puddleWaveStrength, 0.08f, "Standing-water wind ripple normal strength. 0 disables wind ripples.");
  RTX_OPTION("rtx.wetness", float, puddleWaveScale, 1.0f, "Wind ripple wavelength multiplier. 1 starts at a 12 cm wavelength.");
  RTX_OPTION("rtx.wetness", float, puddleDepressionWeight, 0.3f, "Weight of measured ground depressions in the puddle water level.");
  RTX_OPTION("rtx.wetness", float, puddleSlopeLimit, 0.053f, "Ground grade at which puddles start to drain. 0.053 is a 5.3 percent grade.");
  RTX_OPTION("rtx.wetness", float, puddleSlopeFeather, 2.4f, "Slope fade width as a multiple of the slope limit.");
  RTX_OPTION("rtx.wetness", float, puddleEdgeWidth, 0.025f, "Water depth in meters over which a puddle edge becomes fully covered.");
  RTX_OPTION("rtx.wetness", float, puddleShoreWidth, 0.02f, "Height in meters of the damp band above the puddle water line.");
  RTX_OPTION("rtx.wetness", float, puddleShoreDarkening, 0.8f, "Albedo multiplier of the damp puddle shore.");
  RTX_OPTION("rtx.wetness", float, worldGroundMinRoughness, 0.0f, "Minimum base roughness for rain-wet static world ground, capped at the dry material roughness. Standing water uses its own roughness. 0 disables the floor.");
  RTX_OPTION("rtx.wetness", float, puddleRoughness, 0.035f, "Roughness of standing water, from 0 to 1.");
  RTX_OPTION("rtx.wetness", float, puddleOcclusion, 0.5f, "How strongly static overhead cover suppresses puddles, from 0 to 1.");
  RTX_OPTION("rtx.wetness", float, puddleDetailDistance, 400.0f, "Maximum puddle shading distance in meters.");
  RTX_OPTION("rtx.wetness", bool, rainMapEnabled, false, "Measure static ground height, depressions, slope and rain cover around the camera. Puddles fade out at the map boundary. Requires ray queries each frame.");
  RTX_OPTION("rtx.wetness", uint32_t, rainMapResolution, 512, "Rain map texels per side, from 64 to 2048. Increasing resolution increases ray queries and memory use.");
  RTX_OPTION("rtx.wetness", float, rainMapExtent, 128.0f, "Side length of the rain map in meters. Larger extents cover more ground at lower spatial precision.");
  RTX_OPTION("rtx.wetness", float, rainMapTraceAbove, 100.0f, "Rain map trace height above the camera, in meters.");
  RTX_OPTION("rtx.wetness", float, rainMapTraceBelow, 60.0f, "Rain map trace depth below the camera, in meters.");
  RTX_OPTION("rtx.wetness", float, rainOcclusionMinHeight, 2.5f, "Minimum cover height above ground in meters. Lower obstacles do not act as roofs.");
  RTX_OPTION("rtx.wetness", float, rainOcclusionFeather, 1.5f, "Rain cover blur radius in meters, capped at 12 map texels.");
  RTX_OPTION("rtx.wetness", bool, surfaceRetentionEnabled, false, "Keep authored surface wetness after rain stops, then dry it gradually. Rain impacts continue to follow the current rainfall.");
  RTX_OPTION("rtx.wetness", float, surfaceHoldSeconds, 90.0f, "Seconds surface wetness is held after rain stops.");
  RTX_OPTION("rtx.wetness", float, surfaceDrySeconds, 120.0f, "Seconds to dry retained surface wetness after the hold period.");

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
    args.puddleCellSize = puddleCellSize();
    args.puddleCoverage = puddleCoverage();
    args.puddleMicroAmplitude = puddleMicroAmplitude();
    args.puddleAlbedoDarken = puddleAlbedoDarken();
    args.puddleWaveStrength = puddleWaveStrength();
    args.puddleWaveScale = puddleWaveScale();
    args.puddleDepressionWeight = puddleDepressionWeight();
    args.puddleSlopeLimit = puddleSlopeLimit();
    args.puddleSlopeFeather = puddleSlopeFeather();
    args.puddleEdgeWidth = puddleEdgeWidth();
    args.puddleShoreWidth = puddleShoreWidth();
    args.puddleShoreDarkening = puddleShoreDarkening();
    args.worldGroundMinRoughness = worldGroundMinRoughness();
    args.puddleRoughness = puddleRoughness();
    args.puddleOcclusion = puddleOcclusion();
    args.puddleDetailDistance = puddleDetailDistance();

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

  static void sanitizePuddleParams(WetnessArgs& args) {
    const auto finite = [](float value, float fallback, float low, float high) {
      return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
    };
    args.puddleCellSize = finite(args.puddleCellSize, 2.8f, 0.01f, 1000.0f);
    args.puddleCoverage = finite(args.puddleCoverage, 0.7f, 0.0f, 1.0f);
    args.puddleMicroAmplitude = finite(args.puddleMicroAmplitude, 0.03f, 0.0001f, 1.0f);
    args.puddleDepressionWeight = finite(args.puddleDepressionWeight, 0.3f, 0.0f, 10.0f);
    args.puddleSlopeLimit = finite(args.puddleSlopeLimit, 0.053f, 0.0001f, 1.0f);
    args.puddleSlopeFeather = finite(args.puddleSlopeFeather, 2.4f, 0.001f, 10.0f);
    args.puddleEdgeWidth = finite(args.puddleEdgeWidth, 0.025f, 0.0001f, 1.0f);
    args.puddleShoreWidth = finite(args.puddleShoreWidth, 0.02f, 0.0001f, 1.0f);
    args.puddleShoreDarkening = finite(args.puddleShoreDarkening, 0.8f, 0.0f, 1.0f);
    args.puddleAlbedoDarken = finite(args.puddleAlbedoDarken, 0.25f, 0.0f, 1.0f);
    args.worldGroundMinRoughness = finite(args.worldGroundMinRoughness, 0.0f, 0.0f, 1.0f);
    args.puddleRoughness = finite(args.puddleRoughness, 0.035f, 0.0f, 1.0f);
    args.puddleOcclusion = finite(args.puddleOcclusion, 0.5f, 0.0f, 1.0f);
    args.puddleDetailDistance = finite(args.puddleDetailDistance, 400.0f, 0.0f, 10000.0f);
    args.puddleWaveStrength = finite(args.puddleWaveStrength, 0.08f, 0.0f, 2.0f);
    args.puddleWaveScale = finite(args.puddleWaveScale, 1.0f, 0.01f, 100.0f);
  }

};

} // namespace dxvk
#endif
