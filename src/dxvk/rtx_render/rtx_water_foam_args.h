#pragma once
// Pure construction of WaterFoamArgs.
//
// This lives in a header, separate from WaterSystem::compositeFoam, so a unit test
// can exercise the same code the runtime uses. The bug this guards against is
// specific and was real: compositeFoam builds `WaterFoamArgs args {}`, which
// zero-initialises every field, so a field whose assignment is missing silently
// reaches the shader as 0 with no compile error and no log. When that field was
// `strength`, the display shader computed
//     alpha = saturate(density * erosion * pattern * args.strength)
// as identically 0 and every intersection foam disappeared.
//
// The initialiser below therefore seeds every field with a sentinel first. Any
// field the caller forgets keeps its sentinel, and waterFoamArgsIsFullySet()
// reports it, so the omission is caught by a test instead of by the user's eyes.
#include "rtx/pass/water_foam.h"

namespace dxvk {
namespace WaterFoamBuilder {

struct WaterFoamFill {
  uint32_t width = 0;
  uint32_t height = 0;
  float lineWidth = 0.85f;
  float strength = 0.8f;
  float maxDistance = 100.0f;
  float maxMotion = 0.0f;
  uint32_t oceanOnly = 1u;
  uint32_t fieldSize = 257u;
  float originX = 0.0f;
  float originY = 0.0f;
  float previousX = 0.0f;
  float previousY = 0.0f;
  float spacing = 0.4f;
  float dt = 0.0f;
  uint32_t reset = 1u;
  float skyAmbient = 1.0f;
  uint32_t sourceEnabled = kWATER_FOAM_SOURCE_ON;
  uint32_t excludeCharacters = kWATER_FOAM_EXCLUDE_CHARACTERS_ON;
  float runupWidth = 2.4f;
  float waveResponse = 2.5f;
  float lifetime = 7.0f;
  float wetnessLifetime = 20.0f;
  float textureScale = 3.0f;
  float time = 0.0f;
  float crestStrength = 2.0f;
  float crestThreshold = 0.06f;
  float oceanWaveStrength = 1.0f;
  uint32_t waveInputAvailable = 0u;
  float shoreHeight = 1.06f;
  float shoreSpacing = 8.9f;
  float shorePeriod = 4.5f;
  float shoreDepth = 3.1f;
};

// Seeds every field with a sentinel, then assigns every field from cfg.
inline WaterFoamArgs makeWaterFoamArgs(const WaterFoamFill& cfg) {
  WaterFoamArgs args {};
  args.width = kWATER_FOAM_UNSET_U32;
  args.height = kWATER_FOAM_UNSET_U32;
  args.lineWidth = kWATER_FOAM_UNSET_F32;
  args.strength = kWATER_FOAM_UNSET_F32;
  args.maxDistance = kWATER_FOAM_UNSET_F32;
  args.maxMotion = kWATER_FOAM_UNSET_F32;
  args.oceanOnly = kWATER_FOAM_UNSET_U32;
  args.fieldSize = kWATER_FOAM_UNSET_U32;
  args.originX = kWATER_FOAM_UNSET_F32;
  args.originY = kWATER_FOAM_UNSET_F32;
  args.previousX = kWATER_FOAM_UNSET_F32;
  args.previousY = kWATER_FOAM_UNSET_F32;
  args.spacing = kWATER_FOAM_UNSET_F32;
  args.dt = kWATER_FOAM_UNSET_F32;
  args.reset = kWATER_FOAM_UNSET_U32;
  args.skyAmbient = kWATER_FOAM_UNSET_F32;
  args.sourceEnabled = kWATER_FOAM_UNSET_U32;
  args.excludeCharacters = kWATER_FOAM_UNSET_U32;
  args.runupWidth = kWATER_FOAM_UNSET_F32;
  args.waveResponse = kWATER_FOAM_UNSET_F32;
  args.lifetime = kWATER_FOAM_UNSET_F32;
  args.wetnessLifetime = kWATER_FOAM_UNSET_F32;
  args.textureScale = kWATER_FOAM_UNSET_F32;
  args.time = kWATER_FOAM_UNSET_F32;
  args.crestStrength = kWATER_FOAM_UNSET_F32;
  args.crestThreshold = kWATER_FOAM_UNSET_F32;
  args.oceanWaveStrength = kWATER_FOAM_UNSET_F32;
  args.waveInputAvailable = kWATER_FOAM_UNSET_U32;
  args.shoreHeight = kWATER_FOAM_UNSET_F32;
  args.shoreSpacing = kWATER_FOAM_UNSET_F32;
  args.shorePeriod = kWATER_FOAM_UNSET_F32;
  args.shoreDepth = kWATER_FOAM_UNSET_F32;


  args.width = cfg.width;
  args.height = cfg.height;
  // lineWidth must never be 0: the update shader divides by it, and a zero made
  // every water cell read as if it were touching the contact.
  args.lineWidth = cfg.lineWidth > 0.0f ? cfg.lineWidth : 0.85f;
  args.strength = cfg.strength;
  args.maxDistance = cfg.maxDistance;
  args.maxMotion = cfg.maxMotion;
  args.oceanOnly = cfg.oceanOnly;
  args.fieldSize = cfg.fieldSize;
  args.originX = cfg.originX;
  args.originY = cfg.originY;
  args.previousX = cfg.previousX;
  args.previousY = cfg.previousY;
  args.spacing = cfg.spacing;
  args.dt = cfg.dt;
  args.reset = cfg.reset;
  args.skyAmbient = cfg.skyAmbient;
  args.sourceEnabled = cfg.sourceEnabled;
  args.excludeCharacters = cfg.excludeCharacters;
  args.runupWidth = cfg.runupWidth;
  args.waveResponse = cfg.waveResponse;
  args.lifetime = cfg.lifetime;
  args.wetnessLifetime = cfg.wetnessLifetime;
  args.textureScale = cfg.textureScale;
  args.time = cfg.time;
  args.crestStrength = cfg.crestStrength;
  args.crestThreshold = cfg.crestThreshold;
  args.oceanWaveStrength = cfg.oceanWaveStrength;
  args.waveInputAvailable = cfg.waveInputAvailable;
  args.shoreHeight = cfg.shoreHeight;
  args.shoreSpacing = cfg.shoreSpacing;
  args.shorePeriod = cfg.shorePeriod;
  args.shoreDepth = cfg.shoreDepth;

  return args;
}

// True when no field still holds its sentinel and no float field is non-finite.
inline bool waterFoamArgsIsFullySet(const WaterFoamArgs& a) {
  if (a.width == kWATER_FOAM_UNSET_U32) return false;
  if (a.height == kWATER_FOAM_UNSET_U32) return false;
  if (a.oceanOnly == kWATER_FOAM_UNSET_U32) return false;
  if (a.fieldSize == kWATER_FOAM_UNSET_U32) return false;
  if (a.reset == kWATER_FOAM_UNSET_U32) return false;
  if (a.sourceEnabled == kWATER_FOAM_UNSET_U32) return false;
  if (a.excludeCharacters == kWATER_FOAM_UNSET_U32) return false;
  if (a.waveInputAvailable == kWATER_FOAM_UNSET_U32) return false;
  const float floats[] = {a.lineWidth, a.strength, a.maxDistance, a.maxMotion,
    a.originX, a.originY, a.previousX, a.previousY, a.spacing, a.dt, a.skyAmbient,
    a.runupWidth, a.waveResponse, a.lifetime, a.wetnessLifetime, a.textureScale, a.time, a.crestStrength, a.crestThreshold, a.oceanWaveStrength, a.shoreHeight, a.shoreSpacing, a.shorePeriod, a.shoreDepth};
  for (float f : floats) {
    if (f == kWATER_FOAM_UNSET_F32) return false;
    if (!std::isfinite(f)) return false;
  }
  return true;
}

} // namespace WaterFoamBuilder
} // namespace dxvk
