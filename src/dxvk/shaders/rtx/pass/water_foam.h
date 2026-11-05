#pragma once
#include "rtx/utility/shader_types.h"
#define WATER_FOAM_COLOR 30
#define WATER_FOAM_FIELD 31
#define WATER_FOAM_NEXT 32
#define WATER_FOAM_PATTERN 33
#define WATER_FOAM_PREVIOUS_HEIGHT 34
#define WATER_FOAM_WAVES 35
#define WATER_FOAM_SHORE 38
#define WATER_FOAM_BEACH_MASK 39
#define WATER_FOAM_SKY 36
#define WATER_FOAM_ENVIRONMENT_SAMPLER 37
struct WaterFoamArgs {
  uint width;
  uint height;
  float lineWidth;
  float strength;
  float maxDistance;
  float maxMotion;
  uint oceanOnly;
  uint fieldSize;
  float originX;
  float originY;
  float previousX;
  float previousY;
  float spacing;
  float dt;
  uint reset;
  // Multiplier on the ambient term, applied directly (0 removes the ambient
  // entirely). Reuses the former padding slot, so size and offsets are unchanged.
  float skyAmbient;
  // Whether this pass may seed new density. Deliberately separate from lineWidth:
  // deriving "off" from lineWidth == 0 made the update shader compute
  // contact / lineWidth as 0/0, which made every water cell read as if it were
  // touching the contact. lineWidth must always carry a legal positive distance.
  uint sourceEnabled;
  // Skip the third-person player model when probing contacts. Separate from
  // sourceEnabled because overloading one field with both meanings made the
  // diagnostic ambiguous: a zero strength would hide a classification problem, so
  // "disable the exclusion and see" could not distinguish the two.
  uint excludeCharacters;
  float runupWidth;
  float waveResponse;
  float lifetime;
  float wetnessLifetime;
  float textureScale;
  float time;
  float crestStrength;
  float crestThreshold;
  float oceanWaveStrength;
  uint waveInputAvailable;
  float shoreHeight;
  float shoreSpacing;
  float shorePeriod;
  float shoreDepth;

};
static constexpr uint kWATER_FOAM_SOURCE_OFF = 0u;
static constexpr uint kWATER_FOAM_SOURCE_ON = 1u;
static constexpr uint kWATER_FOAM_EXCLUDE_CHARACTERS_OFF = 0u;
static constexpr uint kWATER_FOAM_EXCLUDE_CHARACTERS_ON = 1u;
// Sentinel that every field is seeded with before a fill, so a field that the
// fill forgets can be detected instead of silently reaching the shader as 0.
// This exists because a forgotten `args.strength = ...` made alpha identically 0
// in the display shader and removed all intersection foam with no error anywhere.
static constexpr uint kWATER_FOAM_UNSET_U32 = 0xDEADBEEFu;
static constexpr float kWATER_FOAM_UNSET_F32 = -1234567.0f;
