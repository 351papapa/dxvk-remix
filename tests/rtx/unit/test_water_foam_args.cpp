// Regression test for the WaterFoamArgs construction path.
//
// Context: intersection foam disappeared entirely because compositeFoam()'s
// `args.strength = std::clamp(shoreFoamStrength(), 0.0f, 1.0f);` assignment was
// deleted by accident. `WaterFoamArgs args {}` zero-initialises, so strength
// reached the display shader as 0 and
//     alpha = saturate(density * erosion * pattern * args.strength)
// was identically 0. Nothing failed to compile, nothing logged a warning, and the
// field kept simulating correctly, so every other signal looked healthy.
//
// These tests exercise the same construction function the runtime calls, so a
// forgotten field fails here rather than in the game.

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>

#include "../../../src/dxvk/rtx_render/rtx_water_foam_args.h"

namespace {

int g_failures = 0;

void require(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

dxvk::WaterFoamBuilder::WaterFoamFill baseFill() {
  dxvk::WaterFoamBuilder::WaterFoamFill f {};
  f.width = 1920;
  f.height = 1080;
  f.lineWidth = 0.85f;
  f.strength = 0.8f;
  f.maxDistance = 100.0f;
  f.maxMotion = 0.35f * 0.016f;
  f.oceanOnly = 1u;
  f.fieldSize = 257u;
  f.originX = -51.2f;
  f.originY = -51.2f;
  f.previousX = -51.2f;
  f.previousY = -51.2f;
  f.spacing = 0.4f;
  f.dt = 0.016f;
  f.reset = 1u;
  f.skyAmbient = 1.0f;
  f.sourceEnabled = kWATER_FOAM_SOURCE_ON;
  f.excludeCharacters = kWATER_FOAM_EXCLUDE_CHARACTERS_ON;
  return f;
}

} // namespace

int main() {
  using namespace dxvk::WaterFoamBuilder;

  // 1. The exact regression: a normally configured foam pass must upload a
  //    non-zero strength, otherwise the display shader can never draw anything.
  {
    const WaterFoamArgs args = makeWaterFoamArgs(baseFill());
    require(args.strength > 0.0f, "normal configuration uploads strength > 0");
    require(args.strength == 0.8f, "normal configuration preserves strength 0.8");
    require(args.sourceEnabled == kWATER_FOAM_SOURCE_ON, "source enabled when requested");
    require(args.excludeCharacters == kWATER_FOAM_EXCLUDE_CHARACTERS_ON,
      "character exclusion on when requested");
  }

  // 2. No field may still hold its sentinel, and no float may be non-finite. This
  //    is what catches a field added later whose assignment is forgotten.
  {
    const WaterFoamArgs args = makeWaterFoamArgs(baseFill());
    require(waterFoamArgsIsFullySet(args), "every field is assigned by the fill");
    require(sizeof(WaterFoamArgs) == 32 * sizeof(uint32_t),
      "WaterFoamArgs has 32 32-bit scalars (128 bytes)");
    require(offsetof(WaterFoamArgs, strength) == 3 * sizeof(uint32_t),
      "strength stays at byte offset 12");
    require(offsetof(WaterFoamArgs, lineWidth) == 2 * sizeof(uint32_t),
      "lineWidth stays at byte offset 8");
    require(offsetof(WaterFoamArgs, sourceEnabled) == 16 * sizeof(uint32_t),
      "sourceEnabled stays at byte offset 64");
    require(offsetof(WaterFoamArgs, excludeCharacters) == 17 * sizeof(uint32_t),
      "excludeCharacters stays at byte offset 68");
    require(offsetof(WaterFoamArgs, runupWidth) == 72, "runupWidth is at byte offset 72");
    require(offsetof(WaterFoamArgs, waveResponse) == 76, "waveResponse is at byte offset 76");
    require(offsetof(WaterFoamArgs, lifetime) == 80, "lifetime is at byte offset 80");
    require(offsetof(WaterFoamArgs, wetnessLifetime) == 84, "wetnessLifetime is at byte offset 84");
    require(offsetof(WaterFoamArgs, textureScale) == 88, "textureScale is at byte offset 88");
    require(offsetof(WaterFoamArgs, time) == 92, "time is at byte offset 92");
    require(offsetof(WaterFoamArgs, crestStrength) == 96, "crestStrength is at byte offset 96");
    require(offsetof(WaterFoamArgs, crestThreshold) == 100, "crestThreshold is at byte offset 100");
    require(offsetof(WaterFoamArgs, oceanWaveStrength) == 104, "oceanWaveStrength is at byte offset 104");
    require(offsetof(WaterFoamArgs, waveInputAvailable) == 108, "waveInputAvailable is at byte offset 108");
    require(offsetof(WaterFoamArgs, shoreHeight) == 112, "shore parameters start at byte 112");
    require(offsetof(WaterFoamArgs, shoreDepth) == 124, "shore depth ends the 128 byte contract");
    require(args.shoreSpacing == 8.9f && args.shorePeriod == 4.5f, "shore timing defaults reach the shader");
  }

  // 3. An explicit 0 strength must still be honoured: that is the documented way to
  //    switch the foam off, so it must not be silently replaced by a default.
  {
    WaterFoamFill f = baseFill();
    f.strength = 0.0f;
    const WaterFoamArgs args = makeWaterFoamArgs(f);
    require(args.strength == 0.0f, "an explicit strength of 0 is honoured, not defaulted");
    require(waterFoamArgsIsFullySet(args), "strength 0 still counts as fully set");
  }

  // 4. lineWidth must never be 0, because the update shader divides by it. A zero
  //    produced contact / 0, which read as "every cell touches the contact" and
  //    saturated the entire ocean with foam.
  {
    WaterFoamFill f = baseFill();
    f.lineWidth = 0.0f;
    const WaterFoamArgs args = makeWaterFoamArgs(f);
    require(args.lineWidth > 0.0f, "a zero lineWidth is replaced by a legal positive one");

    f.lineWidth = -5.0f;
    const WaterFoamArgs neg = makeWaterFoamArgs(f);
    require(neg.lineWidth > 0.0f, "a negative lineWidth is replaced by a legal positive one");
  }

  // 5. Source off must be expressed by the flag while lineWidth stays legal, so the
  //    probe can be skipped without producing a degenerate ray or a division by 0.
  {
    WaterFoamFill f = baseFill();
    f.sourceEnabled = kWATER_FOAM_SOURCE_OFF;
    const WaterFoamArgs args = makeWaterFoamArgs(f);
    require(args.sourceEnabled == kWATER_FOAM_SOURCE_OFF, "source off is uploaded as off");
    require(args.lineWidth > 0.0f, "source off keeps a legal lineWidth");
    require(args.strength > 0.0f,
      "source off keeps strength positive so the display path stays independently testable");
  }

  // 6. The A/B switch for the player exclusion must be independent of everything
  //    else. Previously one overloaded field carried both meanings, so with strength
  //    accidentally 0 the exclusion toggle could not distinguish anything.
  {
    WaterFoamFill f = baseFill();
    f.excludeCharacters = kWATER_FOAM_EXCLUDE_CHARACTERS_OFF;
    const WaterFoamArgs args = makeWaterFoamArgs(f);
    require(args.excludeCharacters == kWATER_FOAM_EXCLUDE_CHARACTERS_OFF,
      "character exclusion can be disabled on its own");
    require(args.sourceEnabled == kWATER_FOAM_SOURCE_ON,
      "disabling character exclusion does not disturb the source flag");
    require(args.strength > 0.0f, "disabling character exclusion does not disturb strength");
  }

  // 7. Every field round-trips, including the ones added for this fix.
  {
    WaterFoamFill f = baseFill();
    f.skyAmbient = 0.25f;
    f.reset = 0u;
    f.dt = 0.033f;
    f.runupWidth = 1.7f;
    f.waveResponse = 3.4f;
    f.lifetime = 11.0f;
    f.wetnessLifetime = 4.5f;
    f.textureScale = 2.1f;
    f.time = 123.25f;
    f.crestStrength = 3.25f;
    f.crestThreshold = 0.12f;
    f.oceanWaveStrength = 1.75f;
    f.waveInputAvailable = 1u;
    const WaterFoamArgs args = makeWaterFoamArgs(f);
    require(args.width == f.width && args.height == f.height, "extent round-trips");
    require(args.strength == f.strength && args.lineWidth == f.lineWidth,
      "strength and lineWidth round-trip");
    require(args.skyAmbient == f.skyAmbient, "skyAmbient round-trips");
    require(args.reset == f.reset && args.dt == f.dt, "reset and dt round-trip");
    require(args.originX == f.originX && args.previousY == f.previousY,
      "origin and previous origin round-trip");
    require(args.fieldSize == f.fieldSize && args.spacing == f.spacing,
      "field size and spacing round-trip");
    require(args.maxDistance == f.maxDistance && args.maxMotion == f.maxMotion,
      "maxDistance and maxMotion round-trip");
    require(args.oceanOnly == f.oceanOnly, "oceanOnly round-trips");
    require(args.runupWidth == f.runupWidth, "runupWidth round-trips");
    require(args.waveResponse == f.waveResponse, "waveResponse round-trips");
    require(args.lifetime == f.lifetime, "lifetime round-trips");
    require(args.wetnessLifetime == f.wetnessLifetime, "wetnessLifetime round-trips");
    require(args.textureScale == f.textureScale, "textureScale round-trips");
    require(args.time == f.time, "time round-trips");
    require(args.crestStrength == f.crestStrength, "crestStrength round-trips");
    require(args.crestThreshold == f.crestThreshold, "crestThreshold round-trips");
    require(args.oceanWaveStrength == f.oceanWaveStrength, "oceanWaveStrength round-trips");
    require(args.waveInputAvailable == f.waveInputAvailable, "waveInputAvailable round-trips");
    require(waterFoamArgsIsFullySet(args), "the fully configured fill reports fully set");
  }

  // 8. The sentinel check must actually be able to fail, otherwise it proves nothing.
  {
    WaterFoamArgs forged = makeWaterFoamArgs(baseFill());
    forged.width = kWATER_FOAM_UNSET_U32;
    require(!waterFoamArgsIsFullySet(forged), "an unset field is detected");
    WaterFoamArgs missingWaveFlag = makeWaterFoamArgs(baseFill());
    missingWaveFlag.waveInputAvailable = kWATER_FOAM_UNSET_U32;
    require(!waterFoamArgsIsFullySet(missingWaveFlag), "unset waveInputAvailable is detected");

    WaterFoamArgs nonFinite = forged;
    nonFinite.width = 1u;
    nonFinite.strength = std::numeric_limits<float>::quiet_NaN();
    require(!waterFoamArgsIsFullySet(nonFinite), "a non-finite float is detected");
  }

  {
    const WaterFoamArgs defaults = makeWaterFoamArgs(WaterFoamFill {});
    require(defaults.runupWidth == 2.4f, "runupWidth defaults to 2.4");
    require(defaults.waveResponse == 2.5f, "waveResponse defaults to 2.5");
    require(defaults.lifetime == 7.0f, "lifetime defaults to 7 seconds");
    require(defaults.wetnessLifetime == 20.0f, "wetnessLifetime defaults to 20 seconds");
    require(defaults.textureScale == 3.0f, "textureScale defaults to 3");
    require(defaults.time == 0.0f, "time starts at zero");
    require(defaults.crestStrength == 2.0f, "crestStrength defaults to 2");
    require(defaults.crestThreshold == 0.06f, "crestThreshold defaults to 0.06");
    require(defaults.oceanWaveStrength == 1.0f, "oceanWaveStrength defaults to 1");
    require(defaults.waveInputAvailable == 0u, "missing wave input defaults to unavailable");
    require(waterFoamArgsIsFullySet(defaults), "the default fill is complete and finite");
  }

  // Zero independently disables runup or wave response without disabling contact foam.
  {
    WaterFoamFill f = baseFill();
    f.runupWidth = 0.0f;
    WaterFoamArgs args = makeWaterFoamArgs(f);
    require(args.runupWidth == 0.0f, "an explicit zero runupWidth is honoured");
    require(args.waveResponse == f.waveResponse, "zero runupWidth preserves waveResponse");
    require(waterFoamArgsIsFullySet(args), "zero runupWidth remains a fully set argument");

    f = baseFill();
    f.waveResponse = 0.0f;
    args = makeWaterFoamArgs(f);
    require(args.waveResponse == 0.0f, "an explicit zero waveResponse is honoured");
    require(args.runupWidth == f.runupWidth, "zero waveResponse preserves runupWidth");
    require(waterFoamArgsIsFullySet(args), "zero waveResponse remains a fully set argument");

    f.runupWidth = 0.0f;
    args = makeWaterFoamArgs(f);
    require(args.runupWidth == 0.0f && args.waveResponse == 0.0f,
      "runup and wave response can both be disabled");
    require(args.sourceEnabled == kWATER_FOAM_SOURCE_ON && args.strength == f.strength,
      "disabling runup and wave response preserves contact foam");
  }

  // Crest generation and grounded wetness can be disabled without disabling contact foam.
  {
    WaterFoamFill f = baseFill();
    f.crestStrength = 0.0f;
    f.wetnessLifetime = 0.0f;
    const WaterFoamArgs args = makeWaterFoamArgs(f);
    require(args.crestStrength == 0.0f && args.wetnessLifetime == 0.0f,
      "explicit zero crestStrength and wetnessLifetime are preserved");
    require(args.strength == f.strength && args.sourceEnabled == kWATER_FOAM_SOURCE_ON,
      "disabling crests and wetness preserves contact foam");
    require(waterFoamArgsIsFullySet(args), "zero optional feature values remain fully set");
  }

  // Corrupt one uploaded field at a time so every new validity check must work.
  {
    struct FloatField {
      const char* name;
      float WaterFoamArgs::* member;
    };
    const FloatField fields[] = {
      { "runupWidth", &WaterFoamArgs::runupWidth },
      { "waveResponse", &WaterFoamArgs::waveResponse },
      { "lifetime", &WaterFoamArgs::lifetime },
      { "wetnessLifetime", &WaterFoamArgs::wetnessLifetime },
      { "textureScale", &WaterFoamArgs::textureScale },
      { "time", &WaterFoamArgs::time },
      { "crestStrength", &WaterFoamArgs::crestStrength },
      { "crestThreshold", &WaterFoamArgs::crestThreshold },
      { "oceanWaveStrength", &WaterFoamArgs::oceanWaveStrength },
    };
    const float invalidValues[] = {
      kWATER_FOAM_UNSET_F32,
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
    };
    for (const FloatField& field : fields) {
      for (float value : invalidValues) {
        WaterFoamArgs invalid = makeWaterFoamArgs(baseFill());
        invalid.*(field.member) = value;
        require(!waterFoamArgsIsFullySet(invalid),
          std::string(field.name) + " rejects a sentinel or non-finite value");
      }
      WaterFoamArgs finite = makeWaterFoamArgs(baseFill());
      finite.*(field.member) = std::numeric_limits<float>::max();
      require(waterFoamArgsIsFullySet(finite),
        std::string(field.name) + " accepts a finite value at the float boundary");
    }
  }

  if (g_failures == 0) {
    std::puts("PASS: WaterFoamArgs construction is complete, finite and layout-stable.");
    return 0;
  }
  std::printf("FAILURES: %d\n", g_failures);
  return 1;
}
