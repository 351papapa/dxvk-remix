#pragma once

// The host's light calibration.
//
// A game's lights and emissives come in the game's own units, while a physical sky provider (NUMOS) lights the
// scene in physical units (FrameOutputs::luxPerUnit: one Remix illuminance unit in lux). These options convert
// the game's units to the same physical scale, so a street lamp and the moon land at their real ratio.
//
//   lightsEv            every game / API / converted light (not the sky provider's own sun, moons and
//                       lightning), as a scale 2^lightsEv on their radiance;
//   emissiveOffsetEv    emissive surfaces relative to the lights: their scale is 2^(lightsEv + emissiveOffsetEv).
//                       Games usually author emissives and lights on different scales, hence a separate offset,
//                       stated relative to the lights so the two move together.
//
// Both default to 0, which takes the game's units as Remix's. rtx.emissiveIntensity and
// rtx.lightConversionIntensityFactor stay content factors at 1, and a game integration's own light gain should
// only match the engine's falloff. Measure with the photometer (dev menu, Diagnostics) and the handoff's procedure.

#include <cmath>

#include "rtx_option.h"

namespace dxvk {

  class HostCalibration {
  public:
    RTX_OPTION("rtx.hostCalibration", float, lightsEv, 0.0f,
               "Converts the game's light units to physical units: the radiance of every game, API and converted light "
               "is scaled by 2^lightsEv (stops). The sky provider's own lights (the NUMOS sun, moons and lightning) are "
               "physical already and are not scaled. 0 = use the game's units as they are. Measure it with the "
               "photometer (Diagnostics).");
    RTX_OPTION("rtx.hostCalibration", float, emissiveOffsetEv, 0.0f,
               "The level of emissive surfaces relative to the lights, in stops: emissives are scaled by 2^(lightsEv + "
               "emissiveOffsetEv). Games author emissives and lights on different scales, and this sets the difference.");

    static float lightScale() { return std::exp2(lightsEv()); }
    static float emissiveScale() { return std::exp2(lightsEv() + emissiveOffsetEv()); }
  };

}
