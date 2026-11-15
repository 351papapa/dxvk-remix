#pragma once

#include <algorithm>
#include <cmath>

#include "rtx_option.h"

namespace dxvk {

  // Host limits for local-light haze. The scene's lights and surface lighting are unchanged.
  class NumosHazeLights {
  public:
    RTX_OPTION_ARGS("rtx.numos", float, hazeLightDistanceMeters, 0.0f,
      "Maximum camera-to-light distance for local-light haze, in measured world meters. 0 = unlimited. "
      "Lights fade near this boundary. Ordinary surface lighting is unchanged.",
      args.minValue = 0.0f, args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.numos", float, hazeLightFadeMeters, 0.0f,
      "Width of the camera-distance fade for local-light haze, in measured world meters. "
      "0 = the last 20 percent of Haze Light Camera Distance. Widths larger than the distance fade "
      "over the whole range. Fading also reduces the light's cutoff-derived haze reach.",
      args.minValue = 0.0f, args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.numos", float, hazeSpotlightReachMeters, 0.0f,
      "Maximum haze reach from each spotlight source, in measured world meters. 0 = unlimited. "
      "Caps the influence sphere without extending shorter lights. The existing Local Light Range "
      "Window controls range falloff. Unshaped lights, lightning and ordinary surface lighting are unchanged.",
      args.minValue = 0.0f, args.flags = RtxOptionFlags::UserSetting);

    static float positiveFinite(float value) {
      return std::isfinite(value) && value > 0.0f ? value : 0.0f;
    }

    // Match NUMOS's measured scale, before its separate artistic world/haze scaling.
    static float resolveUnitsPerMeter(float configured, float host) {
      return positiveFinite(configured) > 0.0f ? configured : std::max(positiveFinite(host), 0.001f);
    }

    static float worldDistance(float meters, float unitsPerMeter) {
      return positiveFinite(positiveFinite(meters) * positiveFinite(unitsPerMeter));
    }

    static float cameraWeight(float distanceWorld, float rangeWorld, float fadeWorld) {
      const float range = positiveFinite(rangeWorld);
      if (range == 0.0f || !std::isfinite(distanceWorld)) {
        return 1.0f;
      }
      if (distanceWorld >= range) {
        return 0.0f;
      }
      const float configuredFade = positiveFinite(fadeWorld);
      const float width = configuredFade > 0.0f ? std::min(configuredFade, range) : range * 0.2f;
      if (!(width > 0.0f) || distanceWorld <= range - width) {
        return 1.0f;
      }
      const float remaining = std::clamp((range - distanceWorld) / width, 0.0f, 1.0f);
      return remaining * remaining * (3.0f - 2.0f * remaining);
    }

    static float cappedReach(float influenceRadius, float reachWorld) {
      const float cap = positiveFinite(reachWorld);
      return cap > 0.0f && std::isfinite(influenceRadius) && influenceRadius > cap ? cap : influenceRadius;
    }
  };

}
