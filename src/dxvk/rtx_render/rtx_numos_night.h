#pragma once

#include "rtx_option.h"

namespace dxvk {

  struct NumosNightAppearance {
    RTX_OPTION("rtx.numos", bool, nightAppearance, false,
      "Use separate artistic controls for the night sky seen by the camera.");
    RTX_OPTION_ARGS("rtx.numos", float, moonDiskBrightness, 1.0f,
      "Multiply the apparent brightness of moon disks in the camera sky.", args.minValue = 0.0f, args.maxValue = 2.0f);
    RTX_OPTION_ARGS("rtx.numos", float, starBrightness, 1.0f,
      "Multiply the brightness of visible catalogue stars.", args.minValue = 0.0f, args.maxValue = 64.0f);
    RTX_OPTION_ARGS("rtx.numos", float, starMagnitudeLimit, 7.5f,
      "Faintest catalogue stars to show. A lower magnitude shows fewer stars.", args.minValue = -2.0f, args.maxValue = 7.5f);
    RTX_OPTION_ARGS("rtx.numos", float, milkyWayBrightness, 1.0f,
      "Multiply the brightness of the visible Milky Way and its unresolved stars.", args.minValue = 0.0f, args.maxValue = 8192.0f);
    RTX_OPTION("rtx.numos", bool, legacyMilkyWay, false,
      "Show the procedural Milky Way fixed in world space in the camera sky.");
    RTX_OPTION_ARGS("rtx.numos", float, legacyMilkyWayBrightness, 0.05f,
      "Strength of the procedural Milky Way.", args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.numos", Vector3, legacyMilkyWayOuterColor, Vector3(0.5f, 0.55f, 0.75f),
      "Linear RGB colour of the procedural Milky Way's outer band.", args.minValue = Vector3(0.0f), args.maxValue = Vector3(1.0f));
    RTX_OPTION_ARGS("rtx.numos", Vector3, legacyMilkyWayCoreColor, Vector3(1.0f, 0.85f, 0.55f),
      "Linear RGB colour of the procedural Milky Way's core.", args.minValue = Vector3(0.0f), args.maxValue = Vector3(1.0f));
    RTX_OPTION_ARGS("rtx.numos", float, legacyMilkyWayDustAmount, 0.6f,
      "Strength of the dust lanes in the procedural Milky Way.", args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.numos", Vector3, nightSkyColor, Vector3(0.06f, 0.12f, 0.32f),
      "Linear RGB colour of the added night sky background.", args.minValue = Vector3(0.0f), args.maxValue = Vector3(1.0f));
    RTX_OPTION_ARGS("rtx.numos", float, nightSkyBrightness, 0.0f,
      "Strength of the coloured night sky background. Fades in as the Sun descends below the horizon.",
      args.minValue = 0.0f, args.maxValue = 0.05f);
  };

}
