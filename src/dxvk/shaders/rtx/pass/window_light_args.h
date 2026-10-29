// SPDX-License-Identifier: MIT
// Local runtime window lighting. See CREDITS.md.
#pragma once
#include "../utility/shader_types.h"

struct WindowLightArgs {
  float litRatio = 0.0f;
  uint nightSeed = 0;
  uint streetLampOn = 0;
  uint enabled = 0;
};

#ifdef __cplusplus
#include "rtx_option.h"
static_assert(sizeof(WindowLightArgs) == 16);

namespace dxvk {
  struct WindowLightOptions {
    friend class ImGUI;
    RTX_OPTION("rtx.windowLight", bool, enabled, false, "Enable per-window emission and clock-driven street lamps.");
    RTX_OPTION("rtx.windowLight", std::string, textureNames, "", "Case-insensitive window texture name fragments separated by semicolons. Matches original captured names and replacement filenames.");
    RTX_OPTION("rtx.windowLight", float, litRatio, 0.3f, "Share of window islands lit at the start of the night, from 0 to 1. The share falls to zero by the end hour.");
    RTX_OPTION("rtx.windowLight", float, startHour, 19.0f, "Night start in game hours, from 0 to 24. Equal start and end hours keep a constant lit share.");
    RTX_OPTION("rtx.windowLight", float, endHour, 5.0f, "Night end in game hours, from 0 to 24. Night can cross midnight.");
    RTX_OPTION("rtx.windowLight", uint32_t, nightSeed, 0u, "Additional random seed. Each game night has a stable pattern; changing this value selects another pattern.");
    RTX_OPTION("rtx.windowLight", bool, streetLampsEnabled, false, "Classify street lamps separately. Lamps emit at night or in bad weather and are dark otherwise. Lamp matching takes precedence over window matching.");
    RTX_OPTION("rtx.windowLight", std::string, streetLampTextureNames, "", "Case-insensitive street lamp texture name fragments separated by semicolons. Lamp textures keep their original emissive footprint.");
  };
}
#endif
