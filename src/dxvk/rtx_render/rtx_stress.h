#pragma once

#include "rtx_option.h"

namespace dxvk {
  class DxvkDevice;

  // rtx.debug.stress.*: the options of stress mode (src/util/util_stress.h). Off by default, and never saved.
  class RtxStress {
  public:
    static void onChange(DxvkDevice* device);

    RTX_OPTION_ARGS("rtx.debug.stress", uint32_t, delayUs, 0,
                    "Stress mode: the longest random delay, in microseconds, at the start of worker thread pool tasks, "
                    "command stream chunks and scene command drains, so the render thread lags its caller by varying "
                    "amounts. 0 turns the delays off. For finding timing bugs; it slows everything down.",
                    args.flags = RtxOptionFlags::NoSave,
                    args.onChangeCallback = &onChange);
    RTX_OPTION_ARGS("rtx.debug.stress", float, delayChance, 0.1f,
                    "Stress mode: the chance, from 0 to 1, that a site visit is delayed while rtx.debug.stress.delayUs "
                    "is above 0.",
                    args.flags = RtxOptionFlags::NoSave,
                    args.minValue = 0.f,
                    args.maxValue = 1.f,
                    args.onChangeCallback = &onChange);
    RTX_OPTION_ARGS("rtx.debug.stress", bool, forceSlotReuse, false,
                    "Stress mode: worker thread pools use one task slot per worker and the staged bone matrix blocks "
                    "are rewound at every draw, so a slot is reused as early as the code allows.",
                    args.flags = RtxOptionFlags::NoSave,
                    args.onChangeCallback = &onChange);
    RTX_OPTION_ARGS("rtx.debug.stress", uint32_t, seed, 1,
                    "Stress mode: the seed of the random delays, logged when stress mode starts. The same seed delays "
                    "the same visits of each site by the same amounts. 0 picks a seed from the clock.",
                    args.flags = RtxOptionFlags::NoSave,
                    args.onChangeCallback = &onChange);

    // Passes the options to the stress sites, and logs the configuration when it changes.
    static void apply();
  };
}
