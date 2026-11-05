#include "rtx_stress.h"

#include <chrono>
#include <mutex>

#include "../../util/log/log.h"
#include "../../util/util_stress.h"
#include "../../util/util_string.h"

namespace dxvk {

  void RtxStress::onChange(DxvkDevice*) {
    apply();
  }

  void RtxStress::apply() {
    static std::mutex s_mutex;
    static uint32_t s_clockSeed = 0;
    static bool s_logged = false;
    static bool s_applied = false;
    static stress::Config s_last;
    std::lock_guard<std::mutex> lock(s_mutex);

    stress::Config config;
    config.delayUs = delayUs();
    config.delayChance = delayChance();
    config.forceSlotReuse = forceSlotReuse();
    config.seed = seed();
    if (config.seed == 0) {
      if (s_clockSeed == 0) {
        s_clockSeed = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 1u;
      }
      config.seed = s_clockSeed;
    }

    // A new configuration restarts the random sequences, so an unchanged one is not passed on again.
    const stress::Config& previous = s_last;
    const bool changed = previous.delayUs != config.delayUs || previous.delayChance != config.delayChance ||
                         previous.forceSlotReuse != config.forceSlotReuse || previous.seed != config.seed;
    if (s_applied && !changed) {
      return;
    }
    s_applied = true;
    s_last = config;
    stress::configure(config);
    const bool on = stress::active();
    if (on) {
      Logger::info(str::format("[Stress] on: delayUs=", config.delayUs, " delayChance=", config.delayChance,
                               " forceSlotReuse=", config.forceSlotReuse ? 1 : 0, " seed=", config.seed));
      s_logged = true;
    } else if (!on && s_logged) {
      Logger::info(str::format("[Stress] off (", stress::delaysInjected(), " delays injected)"));
      s_logged = false;
    }
  }

}
