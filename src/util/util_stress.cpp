/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#include "util_stress.h"

#include <chrono>
#include <thread>

#include "log/log.h"
#include "util_string.h"

namespace dxvk::stress {

  std::atomic<bool> g_active { false };

  namespace {
    std::atomic<uint32_t> s_delayUs { 0 };
    std::atomic<uint32_t> s_delayChance { 0 };  // the chance scaled to 2^32 - 1
    std::atomic<uint32_t> s_seed { 1 };
    std::atomic<bool> s_forceSlotReuse { false };
    std::atomic<uint64_t> s_visits[static_cast<uint32_t>(Site::Count)];
    std::atomic<uint64_t> s_delays { 0 };
    std::atomic<uint64_t> s_siteDelays[static_cast<uint32_t>(Site::Count)];
    std::atomic<int64_t> s_nextLogTicks { 0 };
    constexpr std::chrono::seconds kLogPeriod { 10 };

    uint64_t splitmix64(uint64_t x) {
      x += 0x9E3779B97F4A7C15ull;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
      return x ^ (x >> 31);
    }

    // Waits on the clock, since a sleep on Windows can round up to the timer period.
    void wait(uint32_t us) {
      const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
      if (us > 2000) {
        std::this_thread::sleep_for(std::chrono::microseconds(us - 1500));
      }
      while (std::chrono::steady_clock::now() < end) {
        std::this_thread::yield();
      }
    }
  }

  void configure(const Config& config) {
    const float chance = config.delayChance < 0.f ? 0.f : (config.delayChance > 1.f ? 1.f : config.delayChance);
    s_delayUs.store(config.delayUs, std::memory_order_relaxed);
    s_delayChance.store(static_cast<uint32_t>(static_cast<double>(chance) * 4294967295.0), std::memory_order_relaxed);
    s_seed.store(config.seed, std::memory_order_relaxed);
    s_forceSlotReuse.store(config.forceSlotReuse, std::memory_order_relaxed);
    for (auto& visits : s_visits) {
      visits.store(0, std::memory_order_relaxed);
    }
    const bool delays = config.delayUs != 0 && chance > 0.f;
    g_active.store(delays || config.forceSlotReuse, std::memory_order_release);
  }

  Config config() {
    Config config;
    config.delayUs = s_delayUs.load(std::memory_order_relaxed);
    config.delayChance = static_cast<float>(s_delayChance.load(std::memory_order_relaxed) / 4294967295.0);
    config.seed = s_seed.load(std::memory_order_relaxed);
    config.forceSlotReuse = s_forceSlotReuse.load(std::memory_order_relaxed);
    return config;
  }

  // The n-th visit of a site draws the n-th number of the site's sequence, so a run with the same seed delays the
  // same visits of each site by the same amounts, whichever thread makes them.
  void delaySlow(Site site) {
    const uint32_t maxUs = s_delayUs.load(std::memory_order_relaxed);
    const uint32_t chance = s_delayChance.load(std::memory_order_relaxed);
    if (maxUs == 0 || chance == 0) {
      return;
    }
    const uint32_t siteIndex = static_cast<uint32_t>(site);
    const uint64_t visit = s_visits[siteIndex].fetch_add(1, std::memory_order_relaxed);
    const uint64_t stream = splitmix64(static_cast<uint64_t>(s_seed.load(std::memory_order_relaxed)) * 8 + siteIndex);
    const uint64_t r = splitmix64(stream ^ visit);
    if (static_cast<uint32_t>(r) > chance) {
      return;
    }
    s_siteDelays[siteIndex].fetch_add(1, std::memory_order_relaxed);
    s_delays.fetch_add(1, std::memory_order_relaxed);
    // One line at the first delay and then at most every kLogPeriod.
    const int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
    const int64_t period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(kLogPeriod).count();
    int64_t next = s_nextLogTicks.load(std::memory_order_relaxed);
    if (now >= next && s_nextLogTicks.compare_exchange_strong(next, now + period)) {
      Logger::info(str::format("[Stress] ", s_delays.load(std::memory_order_relaxed), " delays injected (pool tasks ",
                               s_siteDelays[0].load(), ", command stream chunks ", s_siteDelays[1].load(),
                               ", scene command drains ", s_siteDelays[2].load(), ")"));
    }
    wait(1 + static_cast<uint32_t>((r >> 32) % maxUs));
  }

  uint64_t delaysInjected() {
    return s_delays.load(std::memory_order_relaxed);
  }

  bool forceSlotReuse() {
    return s_forceSlotReuse.load(std::memory_order_relaxed);
  }

}
