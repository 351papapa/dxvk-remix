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
#pragma once

#include <atomic>
#include <cstdint>

#include "util_likely.h"

namespace dxvk::stress {

  // Stress mode (rtx.debug.stress.*): random delays where work changes threads, and the smallest legal number of
  // reusable slots, so that timing and lifetime bugs show up without a loaded machine. Every site tests one flag, so
  // the cost while stress is off is one load and one predictable branch.

  // The places a delay is injected. Each site draws its own random sequence.
  enum class Site : uint32_t {
    PoolTask,       // the start of a WorkerThreadPool task, on its worker
    CsChunk,        // the start of a command stream chunk, on the CS thread
    SceneCommands,  // the start of a scene command drain, on the CS thread
    Count
  };

  struct Config {
    uint32_t delayUs = 0;        // the longest delay; 0 turns delays off
    float delayChance = 0.f;     // the chance that a site visit is delayed
    bool forceSlotReuse = false;
    uint32_t seed = 1;
  };

  // True while any stress is on.
  extern std::atomic<bool> g_active;

  // Sets the configuration. Thread safe; sites see the change on their next visit.
  void configure(const Config& config);

  Config config();

  // The delay of one site visit while stress is on.
  void delaySlow(Site site);

  // The number of delays injected so far, over all sites.
  uint64_t delaysInjected();

  inline bool active() {
    return unlikely(g_active.load(std::memory_order_relaxed));
  }

  // A random delay at a site, while stress delays are on.
  inline void delay(Site site) {
    if (active()) {
      delaySlow(site);
    }
  }

  // True while stress asks reusable slots to be reused as early as the code allows. Callers test active() first.
  bool forceSlotReuse();

}
