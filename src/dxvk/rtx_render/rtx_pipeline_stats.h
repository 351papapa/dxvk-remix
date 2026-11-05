#pragma once

#include <atomic>
#include <string>

#include "rtx_option.h"
#include "../dxvk_include.h"

namespace dxvk {
  class DxvkShader;
  class DxvkPipelineCache;

  // rtx.debug.logPipelineStats.
  // When on at device creation, DxvkAdapter enables VK_KHR_pipeline_executable_properties, the pipeline creation
  // points add VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR, and right after each pipeline is created (on whichever
  // thread created it: the CS thread, a shader prewarm / state cache worker, or a ray tracing pipeline compile task)
  // the driver's per-executable statistics are logged, once per distinct pipeline name:
  //   [PipelineStats] pipeline=<name> stage=<stages> exec=<executable> subgroup_size=<n> <stat>=<value> ...
  // Off: the only cost is one load of a bool per pipeline creation.
  class RtxPipelineStats {
  public:
    RTX_OPTION("rtx.debug", bool, logPipelineStats, false,
               "Logs the driver's shader statistics (register count, spills, instruction count, binary size and "
               "whatever else the driver reports) for every compute, ray tracing and graphics pipeline Remix creates, "
               "as one [PipelineStats] line per pipeline in remix-dxvk.log, with the Remix shader names. Takes effect "
               "after a restart. Needs VK_KHR_pipeline_executable_properties, and can make shader compilation slightly "
               "slower.");

    // Called by DxvkAdapter once the device exists: true when the extension and feature were enabled.
    static void setCaptureEnabled(bool enabled) {
      s_captureEnabled.store(enabled, std::memory_order_relaxed);
    }

    // VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR while capture is enabled, 0 otherwise.
    static VkPipelineCreateFlags captureFlags() {
      return s_captureEnabled.load(std::memory_order_relaxed) ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;
    }

    // Query and log the statistics of a pipeline created with captureFlags(). Must run after the pipeline creation
    // (including any deferred operation) has completed. Thread safe. Each name is logged once.
    static void logPipeline(const vk::DeviceFn& vkd, VkPipeline pipeline, const std::string& name);

    // "<debug name>@<stage>_<first 16 hex digits of the SPIR-V SHA-1>", no spaces.
    static std::string shaderName(const DxvkShader* shader);

    // A compute or ray tracing pipeline was created through `cache`.
    // `hit`: VK_PIPELINE_CREATION_FEEDBACK's application-cache hit (1 / 0), -1 without
    // feedback. Counted in the cache's summary; a compiled one (a miss taking at least kLoggedMissMs, or without
    // feedback any taking kSlowCreateMs) is logged at info by name.
    // Once the startup prewarm has finished, a compile is logged as one the prewarm missed, up to
    // DxvkPipelineCache::kLoggedLateCompiles per run; the rest are counted in the cache's summary.
    // `remix` marks the runtime's own pipelines, which the startup prewarm reports on.
    // `logBeforePrewarm` false keeps compiles before that point out of the log (the game's graphics pipelines).
    static constexpr double kLoggedMissMs = 20.0;
    static constexpr double kSlowCreateMs = 200.0;
    static void logCreation(DxvkPipelineCache* cache, const char* kind, const std::string& name, double milliseconds, int hit,
                            bool remix, bool logBeforePrewarm = true);

    // Marks the calling thread as a background compile thread (a state cache worker).
    // Pipelines created on any other thread are compiled on demand, and the frame that needs them waits.
    static void markBackgroundThread();
    static bool onBackgroundThread();

  private:
    static std::atomic<bool> s_captureEnabled;
  };
}
