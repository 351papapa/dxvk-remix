#pragma once

// Light BVH select pass entry point. Separate from rtx_fork_light_bvh.h so that header stays free of
// rtx_resources.h.

#include "rtx_resources.h"

namespace dxvk {

  class RtxContext;

  // Dispatches light_bvh_select.comp.slang when LightBVH is the effective mode and
  // rtx.lightSampling.bvhSeparateSelectPass is on. Writes one RTXDI-format reservoir per primary pixel
  // (spatial output page) for the direct integrator and indirect sample stealing. Returns true if it ran.
  // Implementation in rtx_fork_light_bvh.cpp.
  bool dispatchLightBvhSelectPass(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

  // The [LightSamplingStats] counter buffer (always valid, bound in the common ray tracing set) and
  // the per-frame readback/log step, called right after the direct integration dispatch.
  Rc<DxvkBuffer> getLightSamplingStatsBuffer(DxvkDevice* device);
  void lightSamplingStatsAfterDirect(RtxContext* ctx);

} // namespace dxvk
