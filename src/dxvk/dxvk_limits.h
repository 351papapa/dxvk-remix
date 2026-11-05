#pragma once

#include "dxvk_include.h"

namespace dxvk {
  
  enum DxvkLimits : size_t {
    MaxNumRenderTargets         =     8,
    MaxNumVertexAttributes      =    32,
    MaxNumVertexBindings        =    32,
    MaxNumXfbBuffers            =     4,
    MaxNumXfbStreams            =     4,
    MaxNumViewports             =    16,
    MaxNumResourceSlots         =  1216,
    MaxNumActiveBindings        =   384,
    MaxNumQueuedCommandBuffers  =    18,
    MaxNumQueryCountPerPool     =   128,
    MaxNumSpecConstants         =    14,
    MaxUniformBufferSize        = 65536,
    MaxVertexBindingStride      =  2048,
    // NV-DXVK start: increase push constant budget for the tonemap operator args.
    //
    // 256 bytes, since the tonemap apply args are larger than 128 (see the
    // static_assert in shaders/rtx/pass/tonemap/tonemapping.h). The Vulkan
    // minimum is 128, but every GPU Remix targets reports
    // maxPushConstantsSize >= 256. Release builds compile out the size assert
    // in dxvk_context.cpp, so an overflow of m_state.pc.data[bank] would be
    // silent. Only that memcpy and this storage size use the value; pipeline
    // layouts take push-constant ranges from per-shader reflection.
    MaxPushConstantSize         =   256,
    // NV-DXVK end
  };
  
}