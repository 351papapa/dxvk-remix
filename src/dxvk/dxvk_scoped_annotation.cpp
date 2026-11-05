/*
* Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "dxvk_scoped_annotation.h"
#include "dxvk_context.h"
#include "dxvk_device.h"
#include "rtx_render/rtx_gpu_breadcrumbs.h"
#include "client/TracyProfiler.hpp"

// Global overload
TRACY_OBJECT_MEMORY_PROFILING

namespace dxvk {

  namespace {
    // Returns the context's exec command buffer, or VK_NULL_HANDLE between command lists.
    VkCommandBuffer execCmdBuffer(const Rc<DxvkContext>& ctx) {
      const Rc<DxvkCommandList>& cmd = ctx->getCommandList();
      return cmd != nullptr ? cmd->getCmdBuffer(DxvkCmdBuffer::ExecBuffer) : VK_NULL_HANDLE;
    }
  }

  __ScopedAnnotation::__ScopedAnnotation(Rc<DxvkContext> ctx, const char* name) : m_ctx(ctx) {
    VkDebugUtilsLabelEXT info = {
      VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, name, { 1.0f, 0.0f, 0.0f, 1.0f } // Red color
    };
    m_ctx->beginDebugLabel(&info);
    // NV-DXVK start: Integrate Aftermath
    m_ctx->deviceDiagnosticCheckpoint(name);
    // NV-DXVK end

    if (GpuBreadcrumbs* breadcrumbs = m_ctx->getDevice()->gpuBreadcrumbs()) {
      m_breadcrumb = breadcrumbs->begin(execCmdBuffer(m_ctx), name);
    }
  }

  __ScopedAnnotation::~__ScopedAnnotation() {
    // The end marker goes into the command buffer the context records into now.
    if (m_breadcrumb != 0) {
      m_ctx->getDevice()->gpuBreadcrumbs()->end(execCmdBuffer(m_ctx), m_breadcrumb);
    }
    m_ctx->endDebugLabel();
  }

  __ScopedQueueAnnotation::__ScopedQueueAnnotation(DxvkDevice* dev, VkCommandBuffer cmdBuf, const char* name)
    : m_device(dev)
    , m_cmdBuf(cmdBuf) {
    VkDebugUtilsLabelEXT info = {
      VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, name, { 1.0f, 0.0f, 0.0f, 1.0f } // Red color
    };

    if (m_device->instance()->extensions().extDebugUtils) {
      m_device->vkd()->vkCmdBeginDebugUtilsLabelEXT(m_cmdBuf, &info);
    }

    if (m_device->extensions().nvDeviceDiagnosticCheckpoints) {
      m_device->vkd()->vkCmdSetCheckpointNV(m_cmdBuf, (const void*)name);
    }

    if (GpuBreadcrumbs* breadcrumbs = m_device->gpuBreadcrumbs()) {
      m_breadcrumb = breadcrumbs->begin(m_cmdBuf, name);
    }
  }

  __ScopedQueueAnnotation::~__ScopedQueueAnnotation() {
    if (m_breadcrumb != 0) {
      m_device->gpuBreadcrumbs()->end(m_cmdBuf, m_breadcrumb);
    }

    if (m_device->instance()->extensions().extDebugUtils) {
      m_device->vkd()->vkCmdEndDebugUtilsLabelEXT(m_cmdBuf);
    }
  }
}
