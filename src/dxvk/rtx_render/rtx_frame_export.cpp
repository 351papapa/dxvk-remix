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
#include "rtx_frame_export.h"

#include <cstring>

#include "../dxvk_device.h"

namespace dxvk {

  namespace {

    constexpr VkFormat kExportFormat = VK_FORMAT_R8G8B8A8_UNORM;

    // Returns whether a blit copies the format's values into R8G8B8A8_UNORM unchanged apart from their precision.
    // sRGB formats are left out because a blit decodes them, and float formats because their values are not
    // display-encoded.
    bool isUnormColorFormat(VkFormat format) {
      switch (format) {
      case VK_FORMAT_B8G8R8A8_UNORM:
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
      case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      case VK_FORMAT_R5G6B5_UNORM_PACK16:
      case VK_FORMAT_B5G6R5_UNORM_PACK16:
      case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
      case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
        return true;
      default:
        return false;
      }
    }

    HANDLE duplicateHandle(HANDLE handle) {
      HANDLE duplicate = nullptr;
      if (!DuplicateHandle(GetCurrentProcess(), handle, GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
        return nullptr;
      }
      return duplicate;
    }

    void closeHandle(HANDLE handle) {
      if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
      }
    }

    VkImageSubresourceRange colorRange() {
      return VkImageSubresourceRange { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    }

  }

  RtxFrameExport::RtxFrameExport(DxvkDevice* device)
    : CommonDeviceObject(device) {
  }

  RtxFrameExport::~RtxFrameExport() = default;

  void RtxFrameExport::onDestroy() {
    m_target = nullptr;
  }

  const char* RtxFrameExport::describe(FrameExportReason reason) {
    switch (reason) {
    case FrameExportReason::None: return "no reason";
    case FrameExportReason::NoWin32ExternalExtensions: return "the device has no VK_KHR_external_memory_win32 or VK_KHR_external_semaphore_win32";
    case FrameExportReason::NoBackBuffer: return "there is no back buffer";
    case FrameExportReason::ExternalSwapchain: return "the device presents into an external swapchain";
    case FrameExportReason::MultisampledBackBuffer: return "the back buffer is multisampled";
    case FrameExportReason::BackBufferNotCopySource: return "the back buffer cannot be a copy source";
    case FrameExportReason::BackBufferFormatNotUnorm: return "the back buffer format has no display-encoded UNORM copy";
    case FrameExportReason::BackBufferFormatNotBlitSource: return "the back buffer format cannot be a blit source";
    case FrameExportReason::CreationFailed: return "creating the shared image or semaphores failed";
    case FrameExportReason::MemoryNotExportable: return "the image memory cannot be exported";
    case FrameExportReason::NoDedicatedAllocation: return "the image did not get a dedicated allocation";
    case FrameExportReason::SemaphoresNotExportable: return "the semaphores cannot be exported";
    case FrameExportReason::HandleDuplicationFailed: return "duplicating the semaphore handles failed";
    }
    return "an unknown reason";
  }

  void RtxFrameExport::setStatus(FrameExportState state, FrameExportReason reason, uint32_t width, uint32_t height) {
    const uint64_t packed = uint64_t(state)
                          | uint64_t(reason) << 8
                          | uint64_t(width & 0xFFFFFFu) << 16
                          | uint64_t(height & 0xFFFFFFu) << 40;
    m_status.store(packed, std::memory_order_relaxed);
  }

  FrameExportStatus RtxFrameExport::status() const {
    const uint64_t packed = m_status.load(std::memory_order_relaxed);
    FrameExportStatus result;
    result.state = FrameExportState(packed & 0xFFu);
    result.reason = FrameExportReason((packed >> 8) & 0xFFu);
    result.width = uint32_t((packed >> 16) & 0xFFFFFFu);
    result.height = uint32_t((packed >> 40) & 0xFFFFFFu);
    return result;
  }

  FrameExportReason RtxFrameExport::checkSource(const Rc<DxvkImage>& backBuffer) {
    const DxvkImageCreateInfo& info = backBuffer->info();
    if (backBuffer.ptr() == m_checkedImage && info.format == m_checkedFormat && info.usage == m_checkedUsage && info.sampleCount == m_checkedSamples) {
      return m_checkedReason;
    }

    const VkFormatProperties properties = m_device->adapter()->formatProperties(info.format);
    FrameExportReason reason = FrameExportReason::None;
    if (info.sampleCount != VK_SAMPLE_COUNT_1_BIT) {
      reason = FrameExportReason::MultisampledBackBuffer;
    } else if ((info.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
      reason = FrameExportReason::BackBufferNotCopySource;
    } else if (!isUnormColorFormat(info.format)) {
      reason = FrameExportReason::BackBufferFormatNotUnorm;
    } else if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) == 0) {
      reason = FrameExportReason::BackBufferFormatNotBlitSource;
    }

    m_checkedImage = backBuffer.ptr();
    m_checkedFormat = info.format;
    m_checkedUsage = info.usage;
    m_checkedSamples = info.sampleCount;
    m_checkedReason = reason;
    m_checkedCanFilter = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    if (reason != FrameExportReason::None) {
      Logger::info(str::format("External frame: the back buffer format is ", info.format, "."));
    }
    return reason;
  }

  void RtxFrameExport::disable(FrameExportReason reason) {
    m_target = nullptr;
    setStatus(FrameExportState::Unavailable, reason, 0, 0);
    Logger::warn(str::format("External frame: the export is off because ", describe(reason), "."));
  }

  void RtxFrameExport::stop() {
    if (m_target != nullptr) {
      Logger::info("External frame: the export is off.");
    }
    m_target = nullptr;
    setStatus(FrameExportState::Off, FrameExportReason::None, 0, 0);
  }

  FrameExportResult RtxFrameExport::configure(const Rc<DxvkImage>& backBuffer, FrameExportHandles& handles) {
    if (!m_device->extensions().khrExternalMemoryWin32 || !m_device->extensions().khrExternalSemaphoreWin32) {
      disable(FrameExportReason::NoWin32ExternalExtensions);
      return FrameExportResult::Unavailable;
    }
    if (backBuffer == nullptr) {
      disable(FrameExportReason::NoBackBuffer);
      return FrameExportResult::Failure;
    }
    const FrameExportReason sourceProblem = checkSource(backBuffer);
    if (sourceProblem != FrameExportReason::None) {
      disable(sourceProblem);
      return FrameExportResult::Unavailable;
    }

    const VkExtent3D extent = backBuffer->info().extent;

    DxvkImageCreateInfo info;
    info.type = VK_IMAGE_TYPE_2D;
    info.format = kExportFormat;
    info.flags = 0;
    info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    info.extent = { extent.width, extent.height, 1 };
    info.numLayers = 1;
    info.mipLevels = 1;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.layout = VK_IMAGE_LAYOUT_GENERAL;
    info.sharing.mode = DxvkSharedHandleMode::Export;
    info.sharing.type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    Rc<Target> target = new Target();
    HANDLE memoryHandle = nullptr;
    HANDLE readyHandle = nullptr;
    HANDLE releasedHandle = nullptr;

    const auto fail = [&](FrameExportReason reason) {
      closeHandle(memoryHandle);
      closeHandle(readyHandle);
      closeHandle(releasedHandle);
      disable(reason);
      return FrameExportResult::Failure;
    };

    try {
      target->image = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXRenderTarget, "External frame");
      target->ready = RtxSemaphore::createBinary(m_device, "External frame ready", true);
      target->released = RtxSemaphore::createBinary(m_device, "External frame released", true);
    } catch (const DxvkError& e) {
      Logger::err(str::format("External frame: ", e.message()));
      return fail(FrameExportReason::CreationFailed);
    }

    memoryHandle = target->image->sharedHandle();
    if (memoryHandle == nullptr || memoryHandle == INVALID_HANDLE_VALUE) {
      memoryHandle = nullptr;
      return fail(FrameExportReason::MemoryNotExportable);
    }

    // An OPAQUE_WIN32 import needs the whole allocation, so the image must own its memory.
    if (target->image->getMemoryOffset() != 0) {
      return fail(FrameExportReason::NoDedicatedAllocation);
    }

    if (target->ready->sharedHandle() == INVALID_HANDLE_VALUE || target->released->sharedHandle() == INVALID_HANDLE_VALUE) {
      return fail(FrameExportReason::SemaphoresNotExportable);
    }
    readyHandle = duplicateHandle(target->ready->sharedHandle());
    releasedHandle = duplicateHandle(target->released->sharedHandle());
    if (readyHandle == nullptr || releasedHandle == nullptr) {
      return fail(FrameExportReason::HandleDuplicationFailed);
    }

    handles.width = extent.width;
    handles.height = extent.height;
    handles.format = kExportFormat;
    handles.memoryTypeIndex = target->image->getMemoryTypeIndex();
    handles.allocationSize = target->image->memSize();
    std::memcpy(handles.deviceUUID, m_device->adapter()->devicePropertiesExt().coreDeviceId.deviceUUID, sizeof(handles.deviceUUID));
    handles.memory = memoryHandle;
    handles.ready = readyHandle;
    handles.released = releasedHandle;

    // Command lists that used the previous target keep it alive until they complete.
    m_target = target;
    setStatus(FrameExportState::Exporting, FrameExportReason::None, extent.width, extent.height);
    Logger::info(str::format("External frame: exporting ", extent.width, "x", extent.height, "."));
    return FrameExportResult::Success;
  }

  Rc<DxvkCommandList> RtxFrameExport::recordCopy(const Rc<DxvkImage>& backBuffer) {
    if (m_target == nullptr) {
      return nullptr;
    }
    const FrameExportReason sourceProblem = checkSource(backBuffer);
    if (sourceProblem != FrameExportReason::None) {
      disable(sourceProblem);
      return nullptr;
    }

    Target& target = *m_target;
    const DxvkImageCreateInfo& srcInfo = backBuffer->info();
    const VkExtent3D dstExtent = target.image->info().extent;
    const VkImageLayout srcLayout = srcInfo.layout == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const uint32_t queueFamily = m_device->queues().graphics.queueFamily;

    Rc<DxvkCommandList> cmd = m_device->createCommandList();
    cmd->beginRecording();

    VkImageMemoryBarrier acquire[2] = {};
    acquire[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    acquire[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    acquire[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    acquire[0].oldLayout = srcInfo.layout;
    acquire[0].newLayout = srcLayout;
    acquire[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    acquire[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    acquire[0].image = backBuffer->handle();
    acquire[0].subresourceRange = colorRange();

    // The first copy discards the image's contents.
    // Every later copy takes the image back from the consumer, which released it in GENERAL.
    acquire[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    acquire[1].srcAccessMask = 0;
    acquire[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    acquire[1].oldLayout = target.written ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    acquire[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    acquire[1].srcQueueFamilyIndex = target.written ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED;
    acquire[1].dstQueueFamilyIndex = target.written ? queueFamily : VK_QUEUE_FAMILY_IGNORED;
    acquire[1].image = target.image->handle();
    acquire[1].subresourceRange = colorRange();

    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      0, nullptr, 0, nullptr, 2, acquire);

    VkImageBlit region = {};
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.srcOffsets[1] = { int32_t(srcInfo.extent.width), int32_t(srcInfo.extent.height), 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstOffsets[1] = { int32_t(dstExtent.width), int32_t(dstExtent.height), 1 };

    // The back buffer is scaled to the image until the host configures the export again for the new size.
    const bool sameSize = srcInfo.extent.width == dstExtent.width && srcInfo.extent.height == dstExtent.height;
    const VkFilter filter = !sameSize && m_checkedCanFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

    cmd->cmdBlitImage(backBuffer->handle(), srcLayout, target.image->handle(), VK_IMAGE_LAYOUT_GENERAL, 1, &region, filter);

    VkImageMemoryBarrier release[2] = {};
    release[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    release[0].srcAccessMask = 0;
    release[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    release[0].oldLayout = srcLayout;
    release[0].newLayout = srcInfo.layout;
    release[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    release[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    release[0].image = backBuffer->handle();
    release[0].subresourceRange = colorRange();

    release[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    release[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    release[1].dstAccessMask = 0;
    release[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    release[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    release[1].srcQueueFamilyIndex = queueFamily;
    release[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    release[1].image = target.image->handle();
    release[1].subresourceRange = colorRange();

    cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
      0, nullptr, 0, nullptr, 2, release);

    if (target.written) {
      cmd->addWaitSemaphore(target.released->handle());
    }
    cmd->addSignalSemaphore(target.ready->handle());

    cmd->trackResource<DxvkAccess::Read>(backBuffer);
    cmd->trackResource<DxvkAccess::Write>(target.image);
    cmd->trackResource<DxvkAccess::None>(m_target);
    cmd->endRecording();

    target.written = true;
    m_submitStatus.result = VK_NOT_READY;
    m_copyQueued = true;
    return cmd;
  }

  void RtxFrameExport::waitForSubmission() {
    if (m_copyQueued) {
      m_device->waitForSubmission(&m_submitStatus);
      m_copyQueued = false;
    }
  }

}
