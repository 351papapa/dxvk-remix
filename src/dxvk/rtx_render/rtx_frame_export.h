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

#include "../dxvk_cmdlist.h"
#include "../dxvk_image.h"
#include "../dxvk_queue.h"
#include "rtx_common_object.h"
#include "rtx_semaphore.h"

namespace dxvk {

  // The shared image and semaphores of an export, as new NT handles owned by the caller.
  struct FrameExportHandles {
    uint32_t     width = 0;
    uint32_t     height = 0;
    VkFormat     format = VK_FORMAT_UNDEFINED;
    uint32_t     memoryTypeIndex = 0;
    VkDeviceSize allocationSize = 0;
    uint8_t      deviceUUID[VK_UUID_SIZE] = {};
    HANDLE       memory = nullptr;
    HANDLE       ready = nullptr;
    HANDLE       released = nullptr;
  };

  enum class FrameExportResult {
    Success,
    // The device or the back buffer cannot provide the export.
    Unavailable,
    Failure,
  };

  enum class FrameExportState : uint8_t {
    Off,
    Exporting,
    Unavailable,
  };

  // Why the export is unavailable.
  enum class FrameExportReason : uint8_t {
    None,
    NoWin32ExternalExtensions,
    NoBackBuffer,
    ExternalSwapchain,
    MultisampledBackBuffer,
    BackBufferNotCopySource,
    BackBufferFormatNotUnorm,
    BackBufferFormatNotBlitSource,
    CreationFailed,
    MemoryNotExportable,
    NoDedicatedAllocation,
    SemaphoresNotExportable,
    HandleDuplicationFailed,
  };

  // The export's state as one value, so the developer menu reads a consistent copy from any thread.
  struct FrameExportStatus {
    FrameExportState  state = FrameExportState::Off;
    FrameExportReason reason = FrameExportReason::None;
    uint32_t          width = 0;
    uint32_t          height = 0;
  };

  // Copies the presented frame into an image that another Vulkan device or process imports, with a pair of binary
  // semaphores that hand the image back and forth (remixapi_ConfigureExternalFrame, docs/RemixApi.md "External frame").
  // The scene output's swapchain calls it on the thread that presents, under the device lock.
  class RtxFrameExport : public CommonDeviceObject {
  public:
    explicit RtxFrameExport(DxvkDevice* device);
    ~RtxFrameExport() override;

    // Replaces the shared image and semaphores with new ones sized for the back buffer.
    FrameExportResult configure(const Rc<DxvkImage>& backBuffer, FrameExportHandles& handles);

    void stop();

    // Stops the export and states the reason in the log and the developer menu.
    void disable(FrameExportReason reason);

    // Returns the state. Any thread may call it.
    FrameExportStatus status() const;

    // Returns the reason as a clause, such as "the back buffer is multisampled".
    static const char* describe(FrameExportReason reason);

    bool isActive() const {
      return m_target != nullptr;
    }

    // Records the copy of the back buffer into the shared image, with the semaphores attached.
    // Returns nullptr when the export is off or has just stopped because the back buffer cannot be copied.
    // The caller submits the command list with submitCommandListWithStatus and submitStatus().
    Rc<DxvkCommandList> recordCopy(const Rc<DxvkImage>& backBuffer);

    DxvkSubmitStatus* submitStatus() {
      return &m_submitStatus;
    }

    // Waits until the submission thread has submitted the last recorded copy to the Vulkan queue, so a consumer that
    // waits on the ready semaphore after Present returns always finds its signal operation pending.
    // It does not wait for the GPU.
    void waitForSubmission();

    void onDestroy() override;

  private:
    // The shared image and its semaphores. Command lists that use them track it, so it lives until they complete.
    class Target : public DxvkResource {
    public:
      Rc<DxvkImage>    image;
      Rc<RtxSemaphore> ready;
      Rc<RtxSemaphore> released;
      // A ready was signaled, so the next copy waits on released and acquires the image from the consumer.
      bool             written = false;
    };

    // Returns FrameExportReason::None when the back buffer can be copied into an R8G8B8A8_UNORM image.
    // Returns the reason otherwise.
    // The result is kept for the last back buffer image, so a check per Present is a compare.
    FrameExportReason checkSource(const Rc<DxvkImage>& backBuffer);

    void setStatus(FrameExportState state, FrameExportReason reason, uint32_t width, uint32_t height);

    Rc<Target>            m_target;
    // FrameExportStatus packed as state | reason << 8 | width << 16 | height << 40.
    std::atomic<uint64_t> m_status { 0 };
    DxvkSubmitStatus      m_submitStatus;
    bool                  m_copyQueued = false;

    // The last back buffer image checkSource looked at, and its results.
    const DxvkImage*      m_checkedImage = nullptr;
    VkFormat              m_checkedFormat = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags     m_checkedUsage = 0;
    VkSampleCountFlagBits m_checkedSamples = VK_SAMPLE_COUNT_1_BIT;
    FrameExportReason     m_checkedReason = FrameExportReason::None;
    bool                  m_checkedCanFilter = false;
  };

}
