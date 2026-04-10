#include "dxvk_device.h"
#include "rtx_staging.h"

namespace dxvk {

  RtxStagingDataAlloc::RtxStagingDataAlloc(
    const Rc<DxvkDevice>& device,
    const char* name,
    const VkMemoryPropertyFlagBits memFlags,
    const VkBufferUsageFlags usageFlags,
    const VkPipelineStageFlags stages,
    const VkAccessFlags access,
    const VkDeviceSize bufferRequiredAlignmentOverride)
    : m_device(device) 
	  , m_memoryFlags(memFlags)
    , m_usage(usageFlags)
    , m_stages(stages)
    , m_access(access)
    , m_bufferRequiredAlignmentOverride(bufferRequiredAlignmentOverride)
    , m_name(name)
  {

  }


  RtxStagingDataAlloc::~RtxStagingDataAlloc() {

  }

  DxvkBufferSlice RtxStagingDataAlloc::alloc(VkDeviceSize align, VkDeviceSize size) {
    ScopedCpuProfileZone();

    if (size > MaxBufferSize)
      return DxvkBufferSlice(createBuffer(size));
    
    if (m_buffer == nullptr)
      m_buffer = createBuffer(MaxBufferSize);
    
    // Acceleration structure API accepts a VA, which DXVK doesnt recognize as "in use"
    if (!m_buffer->isInUse() && (m_usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) == 0)
      m_offset = 0;
    
    m_offset = dxvk::align(m_offset, align);

    if (m_offset + size > MaxBufferSize) {
      m_offset = 0;

      // Move the current buffer into the tracked pool so it is not lost
      m_buffers.push_back(std::move(m_buffer));

      // Scan all pooled buffers for one that is no longer in use and reuse it.
      // Acceleration structure buffers must never be reused because the AS build
      // API uses a raw VA that DXVK does not track as "in use", so isInUse() would
      // always return false even while the GPU is still reading the data.
      const bool isAccelStructBuffer = (m_usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) != 0;
      for (auto it = m_buffers.begin(); it != m_buffers.end(); ++it) {
        if (!(*it)->isInUse() && !isAccelStructBuffer) {
          m_buffer = std::move(*it);
          m_buffers.erase(it);
          break;
        }
      }

      if (m_buffer == nullptr) {
        if (m_buffers.size() < MaxTotalBuffers) {
          // Still below the hard cap: allocate a new buffer
          m_buffer = createBuffer(MaxBufferSize);
        } else {
          // At the hard cap: apply backpressure.  All buffers are still in use by the
          // GPU, so wait for the oldest one (front) to finish — it was submitted first
          // and is therefore most likely to complete soonest.  This will stall the CPU
          // briefly but prevents unbounded memory growth / OOM crashes.
          m_buffers.front()->waitIdle();
          m_buffer = std::move(m_buffers.front());
          m_buffers.erase(m_buffers.begin());
        }
      }
    }

    DxvkBufferSlice slice(m_buffer, m_offset, size);
    m_offset = dxvk::align(m_offset + size, align);
    return slice;
  }


  void RtxStagingDataAlloc::trim() {
    m_buffer = nullptr;
    m_offset = 0;
    m_buffers.clear();
  }

  Rc<DxvkBuffer> RtxStagingDataAlloc::createBuffer(VkDeviceSize size) {
    DxvkBufferCreateInfo info;
    info.size = size;
    info.access = m_access;
    info.stages = m_stages;
    info.usage = m_usage;
    info.requiredAlignmentOverride = m_bufferRequiredAlignmentOverride;

    return m_device->createBuffer(info, m_memoryFlags, DxvkMemoryStats::Category::AppBuffer, m_name);
  }
}
