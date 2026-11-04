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
    return allocImpl(align, size, MaxBufferSize, false);
  }

  DxvkBufferSlice RtxStagingDataAlloc::allocForDraw(VkDeviceSize align, VkDeviceSize size) {
    // A retained draw can keep a few indices alive for many frames. Limit the
    // rest of the RAM pinned by that slice. Larger draws get their own buffer.
    constexpr VkDeviceSize kDrawPageSize = 1 << 20;
    return allocImpl(align, size, kDrawPageSize, true);
  }

  DxvkBufferSlice RtxStagingDataAlloc::allocImpl(VkDeviceSize align, VkDeviceSize size, VkDeviceSize pageSize, bool requireUnique) {
    ScopedCpuProfileZone();

    const auto canReuse = [&](const Rc<DxvkBuffer>& page) {
      return (!requireUnique || page->hasUniqueRef()) && !page->isInUse() &&
             !(m_usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    };

    if (size > pageSize) {
      return DxvkBufferSlice(createBuffer(size));
    }
    
    if (m_buffer == nullptr) {
      m_buffer = createBuffer(pageSize);
    }
    
    // Acceleration structure API accepts a VA, which DXVK doesnt recognize as "in use"
    if (canReuse(m_buffer)) {
      m_offset = 0;
    }
    
    m_offset = dxvk::align(m_offset, align);

    if (m_offset + size > pageSize) {
      m_offset = 0;

      if (m_buffers.size() < MaxBufferCount) {
        m_buffers.push(std::move(m_buffer));
      }

      if (canReuse(m_buffers.front())) {
        m_buffer = std::move(m_buffers.front());
        m_buffers.pop();
      } else {
        m_buffer = createBuffer(pageSize);
      }
    }

    DxvkBufferSlice slice(m_buffer, m_offset, size);
    m_offset = dxvk::align(m_offset + size, align);
    return slice;
  }


  void RtxStagingDataAlloc::trim() {
    m_buffer = nullptr;
    m_offset = 0;

    while (!m_buffers.empty())
      m_buffers.pop();
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
