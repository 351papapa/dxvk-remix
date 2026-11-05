#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include "../dxvk_include.h"

namespace dxvk {
  class DxvkDevice;

  // Device-loss diagnostics for GPUs without Aftermath. All log lines start with [DeviceFault].
  // With VK_AMD_buffer_marker, every GPU profiler zone writes its sequence number into a host-visible buffer,
  // at zone begin with TOP_OF_PIPE and at zone end with BOTTOM_OF_PIPE. A CPU-side ring keeps each zone's name.
  // On device loss the buffer is read back to list the zones the GPU began but never finished.
  // With VK_EXT_device_fault, the driver's fault description and faulting addresses are logged as well.
  class GpuBreadcrumbs {
  public:
    // Ring size in zones. Must be a power of two.
    static constexpr uint32_t kSlots = 1u << 15;
    static constexpr uint32_t kNameLength = 48;
    static constexpr uint32_t kMaxDepth = 32;

    // Returns nullptr when the device has neither VK_AMD_buffer_marker nor VK_EXT_device_fault enabled.
    static std::unique_ptr<GpuBreadcrumbs> create(DxvkDevice* device);

    ~GpuBreadcrumbs();

    // Writes a zone begin marker into cmdBuffer and returns the sequence number for end().
    // Returns 0 when markers are unavailable.
    uint32_t begin(VkCommandBuffer cmdBuffer, const char* name);
    void end(VkCommandBuffer cmdBuffer, uint32_t seq);

    // Logs the device-loss report. Only the first call logs.
    void logDeviceLost(const char* reason);

  private:
    struct Entry {
      // Zero while the entry is empty or being written.
      std::atomic<uint32_t> seq { 0 };
      uint32_t frameId = 0;
      uint32_t parentSeq = 0;
      uint16_t depth = 0;
      char name[kNameLength] = {};
    };

    struct Snapshot {
      uint32_t seq;
      uint32_t frameId;
      uint32_t parentSeq;
      uint16_t depth;
      bool started;
      bool ended;
      char name[kNameLength];
    };

    explicit GpuBreadcrumbs(DxvkDevice* device);
    bool initMarkers();
    void logMarkers();
    void logDeviceFault();

    DxvkDevice*              m_device;
    Rc<vk::DeviceFn>         m_vkd;
    VkBuffer                 m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory           m_memory = VK_NULL_HANDLE;
    // Begin markers in [0, kSlots), end markers in [kSlots, 2 * kSlots).
    const volatile uint32_t* m_mapped = nullptr;
    std::atomic<uint32_t>    m_nextSeq { 1 };
    std::atomic<bool>        m_logged { false };
    std::unique_ptr<Entry[]> m_entries;
  };
}
