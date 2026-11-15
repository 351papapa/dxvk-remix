#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace numos { struct DispatchDesc; }

namespace dxvk {
  class DxvkBuffer;
  class DxvkContext;
  class DxvkDevice;
  template<typename T> class Rc;

  // Bounded, opt-in readback owned by NumosGpu. No context survives a method call.
  class NumosHazeProbe {
  public:
    explicit NumosHazeProbe(DxvkDevice* device);
    ~NumosHazeProbe();

    void beginFrame(DxvkContext& ctx, uint64_t frameId);
    void trackBuffer(uint64_t handle, const char* debugName);
    void forgetBuffer(uint64_t handle);
    void noteConstants(uint64_t handle, size_t offset, const void* data, size_t size);
    void noteBuffer(uint64_t handle, size_t offset, const void* data, size_t size);

    // Call immediately after the AP cull dispatch. Buffers correspond to slots 0, 2, 3.
    void afterCull(DxvkContext& ctx, const numos::DispatchDesc& desc,
                   const Rc<DxvkBuffer>& args, const Rc<DxvkBuffer>& lights,
                   const Rc<DxvkBuffer>& clusters);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
  };
}
