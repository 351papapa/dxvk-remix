#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace numos { struct DispatchDesc; }

namespace dxvk {
  class DxvkBuffer;
  class DxvkContext;
  class DxvkDevice;
  class SceneManager;
  template<typename T> class Rc;

  // One manually armed cloud frame. CPU identity records are copied before dispatch.
  class NumosCloudProbe {
  public:
    explicit NumosCloudProbe(DxvkDevice* device);
    ~NumosCloudProbe();
    bool enabled() const;
    void beginFrame(DxvkContext& ctx, uint64_t frame);
    void noteConstants(uint64_t handle, size_t offset, const void* data, size_t size);
    Rc<DxvkBuffer> beforeCloudDispatch(DxvkContext& ctx, const numos::DispatchDesc& desc,
      const Rc<DxvkBuffer>& argsBuffer, const Rc<DxvkBuffer>& cameraBuffer,
      const SceneManager& scene, uint32_t width, uint32_t height);
    void endFrame(DxvkContext& ctx);
  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
  };
}
