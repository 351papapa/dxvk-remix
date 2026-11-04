#pragma once

#include "../dxvk/dxvk_resource.h"

namespace dxvk {
  // A worker may outlive its draw, or be cancelled before it runs. Keep the
  // resource and its read access together until the worker is done with it.
  class D3D9WorkerRead {
  public:
    explicit D3D9WorkerRead(DxvkResource* pBuffer) : m_buffer(pBuffer) {
      if (m_buffer != nullptr) {
        m_buffer->acquire(DxvkAccess::Read);
      }
    }

    D3D9WorkerRead(D3D9WorkerRead&&) = default;

    ~D3D9WorkerRead() {
      if (m_buffer != nullptr) {
        m_buffer->release(DxvkAccess::Read);
      }
    }

  private:
    Rc<DxvkResource> m_buffer;
  };
}
