#pragma once

#include <atomic>

namespace dxvk {
  
  /**
   * \brief Reference-counted object
   */
  class RcObject {
    
  public:
    
    /**
     * \brief Increments reference count
     * \returns New reference count
     */
    uint32_t incRef() {
      return ++m_refCount;
    }
    
    /**
     * \brief Decrements reference count
     * \returns New reference count
     */
    uint32_t decRef() {
      return --m_refCount;
    }
    
    // Only meaningful when the caller owns a reference and controls how new
    // references can be acquired (e.g. the producer of a staging page).
    bool hasUniqueRef() const {
      return m_refCount.load(std::memory_order_acquire) == 1;
    }

  private:
    
    std::atomic<uint32_t> m_refCount = { 0u };
    
  };
  
}