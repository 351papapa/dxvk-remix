#pragma once
#include <remix/remix_retained.h>

namespace remixapi::retained {

class InterfaceCache {
public:
  template<typename Fetch>
  remixapi_ErrorCode get(const void* pModule, uint32_t version, Fetch&& fetch, remixapi_RetainedInterface& out) {
    out = {};
    selectModule(pModule);
    if (version != REMIXAPI_RETAINED_VERSION_2 && version != REMIXAPI_RETAINED_VERSION) {
      return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
    }

    auto& slot = version == REMIXAPI_RETAINED_VERSION_2 ? m_v2 : m_v3;
    if (!slot.BatchUpsert) {
      // Do not cache partial output from a failed lookup. The next command can retry this ABI.
      remixapi_RetainedInterface loaded {};
      const auto result = fetch(version, loaded);
      if (result != REMIXAPI_ERROR_CODE_SUCCESS) {
        return result;
      }
      if (loaded.version != version || loaded.structSize != sizeof(loaded) || !loaded.BatchUpsert ||
          !loaded.BatchRemove || !loaded.Clear || !loaded.CommitFrame || !loaded.GetStatus || !loaded.DrawOrdered) {
        return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
      }
      slot = loaded;
    }
    out = slot;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  template<typename Fetch>
  remixapi_ErrorCode getVisibility(const void* pModule, Fetch&& fetch, remixapi_RetainedVisibilityInterface& out) {
    selectModule(pModule);
    out = {};
    if (!m_visibility.BatchSetVisibility) {
      remixapi_RetainedVisibilityInterface loaded {};
      const auto result = fetch(loaded);
      if (result != REMIXAPI_ERROR_CODE_SUCCESS) {
        return result;
      }
      if (loaded.version != REMIXAPI_RETAINED_VISIBILITY_VERSION || loaded.structSize != sizeof(loaded) || !loaded.BatchSetVisibility) {
        return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
      }
      m_visibility = loaded;
    }
    out = m_visibility;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // Clear(epoch) has the same contract for both ABIs. A failed lookup must not supply this callback.
  const remixapi_RetainedInterface& errorInterface() const { return m_v2.Clear ? m_v2 : m_v3; }

private:
  void selectModule(const void* pModule) {
    if (m_pModule != pModule) {
      m_pModule = pModule;
      m_v2 = {};
      m_v3 = {};
      m_visibility = {};
    }
  }
  const void* m_pModule = nullptr;
  remixapi_RetainedInterface m_v2 {}, m_v3 {};
  remixapi_RetainedVisibilityInterface m_visibility {};
};

}
