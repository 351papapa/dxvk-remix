#pragma once
#include <remix/remix_retained.h>

namespace remixapi::retained {

class InterfaceCache {
public:
  template<typename Fetch>
  remixapi_ErrorCode get(const void* pModule, uint32_t version, Fetch&& fetch, remixapi_RetainedInterface& out) {
    out = {};
    if (m_pModule != pModule) {
      m_pModule = pModule;
      m_v2 = {};
      m_v3 = {};
    }
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

  // Clear(epoch) has the same contract for both ABIs. A failed lookup must not supply this callback.
  const remixapi_RetainedInterface& errorInterface() const { return m_v2.Clear ? m_v2 : m_v3; }

private:
  const void* m_pModule = nullptr;
  remixapi_RetainedInterface m_v2 {}, m_v3 {};
};

}
