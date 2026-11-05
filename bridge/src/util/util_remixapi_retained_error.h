#pragma once
#include <remix/remix_retained.h>

namespace remixapi::retained {

inline void latchError(const remixapi_RetainedInterface& retained, uint64_t epoch,
                       remixapi_ErrorCode error, remixapi_ErrorCode& stickyError) {
  if (error == REMIXAPI_ERROR_CODE_SUCCESS) {
    return;
  }
  const bool firstError = stickyError == REMIXAPI_ERROR_CODE_SUCCESS;
  stickyError = error;
  if (firstError && epoch && retained.Clear) {
    // Repeating the current epoch queues native fail-closed handling without deleting the catalog.
    // It runs after prior submissions and before subsequent draws, without waiting for the GPU.
    retained.Clear(epoch);
  }
}

}
