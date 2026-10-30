#define REMIX_ALLOW_X86
#include "util_remixapi_retained_error.h"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

static unsigned checks = 0;
static uint64_t nativeEpoch = 7;
static unsigned catalogCount = 4;
static bool visible = false;
static bool nativeError = false;
static unsigned clearCalls = 0;
static std::vector<std::function<void()>> queue;
static void check(bool result) {
  ++checks;
  if (!result) { std::printf("Retained failure check %u failed\n", checks); std::exit(1); }
}
static remixapi_ErrorCode REMIXAPI_CALL clear(uint64_t epoch) {
  ++clearCalls;
  queue.push_back([epoch] {
    if (!epoch || epoch <= nativeEpoch) { nativeError = true; visible = false; }
    else { nativeEpoch = epoch; nativeError = false; visible = false; catalogCount = 0; }
  });
  return REMIXAPI_ERROR_CODE_SUCCESS;
}
static void drain() { for (auto& command : queue) { command(); } queue.clear(); }
int main() {
  remixapi_RetainedInterface api {};
  api.Clear = clear;
  auto error = REMIXAPI_ERROR_CODE_SUCCESS;
  remixapi::retained::latchError(api, nativeEpoch, REMIXAPI_ERROR_CODE_SUCCESS, error);
  check(clearCalls == 0 && error == REMIXAPI_ERROR_CODE_SUCCESS);
  queue.push_back([] { visible = true; }); // A valid Commit/DrawOrdered precedes the rejected bridge command.
  remixapi::retained::latchError(api, nativeEpoch, REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS, error);
  check(clearCalls == 1 && queue.size() == 2);
  check(error == REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS);
  drain();
  check(!visible && nativeError);
  check(nativeEpoch == 7 && catalogCount == 4);
  remixapi::retained::latchError(api, nativeEpoch, REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS, error);
  check(clearCalls == 1 && queue.empty());
  api.Clear(8); error = REMIXAPI_ERROR_CODE_SUCCESS; drain();
  check(nativeEpoch == 8 && !nativeError && catalogCount == 0);
  catalogCount = 2; queue.push_back([] { visible = true; });
  remixapi::retained::latchError(api, 8, REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS, error);
  drain();
  check(!visible && nativeError && catalogCount == 2 && nativeEpoch == 8);
  check(clearCalls == 3);
  error = REMIXAPI_ERROR_CODE_SUCCESS;
  remixapi::retained::latchError(api, 0, REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS, error);
  check(clearCalls == 3 && error == REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS);
  api.Clear = nullptr; error = REMIXAPI_ERROR_CODE_SUCCESS;
  remixapi::retained::latchError(api, 8, REMIXAPI_ERROR_CODE_GENERAL_FAILURE, error);
  check(clearCalls == 3 && error == REMIXAPI_ERROR_CODE_GENERAL_FAILURE);
  std::printf("Retained error: %u ordered failure/recovery checks passed\n", checks);
  return 0;
}
