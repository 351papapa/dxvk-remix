#define REMIX_ALLOW_X86
#include "util_remixapi_retained_error.h"
#include "util_remixapi_retained_interface_cache.h"
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
static remixapi_ErrorCode REMIXAPI_CALL upsert(uint64_t, uint32_t, const remixapi_RetainedInstance*) { return REMIXAPI_ERROR_CODE_SUCCESS; }
static remixapi_ErrorCode REMIXAPI_CALL ids(uint64_t, uint32_t, const uint64_t*) { return REMIXAPI_ERROR_CODE_SUCCESS; }
static remixapi_ErrorCode REMIXAPI_CALL status(remixapi_RetainedStatus*) { return REMIXAPI_ERROR_CODE_SUCCESS; }
static void testInterfaceCache() {
  remixapi::retained::InterfaceCache cache;
  remixapi_RetainedInterface table {};
  int moduleA = 0, moduleB = 0;
  unsigned fetches = 0;
  enum class Reply { Valid, Failed, WrongVersion, Incomplete };
  Reply reply = Reply::Valid;
  auto fetch = [&](uint32_t version, remixapi_RetainedInterface& out) {
    ++fetches;
    out = {version, sizeof(out), upsert, ids, clear, clear, status, ids};
    if (reply == Reply::Failed) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    if (reply == Reply::WrongVersion) {
      out.version = 99;
    }
    if (reply == Reply::Incomplete) {
      out.DrawOrdered = nullptr;
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  };
  auto get = [&](const void* module, uint32_t version) { return cache.get(module, version, fetch, table); };
  const uint32_t v2 = REMIXAPI_RETAINED_VERSION_2, v3 = REMIXAPI_RETAINED_VERSION;
  check(get(&moduleA, v3) == REMIXAPI_ERROR_CODE_SUCCESS && table.version == v3 && fetches == 1);
  check(get(&moduleA, v2) == REMIXAPI_ERROR_CODE_SUCCESS && table.version == v2 && fetches == 2);
  check(get(&moduleA, v3) == REMIXAPI_ERROR_CODE_SUCCESS && table.version == v3 && fetches == 2);
  check(get(&moduleA, v2) == REMIXAPI_ERROR_CODE_SUCCESS && table.version == v2 && fetches == 2);
  reply = Reply::Failed;
  check(get(&moduleB, v2) == REMIXAPI_ERROR_CODE_GENERAL_FAILURE && !table.Clear && !cache.errorInterface().Clear && fetches == 3);
  reply = Reply::Valid;
  check(get(&moduleB, v2) == REMIXAPI_ERROR_CODE_SUCCESS && table.version == v2 && fetches == 4);
  reply = Reply::WrongVersion;
  check(get(&moduleB, v3) == REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION && !table.BatchUpsert && fetches == 5);
  check(cache.errorInterface().version == v2 && cache.errorInterface().Clear == clear);
  reply = Reply::Incomplete;
  check(get(&moduleB, v3) == REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION && !table.DrawOrdered && fetches == 6);
  check(get(&moduleB, v2) == REMIXAPI_ERROR_CODE_SUCCESS && fetches == 6);
  reply = Reply::Valid;
  check(get(&moduleB, v3) == REMIXAPI_ERROR_CODE_SUCCESS && table.version == v3 && fetches == 7);
}
int main() {
  testInterfaceCache();
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
