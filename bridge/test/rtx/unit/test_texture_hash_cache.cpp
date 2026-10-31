#include "texture_hash_cache.h"

#include <cstdint>
#include <cstdio>

namespace {

  int expect(bool condition, const char* message) {
    if (condition) {
      return 0;
    }
    std::fprintf(stderr, "FAIL: %s\n", message);
    return -1;
  }

} // namespace

int main() {
  constexpr uint64_t kHash = 0x123456789abcdef0ull;
  TextureHashCache first;
  TextureHashCache second;
  uint64_t hash = 0;
  uint64_t version = 0;

  if (expect(!first.tryGet(&hash, &version), "new cache must miss"))
    return -1;
  if (expect(version == first.version(), "miss must return the version to query"))
    return -1;

  first.store(kHash, version);
  if (expect(first.tryGet(&hash, &version) && hash == kHash, "stored hash must hit"))
    return -1;

  second.invalidate();
  if (expect(first.tryGet(&hash, &version) && hash == kHash, "unrelated texture invalidation must preserve hash"))
    return -1;

  const uint64_t staleVersion = version;
  first.invalidate();
  if (expect(!first.tryGet(&hash, &version), "changed texture must miss"))
    return -1;
  if (expect(version == first.version() && version != staleVersion, "changed texture must return a new version"))
    return -1;

  first.store(kHash + 1, staleVersion);
  if (expect(!first.tryGet(&hash, &version), "a result from before an upload must not enter the cache"))
    return -1;

  first.store(0, version);
  if (expect(!first.tryGet(&hash, &version), "zero hash must not enter the cache"))
    return -1;

  first.store(kHash, version);
  if (expect(first.tryGet(&hash, &version) && hash == kHash, "current hash must hit after invalidation"))
    return -1;

  std::puts("texture hash cache tests passed");
  return 0;
}
