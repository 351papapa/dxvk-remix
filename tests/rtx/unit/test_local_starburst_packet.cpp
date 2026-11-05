#define REMIX_ALLOW_X86
#include <remix/remix_local_starbursts.h>
#include <cstdio>
#include <limits>

int main() {
  using remixapi::starbursts::valid;
  constexpr uint32_t stride = sizeof(remixapi_LocalStarburstSource);
  std::array<remixapi_LocalStarburstSource, 256> sources {};
  for (uint32_t i = 0; i < sources.size(); ++i) {
    auto& s = sources[i];
    s.sourceId = i + 1;
    s.aspect = 1;
    s.opacity = 0.5f;
    s.halfScreenHeight = 0.66f;
    s.color[0] = 1;
    s.kind = i % 4;
  }
  auto check = [](bool condition, const char* label) {
    if (!condition) {
      std::fprintf(stderr, "Failed: %s\n", label);
    }
    return condition;
  };
  bool ok = check(valid(1, stride, 0, nullptr), "empty clears");
  ok &= check(valid(1, stride, 256, sources.data()), "full batch");
  ok &= check(!valid(2, stride, 0, nullptr) && !valid(1, 47, 1, sources.data()), "version or layout");
  ok &= check(!valid(1, stride, 1, nullptr) && !valid(1, stride, 257, sources.data()), "bounds before access");
  auto& s = sources.back();
  auto bad = [&]() { return !valid(1, stride, 256, sources.data()); };
  s.sourceId = sources.front().sourceId;
  ok &= check(bad(), "duplicate last ID");
  s.sourceId = 0;
  ok &= check(bad(), "missing ID");
  s.sourceId = UINT64_MAX;
  ok &= check(!bad(), "64-bit ID");
  s.kind = 4;
  ok &= check(bad(), "kind");
  s.kind = 3;
  s.aspect = 0;
  ok &= check(bad(), "aspect");
  s.aspect = 1;
  s.halfScreenHeight = -1;
  ok &= check(bad(), "size");
  s.halfScreenHeight = 1;
  s.position[2] = std::numeric_limits<float>::infinity();
  ok &= check(bad(), "infinite position");
  s.position[2] = 0;
  s.color[1] = std::numeric_limits<float>::quiet_NaN();
  ok &= check(bad(), "non-finite tint");
  s.color[1] = 1;
  s.opacity = -0.1f;
  ok &= check(bad(), "negative opacity");
  s.opacity = 0;
  ok &= check(!bad(), "fully faded source");
  return ok ? 0 : 1;
}
