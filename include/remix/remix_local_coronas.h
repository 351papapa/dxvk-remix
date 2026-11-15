#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Existing SetGameValue/GetGameValue transport; no new bridge command ABI.
// LC01 + eight hex count digits + little-endian IEEE754 record bytes in hex.
// An accepted snapshot replaces this producer's complete current-frame batch.
namespace remixapi { namespace local_coronas {
constexpr char submitKey[] = "__remix.localCoronas.submit_v1";
constexpr char capabilityKey[] = "__remix.localCoronas.runtimeVersion";
constexpr char submitKeyV2[] = "__remix.localCoronas.submit_v2";
constexpr char capabilityKeyV2[] = "__remix.localCoronas.runtimeVersion2";
constexpr char statusKey[] = "__remix.localCoronas.status_v2";
constexpr uint32_t legacyLimit = 512;
constexpr uint32_t limit = 4096;
enum class State : uint32_t { Idle, Accepted, Dispatched, Suppressed, Stale, Late, Duplicate, Unavailable };
// Suppression owns the batch even after composition. A producer may repeat its
// draw callback, so already owned frames must never request a second host draw.
inline State submissionState(bool suppressed, bool alreadyOwned, bool afterComposition, bool available = true) {
  return suppressed ? State::Suppressed : alreadyOwned ? State::Duplicate : afterComposition ? State::Late : available ? State::Accepted : State::Unavailable;
}
inline const char* stateName(State state) {
  switch (state) {
  case State::Accepted: return "accepted / awaiting dispatch";
  case State::Dispatched: return "GPU dispatch recorded (not completion)";
  case State::Suppressed: return "hidden by runtime filter/PostFX switch";
  case State::Stale: return "discarded: frame or scene changed";
  case State::Late: return "rejected: frame already composited";
  case State::Unavailable: return "rejected: RTX or shaders unavailable";
  case State::Duplicate: return "already owned: duplicate frame draw suppressed";
  default: return "idle";
  }
}
#pragma pack(push, 8)
struct Source {
  float position[3];
  float halfWorldSize;
  float color[3];
  float opacity;
  float falloff;
  uint32_t kind;
  uint64_t sourceId;
};
#pragma pack(pop)
static_assert(sizeof(Source) == 48 && offsetof(Source, sourceId) == 40, "Corona ABI");
static_assert(sizeof(float) == 4, "IEEE754 binary32 required");
constexpr size_t headerChars = 12;
constexpr size_t maxChars = headerChars + limit * sizeof(Source) * 2;

inline bool valid(const Source* sources, uint32_t count) {
  if (count > limit || (count && !sources)) return false;
  std::array<uint64_t, limit> ids {};
  for (uint32_t i = 0; i < count; ++i) {
    const Source& s = sources[i];
    for (uint32_t axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(s.position[axis]) || std::abs(s.position[axis]) > 1.0e8f ||
          !std::isfinite(s.color[axis]) || s.color[axis] < 0 || s.color[axis] > 16) return false;
    }
    if (!std::isfinite(s.halfWorldSize) || s.halfWorldSize <= 0 || s.halfWorldSize > 1.0e5f ||
        !std::isfinite(s.opacity) || s.opacity < 0 || s.opacity > 1 ||
        !std::isfinite(s.falloff) || s.falloff < 1 || s.falloff > 32 ||
        s.kind > 3 || !s.sourceId) return false;
    ids[i] = s.sourceId;
  }
  std::sort(ids.begin(), ids.begin() + count);
  return std::adjacent_find(ids.begin(), ids.begin() + count) == ids.begin() + count;
}

inline int nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

inline bool encode(const Source* sources, uint32_t count, std::string& output, bool version2 = false) {
  output.clear();
  if ((!version2 && count > legacyLimit) || !valid(sources, count)) return false;
  constexpr char digits[] = "0123456789abcdef";
  output.resize(headerChars + size_t(count) * sizeof(Source) * 2);
  std::memcpy(output.data(), version2 ? "LC02" : "LC01", 4);
  for (uint32_t i = 0; i < 8; ++i) output[4 + i] = digits[(count >> ((7 - i) * 4)) & 15];
  const auto* bytes = reinterpret_cast<const uint8_t*>(sources);
  for (size_t i = 0; i < size_t(count) * sizeof(Source); ++i) {
    output[headerChars + i * 2] = digits[bytes[i] >> 4];
    output[headerChars + i * 2 + 1] = digits[bytes[i] & 15];
  }
  return true;
}

inline bool decode(const char* text, std::vector<Source>& output, bool version2 = false) {
  output.clear();
  if (!text) return false;
  size_t length = 0;
  while (length <= maxChars && text[length]) ++length;
  if (length < headerChars || length > maxChars || std::memcmp(text, version2 ? "LC02" : "LC01", 4)) return false;
  uint32_t count = 0;
  for (uint32_t i = 4; i < headerChars; ++i) {
    const int n = nibble(text[i]);
    if (n < 0) return false;
    count = (count << 4) | uint32_t(n);
  }
  if (count > (version2 ? limit : legacyLimit) || length != headerChars + size_t(count) * sizeof(Source) * 2) return false;
  std::vector<Source> candidate(count);
  auto* bytes = reinterpret_cast<uint8_t*>(candidate.data());
  for (size_t i = 0; i < size_t(count) * sizeof(Source); ++i) {
    const int hi = nibble(text[headerChars + i * 2]);
    const int lo = nibble(text[headerChars + i * 2 + 1]);
    if (hi < 0 || lo < 0) return false;
    bytes[i] = uint8_t((hi << 4) | lo);
  }
  if (!valid(candidate.data(), count)) return false;
  output = std::move(candidate);
  return true;
}
}} // namespace remixapi::local_coronas
