#pragma once
#include <remix/remix_c.h>

#define REMIXAPI_LOCAL_STARBURST_VERSION 1u
#define REMIXAPI_LOCAL_STARBURST_LIMIT 256u
#define REMIXAPI_SUBMIT_LOCAL_STARBURSTS_NAME "remixapi_SubmitLocalStarbursts"

#pragma pack(push, 8)
typedef struct remixapi_LocalStarburstSource {
  float position[3];
  float halfScreenHeight; // Half-width as a fraction of half the display height.
  float color[3];         // Display tint, each component in [0,1].
  float opacity;          // Includes the host's distance, direction and edge fades.
  float aspect;           // Width divided by height.
  uint32_t kind;          // 0 general, 1 fixture, 2 headlight, 3 taillight.
  uint64_t sourceId;      // Nonzero and unique within the snapshot.
} remixapi_LocalStarburstSource;
#pragma pack(pop)

typedef remixapi_ErrorCode (REMIXAPI_PTR* PFN_remixapi_SubmitLocalStarbursts)(
  uint32_t version, uint32_t stride, uint32_t count, const void* records);

#ifdef __cplusplus
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <array>

namespace remixapi::starbursts {
  static_assert(sizeof(remixapi_LocalStarburstSource) == 48);
  static_assert(offsetof(remixapi_LocalStarburstSource, sourceId) == 40);

  inline bool valid(uint32_t version, uint32_t stride, uint32_t count, const remixapi_LocalStarburstSource* src) {
    if (version != REMIXAPI_LOCAL_STARBURST_VERSION || stride != sizeof(*src) ||
        count > REMIXAPI_LOCAL_STARBURST_LIMIT || (count && !src)) {
      return false;
    }
    std::array<uint64_t, REMIXAPI_LOCAL_STARBURST_LIMIT> ids;
    for (uint32_t i = 0; i < count; ++i) {
      const auto& s = src[i];
      if (!s.sourceId || s.kind > 3 || !(s.opacity >= 0 && s.opacity <= 1) ||
          !(s.halfScreenHeight > 0 && s.halfScreenHeight <= 4) || !(s.aspect >= 0.0625f && s.aspect <= 16)) {
        return false;
      }
      for (unsigned j = 0; j < 3; ++j) {
        if (!std::isfinite(s.position[j]) || !(s.color[j] >= 0 && s.color[j] <= 1)) {
          return false;
        }
      }
      ids[i] = s.sourceId;
    }
    std::sort(ids.begin(), ids.begin() + count);
    return std::adjacent_find(ids.begin(), ids.begin() + count) == ids.begin() + count;
  }
}
#endif
