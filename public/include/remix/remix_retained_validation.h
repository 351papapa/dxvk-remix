#pragma once
#include <remix/remix_retained.h>
#include <cmath>
#include <cstring>

namespace remixapi_retained {
inline constexpr const char* clientVisibilityKey = "vc.client.retainedVisibilityVersion";
inline constexpr const char* bridgeVisibilityKey = "vc.bridge.retainedVisibilityVersion";
inline constexpr const char* runtimeVisibilityKey = "vc.api.retainedVisibilityVersion";
inline bool supportedVersion(uint32_t version) {
  return version == REMIXAPI_RETAINED_VERSION_2 || version == REMIXAPI_RETAINED_VERSION;
}
inline const char* clientQueryKey(const char* key) {
  return std::strcmp(key, clientVisibilityKey) == 0 ? bridgeVisibilityKey : key;
}
inline bool valid(const remixapi_RetainedInstance& instance, uint32_t version = REMIXAPI_RETAINED_VERSION_2) {
  const uint32_t allowed = REMIXAPI_RETAINED_ORDERED | (version == REMIXAPI_RETAINED_VERSION ? REMIXAPI_RETAINED_HIDDEN : 0u);
  if (!supportedVersion(version) || !instance.id || !instance.mesh || (instance.flags & ~allowed) ||
      (instance.flags & (REMIXAPI_RETAINED_ORDERED | REMIXAPI_RETAINED_HIDDEN)) == (REMIXAPI_RETAINED_ORDERED | REMIXAPI_RETAINED_HIDDEN) ||
      instance.doubleSided > 1 || instance.useBlend > 1) {
    return false;
  }
  for (const auto& row : instance.transform.matrix) {
    for (float value : row) {
      if (!std::isfinite(value)) {
        return false;
      }
    }
  }
  if (!instance.useBlend) {
    return true;
  }
  const auto& blend = instance.blend;
  return blend.alphaTestEnabled <= 1 && blend.alphaTestReferenceValue <= 255 &&
    blend.alphaTestCompareOp <= 7 && blend.alphaBlendEnabled <= 1 &&
    blend.srcColorBlendFactor <= 18 && blend.dstColorBlendFactor <= 18 && blend.colorBlendOp <= 4 &&
    blend.srcAlphaBlendFactor <= 18 && blend.dstAlphaBlendFactor <= 18 && blend.alphaBlendOp <= 4 &&
    blend.textureColorArg1Source <= 3 && blend.textureColorArg2Source <= 3 && blend.textureColorOperation <= 7 &&
    blend.textureAlphaArg1Source <= 3 && blend.textureAlphaArg2Source <= 3 && blend.textureAlphaOperation <= 7 &&
    blend.isTextureFactorBlend <= 1 && blend.isVertexColorBakedLighting <= 1 && blend.writeMask <= 15;
}
inline bool sameDescription(const remixapi_RetainedInstance& a, const remixapi_RetainedInstance& b) {
  return a.id == b.id && a.mesh == b.mesh && a.categoryFlags == b.categoryFlags &&
    a.doubleSided == b.doubleSided && a.wetTag == b.wetTag && a.useBlend == b.useBlend &&
    a.objectPickingValue == b.objectPickingValue &&
    (a.flags & ~REMIXAPI_RETAINED_HIDDEN) == (b.flags & ~REMIXAPI_RETAINED_HIDDEN) &&
    std::memcmp(&a.blend, &b.blend, sizeof(a.blend)) == 0;
}
inline bool sameTransform(const remixapi_RetainedInstance& a, const remixapi_RetainedInstance& b) {
  return std::memcmp(&a.transform, &b.transform, sizeof(a.transform)) == 0;
}
inline bool visibleUnordered(const remixapi_RetainedInstance& instance) {
  return (instance.flags & (REMIXAPI_RETAINED_ORDERED | REMIXAPI_RETAINED_HIDDEN)) == 0;
}
}
