#pragma once
#include <remix/remix_retained.h>
#include <cmath>

namespace remixapi_retained {
inline bool valid(const remixapi_RetainedInstance& instance) {
  if (!instance.id || !instance.mesh || (instance.flags & ~REMIXAPI_RETAINED_ORDERED) || instance.doubleSided > 1 || instance.useBlend > 1) {
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
}
