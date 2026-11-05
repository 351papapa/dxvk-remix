#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../../util/util_matrix.h"

namespace dxvk {
  // Explicit opt-in replacement data, in the game's palette order and bind space.
  struct SkinningRetargetData {
    std::vector<int32_t> parents;
    std::vector<Matrix4> originalBind;
    std::vector<Matrix4> replacementInverseBind;
    std::vector<Matrix4> localRestCorrection;
  };

  // Preserve animated local transforms while replacing the local rest transforms.
  // The GPU vertex deformation pass consumes the resulting palette.
  inline bool retargetSkinningPalette(const SkinningRetargetData& data,
      const Matrix4* input, uint32_t count, Matrix4* output,
      const char** failure = nullptr, uint32_t* failureBone = nullptr) {
    const auto reject = [&](const char* reason, uint32_t index) {
      if (failure) {
        *failure = reason;
      }
      if (failureBone) {
        *failureBone = index;
      }
      return false;
    };
    const uint32_t n = uint32_t(data.parents.size());
    if (!n || !input || !output || (count != n && count != n + 1) || n != data.originalBind.size()
        || n != data.replacementInverseBind.size()
        || n != data.localRestCorrection.size()) {
      return reject("metadata/palette count", count);
    }
    // A host may append a zero palette entry for zero-weight influence slots.
    if (count == n + 1) {
      for (uint32_t c = 0; c < 4; ++c) {
        for (uint32_t r = 0; r < 4; ++r) {
          if (!(c == 3 && r == 3) && input[n][c][r] != 0.f) {
            return reject("nonzero extra palette slot", n);
          }
        }
      }
      // remixapi_Transform has three rows; tomat4 appends homogeneous w=1.
      if (input[n][3][3] != 0.f && input[n][3][3] != 1.f) {
        return reject("invalid extra slot w", n);
      }
    }
    std::vector<Matrix4> originalPose(n), replacementPose(n);
    for (uint32_t i = 0; i < n; ++i) {
      if (data.parents[i] < -1 || data.parents[i] >= int32_t(i)) {
        return reject("parent index", i);
      }
      originalPose[i] = input[i] * data.originalBind[i];
      // Singular animation matrices cannot be used to recover local transforms.
      const auto& m = originalPose[i];
      const float det = dot(Vector3(m[0].x, m[0].y, m[0].z),
        cross(Vector3(m[1].x, m[1].y, m[1].z), Vector3(m[2].x, m[2].y, m[2].z)));
      if (!std::isfinite(det) || std::abs(det) < 1e-10f) {
        return reject("singular original pose", i);
      }
      for (uint32_t c = 0; c < 4; ++c) {
        for (uint32_t r = 0; r < 4; ++r) {
          if (!std::isfinite(m[c][r])) {
            return reject("nonfinite original pose", i);
          }
        }
      }
    }
    for (uint32_t i = 0; i < n; ++i) {
      const int32_t p = data.parents[i];
      if (p < 0) {
        // The root palette also carries the game's model-space axis conversion
        // and root motion. Preserve that global motion without conjugating it
        // by the replacement root's pivot offset.
        replacementPose[i] = input[i] * inverse(data.replacementInverseBind[i]);
      } else {
        const Matrix4 local = inverse(originalPose[p]) * originalPose[i];
        replacementPose[i] = replacementPose[p] * data.localRestCorrection[i] * local;
      }
    }
    for (uint32_t i = 0; i < n; ++i) {
      output[i] = replacementPose[i] * data.replacementInverseBind[i];
    }
    if (count == n + 1) {
      output[n] = input[n];
    }
    return true;
  }
}
