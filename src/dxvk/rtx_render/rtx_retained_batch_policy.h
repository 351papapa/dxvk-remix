/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once

namespace dxvk::retained_batch_policy {

// Disabled emission is forced to zero by the material shader.
// An enabled nonzero, infinite or NaN authored intensity keeps its canonical surface.
inline bool hasNoEffectiveAuthoredEmission(bool enabled, float authoredIntensity) {
  return !enabled || authoredIntensity == 0.f;
}

}
