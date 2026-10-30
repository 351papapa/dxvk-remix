// SPDX-License-Identifier: MIT
#pragma once
#include "rtx/utility/shader_types.h"
struct LocalSphereSelectArgs {
  vec3 cameraPosition;
  uint inputCount;
  uint firstPass;
  uint groupCount;
  uint sampleCount;
  uint padding;
};
#ifdef __cplusplus
static_assert(sizeof(LocalSphereSelectArgs) == 32);
#endif
