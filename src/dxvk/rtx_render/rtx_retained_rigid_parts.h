/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <cstddef>

namespace dxvk::retained_compat {

  template<typename Draw, typename Transforms, typename Part, typename Material>
  const Material& applyExpandedPart(Draw& draw, const Transforms& parent, const Part& part, const Material& fallback) {
    auto& transforms = draw.modifyTransformData();
    transforms = parent;
    transforms.objectToWorld = parent.objectToWorld * part.submeshToObject;
    transforms.objectToView = parent.worldToView * transforms.objectToWorld;
    if (part.meshReplacement) {
      transforms.textureTransform = {};
      transforms.texgenMode = decltype(transforms.texgenMode)::None;
      transforms.instancesToObject = part.instancesToObject;
      transforms.instancesFromHost = false;
    }
    draw.modifyCategoryFlags() = part.categories;
    return part.material ? *part.material : fallback;
  }

  // A changed last part must not let the first one spend its first update of the frame.
  template<typename Check, typename Update>
  bool updateRigidParts(size_t count, Check check, Update update) {
    if (!count) {
      return false;
    }
    for (size_t i = 0; i < count; ++i) {
      if (!check(i)) {
        return false;
      }
    }
    for (size_t i = 0; i < count; ++i) {
      update(i);
    }
    return true;
  }

}
