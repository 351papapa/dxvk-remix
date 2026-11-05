/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "rtx_graph_instance.h"
#include "rtx_graph_manager.h"
#include "../rtx_retained_scene.h"

namespace dxvk {

PrimInstance ScenePrimOwner::resolve(uint32_t primIndex) const {
  if (scene == nullptr || primIndex >= slots.size()) {
    return PrimInstance();
  }
  const Slot& slot = slots[primIndex];
  switch (slot.type) {
  case PrimInstance::Type::Instance: {
    RtInstance* instance = scene->submeshInstance(handle, slot.submesh);
    return instance != nullptr ? PrimInstance(instance) : PrimInstance();
  }
  case PrimInstance::Type::Light:
    return slot.light != nullptr ? PrimInstance(slot.light) : PrimInstance();
  default:
    return PrimInstance();
  }
}

void GraphInstance::removeInstance() {
  m_graphManager->removeInstance(m_id);
}

} // namespace dxvk
