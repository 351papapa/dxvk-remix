/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Scene interface.
// The types an API frontend hands to the backend scene, and the options that switch the new path on.

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <remix/remix_c.h>

#include "rtx_frontend_interface.h"
#include "rtx_option.h"
#include "rtx_scene_manager.h"

namespace dxvk {
  namespace scene {

    class Options {
    public:
      RTX_OPTION("rtx.scene", bool, originShiftEvent, false,
                 "A scene-origin change shifts the scene's instances as one event, which "
                 "keeps their history, instead of moving each one. The moved instances keep their merged BLAS.");
      RTX_OPTION("rtx.scene", uint32_t, selfCheckFrames, 0,
                 "Diagnostic for the retained scene: every this many frames, compares every instance the scene "
                 "owns with the host's description (transform, hidden state and material). Mismatches are logged as "
                 "[SceneCheck] and the instance is rebuilt. 0 = off.");
      RTX_OPTION("rtx.scene", bool, directInstancerSplit, false,
                 "Diagnostic for opacity micromaps on point instancers: the scene owns "
                 "the point-instanced Remix API submeshes of every second mesh (chosen by mesh handle), and the rest "
                 "stay on the retained path. With rtx.profile.detailedLogging, the [D3d9Alpha] pointInstancers line "
                 "then compares both lanes in one run. Takes effect when the instances are next built.");
    };

    using Handle = uint64_t;

    // UpdateRetainedInstances, one entry. flags: REMIXAPI_RETAINED_UPDATE_TRANSFORM / _VISIBILITY.
    struct InstanceUpdate {
      Handle handle = 0;
      uint32_t flags = 0;
      Matrix4 objectToWorld;
      bool visible = true;
    };

    // One command for the backend scene. Create / Describe carry the prepared description (the expensive half of a
    // create, done on the calling thread).
    struct SceneCommand {
      enum class Type : uint8_t { Create, Describe, Update, Destroy, Frame };
      Type type = Type::Frame;
      Handle handle = 0;
      frontend::PreparedInstance instance;
      InstanceUpdate update;
      Vector3 origin { 0.f, 0.f, 0.f };   // Frame: DrawRetainedInstances' scene origin
      uint32_t events = 0;                // Frame: frontend::FrameEventBits
    };

  } // namespace scene
} // namespace dxvk
