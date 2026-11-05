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

// Translator toolkit: what a frontend's two threads exchange. The thread that sees the game's draws decides per draw
// and collects the frame's commands in a FrameBatch. The render thread applies the batch once per frame and answers
// through the Channel.

#include <cstdint>
#include <mutex>
#include <vector>

#include "fog_order.h"
#include "../../util/util_matrix.h"

namespace dxvk {
  namespace translator {

    // Render thread to draw thread: what became of a handed-over object.
    struct Channel {
      enum class Kind : uint8_t {
        Rejected,  // it cannot be owned by the scene: its draws are dropped (with a back-off)
        Released,  // the render thread wants it back (aged out, a reset, rebuilt and rejected): the draw thread sends
                   // a destroy and hands the draw over again. The scene keeps the entry until that destroy, so
                   // nothing disappears while the draw thread still skips the draw.
        Retry,     // it cannot be built yet (a 3D-skybox draw before the first main camera): handed back without a
                   // back-off, so the next draw is handed over again.
      };
      struct Result {
        uint64_t handle;
        Kind kind;
      };
      std::mutex mutex;
      std::vector<Result> results;
      // An option that invalidates draw-call translation changed: every scene-owned object is prepared and described
      // again at its next draw.
      bool reprepare = false;

      void push(uint64_t handle, Kind kind) {
        std::lock_guard<std::mutex> lock(mutex);
        results.push_back({ handle, kind });
      }

      void requestReprepare() {
        std::lock_guard<std::mutex> lock(mutex);
        reprepare = true;
      }
    };

    // Draw thread to render thread: the frame's commands for the objects the scene owns.
    struct FrameBatch {
      struct Move {
        uint64_t handle;
        Matrix4 objectToWorld;
      };
      std::vector<uint64_t> seen;        // scene-owned objects drawn this frame (unchanged, moved or updated)
      std::vector<Move> moves;
      std::vector<uint64_t> destroys;    // scene-owned objects whose draw changed, or that the scene handed back
      // Scene-owned decals whose rank in this frame's draw order changed, or that were built or rebuilt (DecalRank).
      struct SortOrder {
        uint64_t handle;
        uint8_t order;
      };
      std::vector<SortOrder> sortOrders;
      // The frame's fog states in draw order (FogOrder).
      std::vector<FogRecord> fogOrder;

      bool empty() const {
        return seen.empty() && moves.empty() && destroys.empty() && sortOrders.empty() && fogOrder.empty();
      }
    };

  } // namespace translator
} // namespace dxvk
