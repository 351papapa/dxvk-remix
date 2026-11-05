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

// Translator toolkit: the frame's fixed-function fog states in the order the game drew them, each once, which the
// frontend sends as FrameInfo::fog.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "../../util/util_vector.h"
#include "../../util/xxHash/xxhash.h"

namespace dxvk {
  namespace translator {

    // A fixed-function fog state. mode holds a frontend::FogMode value, and 0 is no fog. The layout is that of the
    // backend's FogState, so both hash alike.
    struct FogRecord {
      uint32_t mode = 0;
      Vector3 color = Vector3();
      float scale = 0.f;
      float end = 0.f;
      float density = 0.f;

      XXH64_hash_t hash() const {
        return XXH3_64bits(this, sizeof(FogRecord));
      }
    };

    class FogOrder {
    public:
      // Appends the draw's fog to order, unless the draw has no fog or the fog is in order already.
      void note(const FogRecord& fog, std::vector<FogRecord>& order) {
        if (fog.mode == 0) {
          return;
        }
        const XXH64_hash_t hash = fog.hash();
        if (std::find(m_seen.begin(), m_seen.end(), hash) != m_seen.end()) {
          return;
        }
        m_seen.push_back(hash);
        order.push_back(fog);
      }

      void endFrame() {
        m_seen.clear();
      }

    private:
      std::vector<XXH64_hash_t> m_seen;
    };

  } // namespace translator
} // namespace dxvk
