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

// Translator toolkit: the sort order of decals. The scene draws a frontend's decals in the rank each had among the
// frame's decals in draw order.

#include <cstdint>

namespace dxvk {
  namespace translator {

    class DecalRank {
    public:
      // The rank of the frame's next decal. The GPU stores 8 bits, so the rank wraps.
      uint8_t next() {
        return uint8_t(m_count++ & 0xFFu);
      }

      // True when an object's rank must be sent: it changed, or the object was built or rebuilt, which assigns a
      // creation order. lastSent is the rank last sent (-1 = none) and takes rank when it returns true.
      static bool needsSend(int32_t& lastSent, uint8_t rank, bool built) {
        if (!built && lastSent == int32_t(rank)) {
          return false;
        }
        lastSent = rank;
        return true;
      }

      void endFrame() {
        m_count = 0;
      }

    private:
      uint32_t m_count = 0;
    };

  } // namespace translator
} // namespace dxvk
