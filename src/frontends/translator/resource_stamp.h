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

// Translator toolkit: resource ids and versions. No API types.
//
// A frontend gives every resource it reports (buffer, texture, display list, ...) an id, and bumps a version on every
// write. Both come from one process-wide counter, so they are never reused: a new object created at a freed object's
// address can never look like the old one, and a version names one content state of one resource.

#include <atomic>
#include <cstdint>

namespace dxvk {
  namespace translator {

    inline uint64_t nextStamp() {
      static std::atomic<uint64_t> s_counter { 0 };
      return s_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    // id 0 = no resource. version 0 = unversioned: the content must be treated as changed on every use.
    struct ResourceRef {
      uint64_t id = 0;
      uint64_t version = 0;

      bool operator==(const ResourceRef& other) const { return id == other.id && version == other.version; }
      bool operator!=(const ResourceRef& other) const { return !(*this == other); }
    };

  } // namespace translator
} // namespace dxvk
