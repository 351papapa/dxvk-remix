/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>

// A texture hash remains valid across changes to other textures.
class TextureHashCache {
  uint64_t m_version = 1;
  uint64_t m_hash = 0;
  uint64_t m_hashVersion = 0;

public:
  uint64_t version() const {
    return m_version;
  }

  bool tryGet(uint64_t* hash, uint64_t* version) const {
    *version = m_version;
    if (m_hash == 0 || m_hashVersion != m_version) {
      return false;
    }
    *hash = m_hash;
    return true;
  }

  void store(uint64_t hash, uint64_t version) {
    if (hash == 0 || version != m_version) {
      return;
    }
    m_hash = hash;
    m_hashVersion = version;
  }

  void invalidate() {
    ++m_version;
    if (m_version == 0) {
      m_version = 1;
    }
  }
};
