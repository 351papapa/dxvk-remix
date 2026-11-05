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

// rtx.alphaBake.softwareOmm. Packs a baked, compacted opacity micromap array into a software store. Word t < numTriangles
// is triangle t's index (OMM_SPECIAL_INDEX_* or its mixed slot, from the compaction pass). Then come wordsPerTriangle
// words per mixed slot, copied from the baked array at the slot's byte offset, which the compaction pass's triangle
// array holds (VkMicromapTriangleEXT layout, [0] dataOffset in bytes).

#define OMM_SW_PACK_BINDING_ARRAY_INPUT     0
#define OMM_SW_PACK_BINDING_INDEX_INPUT     1
#define OMM_SW_PACK_BINDING_TRIANGLE_INPUT  2
#define OMM_SW_PACK_BINDING_STORE_OUTPUT    3

#define OMM_SW_PACK_THREADS_PER_GROUP       64

struct OmmSwPackArgs
{
  uint32_t numTriangles;
  uint32_t wordsPerTriangle;
  uint32_t numMixed;
  uint32_t totalWords;       // numTriangles + numMixed * wordsPerTriangle
  uint32_t groupsX;          // workgroups along x (one thread per store word, 2D dispatch)
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};
