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

// rtx.alphaBake.compactMicromaps. A triangle of a baked opacity
// micromap array whose micro-triangles are all transparent or all opaque gets a special index
// instead of micromap data; the other (mixed) triangles are listed in a compact triangle array that
// points at their data in the baked array.

#define OMM_COMPACT_BINDING_ARRAY_INPUT     0
#define OMM_COMPACT_BINDING_INDEX_OUTPUT    1 // uint32 per triangle: special index or compact index
#define OMM_COMPACT_BINDING_TRIANGLE_OUTPUT 2 // VkMicromapTriangleEXT (2 uint32) per mixed triangle
#define OMM_COMPACT_BINDING_COUNTER         3 // [0] mixed triangles

#define OMM_COMPACT_THREADS_PER_GROUP       64

// VkOpacityMicromapSpecialIndexEXT
#define OMM_SPECIAL_INDEX_FULLY_TRANSPARENT 0xFFFFFFFFu // -1
#define OMM_SPECIAL_INDEX_FULLY_OPAQUE      0xFFFFFFFEu // -2

struct OmmCompactArgs
{
  uint32_t numTriangles;
  uint32_t wordsPerTriangle; // 32-bit words of the array per triangle
  uint32_t opaqueWord;       // a word of all-opaque micro-triangles (0x55555555 4-state, 0xFFFFFFFF 2-state)
  uint32_t triangleWord1;    // VkMicromapTriangleEXT subdivisionLevel | format << 16
  uint32_t groupsX;          // workgroups along x (one workgroup per triangle, 2D dispatch)
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};
