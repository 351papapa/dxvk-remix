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

// rtx.alphaBake.census. Classifies the triangles of a baked
// opacity micromap array as uniformly transparent, uniformly opaque or mixed.

#define OMM_CENSUS_BINDING_ARRAY_INPUT  0
#define OMM_CENSUS_BINDING_COUNTERS     1

#define OMM_CENSUS_THREADS_PER_GROUP    64

// The counter buffer (uint32) holds [0..2] transparent, opaque and mixed triangles over the window,
// then 3 counters in the same order per census slot, one slot per micromap.
#define OMM_CENSUS_GLOBAL_COUNTERS      4

struct OmmCensusArgs
{
  uint32_t numTriangles;
  uint32_t wordsPerTriangle; // 32-bit words of the array per triangle (the bytes per triangle are a multiple of 4)
  uint32_t opaqueWord;       // a word of all-opaque micro-triangles (0x55555555 4-state, 0xFFFFFFFF 2-state)
  uint32_t slotOffset;       // first counter of this micromap's slot
  uint32_t groupsX;          // workgroups along x (one workgroup per triangle, 2D dispatch)
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};
