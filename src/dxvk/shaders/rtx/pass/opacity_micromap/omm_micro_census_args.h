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

// rtx.alphaBake.censusMicroStates. Counts the micro-triangle states (T / O / UT / UO = 0..3) of a baked opacity
// micromap array, split by format (4-state / 2-state), subdivision level and triangle class. Uniform triangles, which
// compaction gives a special index, have every micro-triangle T or every one O, and mixed triangles are the rest.

#define OMM_MICRO_CENSUS_BINDING_ARRAY_INPUT  0
#define OMM_MICRO_CENSUS_BINDING_COUNTERS     1

#define OMM_MICRO_CENSUS_THREADS_PER_GROUP    64

#define OMM_MICRO_CENSUS_LEVELS               16 // subdivision levels 0..15 (higher ones are clamped to 15)

// One bin per (class, format, level), with 64-bit counters stored as (low, high) uint32 pairs:
//   [0..7]   micro-triangles in state T / O / UT / UO
//   [8..9]   triangles
//   [10..17] sum over triangles of the triangle's share of each state, 1.0 = 65536 (triangle-weighted)
#define OMM_MICRO_CENSUS_BIN_TRIANGLES        8
#define OMM_MICRO_CENSUS_BIN_SHARES           10
#define OMM_MICRO_CENSUS_BIN_WORDS            18
#define OMM_MICRO_CENSUS_SHARE_ONE            65536
// bin = (classIndex * 2 + twoState) * OMM_MICRO_CENSUS_LEVELS + level, classIndex 0 uniform / 1 mixed
#define OMM_MICRO_CENSUS_BINS                 (2 * 2 * OMM_MICRO_CENSUS_LEVELS)
#define OMM_MICRO_CENSUS_COUNTER_WORDS        (OMM_MICRO_CENSUS_BINS * OMM_MICRO_CENSUS_BIN_WORDS)

struct OmmMicroCensusArgs
{
  uint32_t numTriangles;
  uint32_t wordsPerTriangle; // 32-bit words of the array per triangle (the bytes per triangle are a multiple of 4)
  uint32_t twoState;         // 1: 1 bit per micro-triangle (T / O), 0: 2 bits (T / O / UT / UO), LSB first
  uint32_t level;            // subdivision level, clamped to OMM_MICRO_CENSUS_LEVELS - 1
  uint32_t groupsX;          // workgroups along x (one workgroup per triangle, 2D dispatch)
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};
