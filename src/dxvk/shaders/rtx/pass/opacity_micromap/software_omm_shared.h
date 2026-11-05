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

// Software opacity micromap constants shared by C++ and shaders.

#include "rtx/pass/frame_features_validate.h"

// RaytraceArgs::softOmmFlags
#define SOFT_OMM_FLAG_VISIBILITY            1u // rtx.alphaBake.softwareOmmVisibility: visibility rays act on the state
#define SOFT_OMM_FLAG_COUNTERS              2u // rtx.alphaBake.softwareOmmCounters: candidates are counted

// Candidate states: the micromap states (VkOpacityMicromapSpecialIndexEXT order), plus "no software data".
#define SOFT_OMM_STATE_TRANSPARENT          0u
#define SOFT_OMM_STATE_OPAQUE               1u
#define SOFT_OMM_STATE_UNKNOWN_TRANSPARENT  2u
#define SOFT_OMM_STATE_UNKNOWN_OPAQUE       3u
#define SOFT_OMM_STATE_NONE                 4u

// rtx.alphaBake.softwareOmmCounters: SOFT_OMM_COUNTER_STATES counters (one per state above) per pass, in the
// LightSamplingStats buffer (bound to every ray tracing pass) after the frame-feature slots.
#define SOFT_OMM_COUNTER_PASS_OTHER         0u
#define SOFT_OMM_COUNTER_PASS_DIRECT        1u // integrate_direct
#define SOFT_OMM_COUNTER_PASS_INDIRECT      2u // integrate_indirect (ray gen and closest hit)
#define SOFT_OMM_COUNTER_PASS_NEE           3u // integrate_nee
#define SOFT_OMM_COUNTER_PASSES             4u
#define SOFT_OMM_COUNTER_STATES             5u
#define SOFT_OMM_COUNTER_BASE               (FRAME_FEATURES_SEEN_SLOT + FRAME_FEATURES_SEEN_SLOTS)
#define SOFT_OMM_COUNTER_SLOTS              (SOFT_OMM_COUNTER_PASSES * SOFT_OMM_COUNTER_STATES)

// The resolve skip (rtx.alphaBake.softwareOmmResolveSkip, software_omm.slangh).
#define SOFT_OMM_FLAG_RESOLVE_SKIP          4u // RaytraceArgs::softOmmFlags: the resolve loops skip as hardware micromaps cull
#define SOFT_OMM_FLAG_RESOLVE_SKIP_INDIRECT 8u // RaytraceArgs::softOmmFlags: ... the indirect ones too (needs RESOLVE_SKIP;
                                               // rtx.alphaBake.softwareOmmResolveSkipIndirect)
#define SOFT_OMM_ENTRY_NON_OPAQUE           (1u << 21) // table x: the surface's geometry is non-opaque (or FORCE_NO_OPAQUE)
#define SOFT_OMM_ENTRY_FORCE_NO_OPAQUE      (1u << 22) // table x: the instance has VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR
#define SOFT_OMM_SKIP_NONE                  0u
#define SOFT_OMM_SKIP_TRANSPARENT           1u // FORCE_OPAQUE iteration: skip transparent micro-triangles
#define SOFT_OMM_SKIP_CULL_NON_OPAQUE       2u // emulated CULL_NON_OPAQUE iteration
#define SOFT_OMM_RESOLVE_MAX_SKIPS          16u
#define SOFT_OMM_PAYLOAD_SKIP_TRANSPARENT     (-1024.0f) // payload segmentHitDistance before a TraceRay (skip mode)
#define SOFT_OMM_PAYLOAD_SKIP_CULL_NON_OPAQUE (-2048.0f)
// Resolve counters, after the visibility counters: per resolve pass, per kind.
#define SOFT_OMM_RESOLVE_PASS_OTHER         0u
#define SOFT_OMM_RESOLVE_PASS_GBUFFER       1u // primary and PSR resolves (gbuffer passes)
#define SOFT_OMM_RESOLVE_PASS_INDIRECT      2u // integrate_indirect (incl. its SHaRC update / query variants)
#define SOFT_OMM_RESOLVE_PASSES             3u
#define SOFT_OMM_RESOLVE_COUNTER_TRANSPARENT 0u // a re-trace past a transparent micro-triangle (FORCE_OPAQUE iteration)
#define SOFT_OMM_RESOLVE_COUNTER_NON_OPAQUE  1u // a re-trace in an emulated CULL_NON_OPAQUE iteration (any culled hit)
#define SOFT_OMM_RESOLVE_COUNTER_CAP         2u // a ray reached the re-trace cap and fell back to the hit function or a real CULL_NON_OPAQUE trace
#define SOFT_OMM_RESOLVE_COUNTER_KINDS       3u
#define SOFT_OMM_RESOLVE_COUNTER_BASE       (SOFT_OMM_COUNTER_BASE + SOFT_OMM_COUNTER_SLOTS)
#define SOFT_OMM_RESOLVE_COUNTER_SLOTS      (SOFT_OMM_RESOLVE_PASSES * SOFT_OMM_RESOLVE_COUNTER_KINDS)

// The traversal (rtx.alphaBake.softwareOmmTraversal, software_omm.slangh). The ray query resolve loops trace without
// FORCE_OPAQUE and act on non-opaque candidates in a candidate loop. These are RaytraceArgs::softOmmFlags bits, one per
// traversal pass. The host writes rtx.alphaBake.softwareOmmTraversal (bits 1 / 2 / 4) shifted by SOFT_OMM_FLAG_TRAVERSAL_SHIFT.
#define SOFT_OMM_FLAG_TRAVERSAL_SHIFT        4u
#define SOFT_OMM_FLAG_TRAVERSAL_GBUFFER      16u // primary and PSR ray query resolves (gbuffer passes)
#define SOFT_OMM_FLAG_TRAVERSAL_SHARC_UPDATE 32u // SHaRC update ray query resolves
#define SOFT_OMM_FLAG_TRAVERSAL_INDIRECT     64u // indirect integration ray query resolves (integrate_indirect_rayquery*)
#define SOFT_OMM_FLAG_TRAVERSAL_MASK         (SOFT_OMM_FLAG_TRAVERSAL_GBUFFER | SOFT_OMM_FLAG_TRAVERSAL_SHARC_UPDATE | \
                                              SOFT_OMM_FLAG_TRAVERSAL_INDIRECT)
// Traversal passes (SOFT_OMM_TRAVERSAL_PASS_ID, a compile-time constant at each resolve call site).
#define SOFT_OMM_TRAVERSAL_PASS_OTHER        0u // never acts
#define SOFT_OMM_TRAVERSAL_PASS_GBUFFER      1u
#define SOFT_OMM_TRAVERSAL_PASS_SHARC_UPDATE 2u
#define SOFT_OMM_TRAVERSAL_PASS_INDIRECT     3u
#define SOFT_OMM_TRAVERSAL_PASSES            4u
// Candidate loop modes (bit field): what the iteration asked for, plus the 2-state OMM ray flag.
#define SOFT_OMM_TRAVERSAL_NONE              0u
#define SOFT_OMM_TRAVERSAL_FORCE_OPAQUE      1u // FORCE_OPAQUE iteration: skip transparent, commit everything else
#define SOFT_OMM_TRAVERSAL_CULL_NON_OPAQUE   2u // CULL_NON_OPAQUE iteration: commit only what hardware keeps opaque
#define SOFT_OMM_TRAVERSAL_TWO_STATE         4u // ForceOpacityMicromap2StateEXT: unknown-transparent / -opaque -> T / O
#define SOFT_OMM_RAY_FLAG_FORCE_2_STATE      0x400u // the ForceOpacityMicromap2StateEXT ray flag value
// Traversal counters, after the resolve counters: per traversal pass, per kind (one per non-opaque candidate).
#define SOFT_OMM_TRAVERSAL_COUNTER_TRANSPARENT 0u // transparent micro-triangle: skipped
#define SOFT_OMM_TRAVERSAL_COUNTER_OPAQUE      1u // opaque micro-triangle: committed (the hit needs no alpha test)
#define SOFT_OMM_TRAVERSAL_COUNTER_UNKNOWN     2u // unknown micro-triangle: committed (the hit function alpha tests)
#define SOFT_OMM_TRAVERSAL_COUNTER_NO_DATA     3u // surface without software data: committed (as with FORCE_OPAQUE)
#define SOFT_OMM_TRAVERSAL_COUNTER_CULLED      4u // CULL_NON_OPAQUE iteration: skipped (anything but a kept opaque one)
#define SOFT_OMM_TRAVERSAL_COUNTER_KINDS       5u
#define SOFT_OMM_TRAVERSAL_COUNTER_BASE      (SOFT_OMM_RESOLVE_COUNTER_BASE + SOFT_OMM_RESOLVE_COUNTER_SLOTS)
#define SOFT_OMM_TRAVERSAL_COUNTER_SLOTS     (SOFT_OMM_TRAVERSAL_PASSES * SOFT_OMM_TRAVERSAL_COUNTER_KINDS)

// The self-test (rtx.alphaBake.softwareOmmSelfTest, omm_sw_selftest.comp.slang) covers levels 0..SOFT_OMM_SELFTEST_MAX_LEVEL,
// with two result words per level (mismatching micro-triangles, failed edge checks).
#define SOFT_OMM_SELFTEST_MAX_LEVEL         12u
#define SOFT_OMM_SELFTEST_LEVELS            (SOFT_OMM_SELFTEST_MAX_LEVEL + 1u)
#define SOFT_OMM_SELFTEST_THREADS_PER_GROUP 64u
#define SOFT_OMM_SELFTEST_BINDING_RESULTS   0

struct SoftOmmSelfTestArgs
{
  uint32_t totalThreads;     // sum over the levels of 4^level
  uint32_t groupsX;          // workgroups along x (2D dispatch)
  uint32_t pad0;
  uint32_t pad1;
};
