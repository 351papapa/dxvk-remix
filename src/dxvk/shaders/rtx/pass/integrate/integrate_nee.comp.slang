//!feature wetness ENABLE_SURFACE_WETNESS
/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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

// NV-DXVK start: Compile NRC and ReSTIR GI NEE paths only in their active variants.
//!variant integrate_nee_plain.comp
//!>       ENABLE_NRC=0

//!variant integrate_nee_nrc.comp
//!>       ENABLE_NRC=1

//!variant integrate_nee_restir_gi.comp
//!>       RAB_HAS_RESTIR_GI_RESERVOIRS=1

/* rtx.integrateIndirect.slimAssembly. Slim variants compile out POM, and are bound while the frame's POM mode is off
   or POM does not affect NEE-cache shadows. Lean variants also set INDIRECT_SLIM_LEAN_VISIBILITY, for frames where
   the NEE ray cannot hit a translucent or thin-opaque surface. The host binds them only while the compiled-out
   branches cannot run. */

//!variant integrate_nee_plain_slim.comp
//!>       ENABLE_NRC=0
//!>       INDIRECT_SLIM_NO_POM

//!variant integrate_nee_plain_lean.comp
//!>       ENABLE_NRC=0
//!>       INDIRECT_SLIM_NO_POM
//!>       INDIRECT_SLIM_LEAN_VISIBILITY=1

//!variant integrate_nee_nrc_slim.comp
//!>       ENABLE_NRC=1
//!>       INDIRECT_SLIM_NO_POM

//!variant integrate_nee_nrc_lean.comp
//!>       ENABLE_NRC=1
//!>       INDIRECT_SLIM_NO_POM
//!>       INDIRECT_SLIM_LEAN_VISIBILITY=1

//!variant integrate_nee_restir_gi_slim.comp
//!>       RAB_HAS_RESTIR_GI_RESERVOIRS=1
//!>       INDIRECT_SLIM_NO_POM

//!variant integrate_nee_restir_gi_lean.comp
//!>       RAB_HAS_RESTIR_GI_RESERVOIRS=1
//!>       INDIRECT_SLIM_NO_POM
//!>       INDIRECT_SLIM_LEAN_VISIBILITY=1

/* rtx.integrateIndirect.slimKeepPom. Lean visibility with POM kept. The slim level with POM kept is the full shader. */

//!variant integrate_nee_plain_leanpom.comp
//!>       ENABLE_NRC=0
//!>       INDIRECT_SLIM_LEAN_VISIBILITY=1

//!variant integrate_nee_nrc_leanpom.comp
//!>       ENABLE_NRC=1
//!>       INDIRECT_SLIM_LEAN_VISIBILITY=1

//!variant integrate_nee_restir_gi_leanpom.comp
//!>       RAB_HAS_RESTIR_GI_RESERVOIRS=1
//!>       INDIRECT_SLIM_LEAN_VISIBILITY=1

//!end-variants
// NV-DXVK end

#define RAB_HAS_CURRENT_GBUFFER 1
// INDIRECT_SLIM_NO_POM variants are bound only while POM cannot run (rtx.integrateIndirect.slimAssembly).
#ifndef INDIRECT_SLIM_NO_POM
#define OPAQUE_MATERIAL_USE_POM 1
#endif
// Slot of this pass in the rtx.alphaBake.softwareOmmCounters counters.
#define SOFT_OMM_COUNTER_PASS SOFT_OMM_COUNTER_PASS_NEE

#include "rtx/pass/integrate/integrate_nee_binding_indices.h"

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/integrate/integrate_nee_bindings.slangh"
#include "rtx/pass/sparse_rendering/sparse_rendering.slangh"
#include "rtx/pass/sparse_rendering/compact_active_pixels.slangh"
#include "rtx/utility/gbuffer_helpers.slangh"
#include "rtx/utility/debug_view_helpers.slangh"
#include "rtx/algorithm/integrator_helpers.slangh"
#include "rtxdi/DlssEnhancementFilterFunctions.slangh"
#include "rtx/algorithm/nee_cache.h"
#include "rtx/algorithm/integrator.slangh"
#include "rtx/pass/nrc/nrc_utilities.slangh"
#include "rtx/pass/nrc/nrc_debug_view_utilities.slangh"

#include "rtx/pass/integrate/integrate_nee.slangh"

// The pass can be launched over the active-pixel list, which counts groups of the consumer group size.
#if INTEGRATE_NEE_THREADS_DISPATCH_WIDTH != COMPACT_ACTIVE_PIXELS_CONSUMER_GROUP_SIZE_X || \
    INTEGRATE_NEE_THREADS_DISPATCH_HEIGHT != COMPACT_ACTIVE_PIXELS_CONSUMER_GROUP_SIZE_Y
#error "Integrate NEE's group size must match COMPACT_ACTIVE_PIXELS_CONSUMER_GROUP_SIZE_X / _Y."
#endif

[shader("compute")]
[numthreads(INTEGRATE_NEE_THREADS_DISPATCH_WIDTH, INTEGRATE_NEE_THREADS_DISPATCH_HEIGHT, 1)]
void main(uint2 threadIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID)
{
  uvec2 pixelCoordinate;
  if (!loadLaunchPixelCoordinate(threadIndex, ActiveLocalPixelCoords, ActivePixelList,
                                 cb.camera.resolution, cb.sparseRenderingArgs, COMPACT_ACTIVE_PIXELS_LAUNCH_INTEGRATE_NEE, pixelCoordinate))
  {
    return;
  }

  assembleIndirectLighting(pixelCoordinate);
}
