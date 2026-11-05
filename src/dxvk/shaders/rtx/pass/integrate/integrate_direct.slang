//!feature wetness ENABLE_SURFACE_WETNESS
/*
* Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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

//!variant integrate_direct_rayquery.comp

//!variant integrate_direct_rayquery_raygen.rgen
//!>       RAY_PIPELINE

// Note: these carry the deferred NRC training-path setup. It is a variant rather than a runtime branch so the
// variants above, selected whenever the setup stays in the GBuffer, compile without any of the NRC code.
//!variant integrate_direct_nrc_rayquery.comp
//!>       ENABLE_NRC=1

//!variant integrate_direct_nrc_rayquery_raygen.rgen
//!>       RAY_PIPELINE ENABLE_NRC=1

// rtx.lightSampling.bvhDedicatedSunSample adds a separate distant-light sample and shadow ray to the primary LightBVH
// sample. These are bound only when LightBvh::dedicatedSunActive(), so the variants above compile without the second
// visibility-ray site.
//!variant integrate_direct_sun_rayquery.comp
//!>       LIGHT_BVH_DEDICATED_SUN=1

// rtx.integrateDirect.slimVariants. These compile out diffusion-profile SSS (NoSss), see integrator_direct.slangh and
// selectSlimLevel().
//!variant integrate_direct_nosss_rayquery.comp
//!>       INTEGRATE_DIRECT_SSS_PROFILE=0

//!variant integrate_direct_sun_nosss_rayquery.comp
//!>       LIGHT_BVH_DEDICATED_SUN=1 INTEGRATE_DIRECT_SSS_PROFILE=0

// The Lean levels, which also drop the translucent / thin-opaque shadow-ray handling, as two kernels (integrator_direct.slangh,
// INTEGRATE_DIRECT_SPLIT).
// The shade kernel does everything up to the NEE visibility rays and writes the per-pixel record. It traces nothing, so one
// per dedicated-sun mode serves Lean, LeanCore and LeanLegacy.
// The trace kernel traces the visibility rays, accumulates and writes the outputs. There is one per visibility form, and any
// sun sample is in the record.
// The _core and _legacy trace forms read the NEE cutout candidates from the core material record (rtx.integrateDirect.coreMaterial,
// visibility_core_material.slangh). _core applies the TextureOps part of the legacy record per surface, and _legacy the whole
// LegacySurface record.
//!variant integrate_direct_split_shade_lean_rayquery.comp
//!>       INTEGRATE_DIRECT_SSS_PROFILE=0 INTEGRATE_DIRECT_LEAN_VISIBILITY=1 INTEGRATE_DIRECT_SPLIT=1

//!variant integrate_direct_split_shade_sun_lean_rayquery.comp
//!>       LIGHT_BVH_DEDICATED_SUN=1 INTEGRATE_DIRECT_SSS_PROFILE=0 INTEGRATE_DIRECT_LEAN_VISIBILITY=1 INTEGRATE_DIRECT_SPLIT=1

//!variant integrate_direct_split_trace_lean_rayquery.comp
//!>       INTEGRATE_DIRECT_SSS_PROFILE=0 INTEGRATE_DIRECT_LEAN_VISIBILITY=1 INTEGRATE_DIRECT_SPLIT=2

//!variant integrate_direct_split_trace_lean_core_rayquery.comp
//!>       INTEGRATE_DIRECT_SSS_PROFILE=0 INTEGRATE_DIRECT_LEAN_VISIBILITY=1 VISIBILITY_CORE_MATERIAL=1 SURFACE_LEGACY_MODULE=0 INTEGRATE_DIRECT_SPLIT=2

//!variant integrate_direct_split_trace_lean_legacy_rayquery.comp
//!>       INTEGRATE_DIRECT_SSS_PROFILE=0 INTEGRATE_DIRECT_LEAN_VISIBILITY=1 VISIBILITY_CORE_MATERIAL=1 SURFACE_LEGACY_MODULE=1 INTEGRATE_DIRECT_SPLIT=2

// rtx.frameFeatures.validate. The Full kernels with the light-type seen bits (FRAME_FEATURES_VALIDATE),
// bound only while the option is on, so the variants above do not carry them.
//!variant integrate_direct_validate_rayquery.comp
//!>       FRAME_FEATURES_VALIDATE=1

//!variant integrate_direct_sun_validate_rayquery.comp
//!>       LIGHT_BVH_DEDICATED_SUN=1 FRAME_FEATURES_VALIDATE=1

//!end-variants
#define ATMOSPHERE_AVAILABLE
// The [LightSamplingStats] counters, gated at runtime by cb.lightSamplingStatsEnabled.
#define LIGHT_SAMPLING_STATS 1

#define USE_32BIT_RAY_DIRECTION 1
// Note: Use thin film in the direct integration pass for its usage in direct lighting (as this is where
// the effect is most prominent outside of PSR).
#define OPAQUE_MATERIAL_USE_THIN_FILM 1
#ifndef VISIBILITY_CORE_MATERIAL
#define OPAQUE_MATERIAL_USE_POM 1
#endif
#define RAB_HAS_RTXDI_RESERVOIRS 1
#define SHADOW_TERMINATOR_FIX_ALLOWED 1
// Slot of this pass in the rtx.alphaBake.softwareOmmCounters counters.
#define SOFT_OMM_COUNTER_PASS SOFT_OMM_COUNTER_PASS_DIRECT

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/integrate/integrate_direct_binding_indices.h"
#include "rtx/pass/integrate/integrate_direct_bindings.slangh"
// Note: Needed for declaration of integration ray payload structure
#include "rtx/algorithm/integrator.slangh"
#include "rtx/pass/nrc/nrc_integrator_utilities.slangh"

#include "rtx/pass/integrate/integrate_direct.slangh"

#ifdef RAY_PIPELINE
[shader("raygeneration")]
void main()
{
  integrate_direct_pass(DispatchRaysIndex().xy);
}
#elif INTEGRATE_DIRECT_SPLIT == 2
// The trace kernel of the split Lean levels.
[shader("compute")]
[numthreads(INTEGRATE_DIRECT_THREADS_DISPATCH_WIDTH, INTEGRATE_DIRECT_THREADS_DISPATCH_HEIGHT, 1)]
void main(uint2 dispatchThreadId : SV_DispatchThreadID)
{
  integrate_direct_trace_pass(dispatchThreadId);
}
#else
[shader("compute")]
[numthreads(INTEGRATE_DIRECT_THREADS_DISPATCH_WIDTH, INTEGRATE_DIRECT_THREADS_DISPATCH_HEIGHT, 1)]
void main(uint2 dispatchThreadId : SV_DispatchThreadID)
{
  integrate_direct_pass(dispatchThreadId);
}
#endif