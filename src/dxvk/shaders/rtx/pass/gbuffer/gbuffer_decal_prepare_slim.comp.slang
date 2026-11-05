//!feature wetness ENABLE_SURFACE_WETNESS
//!feature particle_sky ENABLE_PARTICLE_SKY_HEMISPHERE
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

/* rtx.gbuffer.slimVariants. The deferred unordered decal kernel (gbuffer_decal_prepare.comp.slang, non-debug) with the
   slim and slimpom levels of gbuffer_rayquery_slim.comp.slang. The lean levels bind these too, because
   GBUFFER_SLIM_NO_SSS compiles to the same binary here. */

//!variant gbuffer_decal_prepare_slim.comp
//!>       GBUFFER_SLIM_NO_POM GBUFFER_SLIM_NO_THIN_FILM GBUFFER_SLIM_NO_PORTALS

//!variant gbuffer_decal_prepare_slimpom.comp
//!>       GBUFFER_SLIM_NO_THIN_FILM GBUFFER_SLIM_NO_PORTALS

// The deferred unordered particle resolve: the same kernels plus the primary rays' unordered particle walk,
// once per pixel over the whole camera-to-hit ray (geometryResolverDeferredUnorderedParticles). Like the slim primary
// kernels they are WBOIT only, and only slim levels have them, since without portals the camera ray is the primary ray.
//!variant gbuffer_decal_prepare_particles_wboit_slim.comp
//!>       GBUFFER_SLIM_NO_POM GBUFFER_SLIM_NO_THIN_FILM GBUFFER_SLIM_NO_PORTALS UNORDERED_RESOLVE_WBOIT GBUFFER_DECAL_PARTICLES=1

//!variant gbuffer_decal_prepare_particles_wboit_slimpom.comp
//!>       GBUFFER_SLIM_NO_THIN_FILM GBUFFER_SLIM_NO_PORTALS UNORDERED_RESOLVE_WBOIT GBUFFER_DECAL_PARTICLES=1

//!end-variants

#include "rtx/pass/gbuffer/gbuffer_decal_prepare.comp.slang"
