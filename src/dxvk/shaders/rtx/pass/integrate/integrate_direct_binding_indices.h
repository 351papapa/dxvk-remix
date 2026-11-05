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
#pragma once

#include "rtx/pass/common_binding_indices.h"

// Thread group dimensions for Integrate Direct compute dispatch
#define INTEGRATE_DIRECT_THREADS_DISPATCH_WIDTH 16
#define INTEGRATE_DIRECT_THREADS_DISPATCH_HEIGHT 8

// Inputs
#define INTEGRATE_DIRECT_BINDING_SKYPROBE                                           39

#define INTEGRATE_DIRECT_BINDING_SHARED_INTEGRATION_SURFACE_PDF_INPUT               40
#define INTEGRATE_DIRECT_BINDING_SHARED_MATERIAL_DATA0_INPUT                        41
#define INTEGRATE_DIRECT_BINDING_SHARED_MATERIAL_DATA1_INPUT                        42
#define INTEGRATE_DIRECT_BINDING_SHARED_TEXTURE_COORD_INPUT                         43
#define INTEGRATE_DIRECT_BINDING_SHARED_SURFACE_INDEX_INPUT                         44
#define INTEGRATE_DIRECT_BINDING_SHARED_SUBSURFACE_DATA_INPUT                       45
#define INTEGRATE_DIRECT_BINDING_SHARED_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT     46
#define INTEGRATE_DIRECT_BINDING_SHARED_SHADOW_TERMINATOR_FIX_INPUT                 73

#define INTEGRATE_DIRECT_BINDING_PRIMARY_WORLD_SHADING_NORMAL_INPUT                 47
#define INTEGRATE_DIRECT_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_INPUT                 48
#define INTEGRATE_DIRECT_BINDING_PRIMARY_ALBEDO_INPUT                               49
#define INTEGRATE_DIRECT_BINDING_PRIMARY_VIEW_DIRECTION_INPUT                       50
#define INTEGRATE_DIRECT_BINDING_PRIMARY_CONE_RADIUS_INPUT                          51
#define INTEGRATE_DIRECT_BINDING_PRIMARY_WORLD_POSITION_WORLD_TRIANGLE_NORMAL_INPUT 52
#define INTEGRATE_DIRECT_BINDING_PRIMARY_POSITION_ERROR_INPUT                       53
#define INTEGRATE_DIRECT_BINDING_PRIMARY_RTXDI_RESERVOIR                            54

#define INTEGRATE_DIRECT_BINDING_SECONDARY_WORLD_SHADING_NORMAL_INPUT               55
#define INTEGRATE_DIRECT_BINDING_SECONDARY_ALBEDO_INPUT                             56
#define INTEGRATE_DIRECT_BINDING_SECONDARY_VIEW_DIRECTION_INPUT                     57
#define INTEGRATE_DIRECT_BINDING_SECONDARY_CONE_RADIUS_INPUT                        58

#define INTEGRATE_DIRECT_BINDING_NEE_CACHE                                          59
#define INTEGRATE_DIRECT_BINDING_NEE_CACHE_THREAD_TASK                              60
#define INTEGRATE_DIRECT_BINDING_NEE_CACHE_TASK                                     61


#define INTEGRATE_DIRECT_BINDING_ACTIVE_LOCAL_PIXEL_COORDS_INPUT                    65
#define INTEGRATE_DIRECT_BINDING_ACTIVE_PIXEL_LIST_INPUT                            66

#define INTEGRATE_DIRECT_BINDING_NRC_TRAINING_QUERY_RESERVOIR_INPUT                     69
#define INTEGRATE_DIRECT_BINDING_PRIMARY_HIT_DISTANCE_INPUT                         70
#define INTEGRATE_DIRECT_BINDING_SECONDARY_HIT_DISTANCE_INPUT                       71

// Inputs/Outputs

#define INTEGRATE_DIRECT_BINDING_SHARED_FLAGS_INPUT_OUTPUT                          90
#define INTEGRATE_DIRECT_BINDING_SHARED_MEDIUM_MATERIAL_INDEX_INPUT_OUTPUT          91
#define INTEGRATE_DIRECT_BINDING_PRIMARY_BASE_REFLECTIVITY_INPUT_OUTPUT             92
#define INTEGRATE_DIRECT_BINDING_SECONDARY_BASE_REFLECTIVITY_INPUT_OUTPUT           93

// Outputs

#define INTEGRATE_DIRECT_BINDING_PRIMARY_DIRECT_DIFFUSE_LOBE_RADIANCE_OUTPUT        94
#define INTEGRATE_DIRECT_BINDING_PRIMARY_DIRECT_SPECULAR_LOBE_RADIANCE_OUTPUT       95
#define INTEGRATE_DIRECT_BINDING_SECONDARY_COMBINED_DIFFUSE_LOBE_RADIANCE_OUTPUT    96
#define INTEGRATE_DIRECT_BINDING_SECONDARY_COMBINED_SPECULAR_LOBE_RADIANCE_OUTPUT   97

#define INTEGRATE_DIRECT_BINDING_PRIMARY_RTXDI_ILLUMINANCE_OUTPUT                   98

#define INTEGRATE_DIRECT_BINDING_INDIRECT_THROUGHPUT_CONE_RADIUS_OUTPUT             99

#define INTEGRATE_DIRECT_BINDING_NEE_CACHE_SAMPLE                                   100

#define INTEGRATE_DIRECT_BINDING_NRC_QUERY_PATH_INFO_OUTPUT                         101
#define INTEGRATE_DIRECT_BINDING_NRC_TRAINING_PATH_INFO_OUTPUT                      102
#define INTEGRATE_DIRECT_BINDING_NRC_TRAINING_PATH_VERTICES_OUTPUT                  103
#define INTEGRATE_DIRECT_BINDING_NRC_QUERY_RADIANCE_PARAMS_OUTPUT                   104
#define INTEGRATE_DIRECT_BINDING_NRC_COUNTERS_OUTPUT                                105
#define INTEGRATE_DIRECT_BINDING_NRC_QUERY_PATH_DATA0_OUTPUT                        106
#define INTEGRATE_DIRECT_BINDING_NRC_TRAINING_PATH_DATA1_OUTPUT                     107

// The per-pixel record the split shade kernel writes and the split trace kernel reads.
#define INTEGRATE_DIRECT_BINDING_SPLIT_RECORDS                                      108

// The record layout: uint4 planes of (camera width * height) entries each, so neighbouring pixels are adjacent.
// Plane 0 is written for every pixel whose selected surface is a hit, the others only as the header flags say.
// Pixels whose selected surface missed get nothing, and the trace kernel skips them by the same G-buffer test.
// The comments on the plane defines below give the packing of each plane.
#define INTEGRATE_DIRECT_SPLIT_PLANE_LIGHT     0  // x: header, yzw: local light sample position (fp32)
#define INTEGRATE_DIRECT_SPLIT_PLANE_RADIANCE  1  // xyz: local diffuse rgb + specular rgb (fp16), w: illuminance or origin position error
#define INTEGRATE_DIRECT_SPLIT_PLANE_SUN       2  // xyz: sun diffuse rgb + specular rgb (fp16), w: sun direction (snorm16x2 octahedral)
// A secondary integration surface's G-buffer origin is aliased with outputs the shade kernel writes, so the record carries
// it. The sun sample is taken on primary surfaces only, so the two never meet.
#define INTEGRATE_DIRECT_SPLIT_PLANE_ORIGIN    2  // encoded world position / triangle TBN, aliasing the sun plane
#define INTEGRATE_DIRECT_SPLIT_PLANE_COUNT     3

// The header: these flags, the sampled transport portal index << 8 and the light index << 16.
#define INTEGRATE_DIRECT_SPLIT_FLAG_LOCAL      (1u << 0) // a valid local light sample (RTXDI / RIS / light BVH)
#define INTEGRATE_DIRECT_SPLIT_FLAG_RTXDI      (1u << 1) // the local sample came from the RTXDI reservoir
#define INTEGRATE_DIRECT_SPLIT_FLAG_SUN        (1u << 2) // a valid dedicated sun sample
#define INTEGRATE_DIRECT_SPLIT_FLAG_ORIGIN     (1u << 3) // the origin plane holds the secondary surface's origin

// Aliased Inputs/Outputs

#define INTEGRATE_DIRECT_BINDING_ALIASED_DATA_0   150
#define INTEGRATE_DIRECT_BINDING_SECONDARY_WORLD_POSITION_WORLD_TRIANGLE_NORMAL_INPUT (INTEGRATE_DIRECT_BINDING_ALIASED_DATA_0 + 0)
#define INTEGRATE_DIRECT_BINDING_INDIRECT_RAY_ORIGIN_DIRECTION_OUTPUT                 (INTEGRATE_DIRECT_BINDING_ALIASED_DATA_0 + 1)
#define INTEGRATE_DIRECT_BINDING_ALIASED_DATA_1   152
#define INTEGRATE_DIRECT_BINDING_SECONDARY_PERCEPTUAL_ROUGHNESS_INPUT             (INTEGRATE_DIRECT_BINDING_ALIASED_DATA_1 + 0)
#define INTEGRATE_DIRECT_BINDING_INDIRECT_FIRST_HIT_PERCEPTUAL_ROUGHNESS_OUTPUT   (INTEGRATE_DIRECT_BINDING_ALIASED_DATA_1 + 1)
#define INTEGRATE_DIRECT_BINDING_ALIASED_DATA_2   154
#define INTEGRATE_DIRECT_BINDING_SECONDARY_POSITION_ERROR_INPUT            (INTEGRATE_DIRECT_BINDING_ALIASED_DATA_2 + 0)
#define INTEGRATE_DIRECT_BINDING_INDIRECT_FIRST_SAMPLED_LOBE_DATA_OUTPUT   (INTEGRATE_DIRECT_BINDING_ALIASED_DATA_2 + 1)

#define INTEGRATE_DIRECT_MIN_BINDING                           INTEGRATE_DIRECT_BINDING_SKYPROBE

#if INTEGRATE_DIRECT_MIN_BINDING <= COMMON_MAX_BINDING
#error "Increase the base index of Integrate Direct bindings to avoid overlap with common bindings!"
#endif
