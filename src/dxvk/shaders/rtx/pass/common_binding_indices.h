/*
* Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

// These are set indices - not bindings
#define BINDING_SET_BINDLESS_RAW_BUFFER          1
#define BINDING_SET_BINDLESS_TEXTURE2D           2
#define BINDING_SET_BINDLESS_SAMPLER             3


#define BINDING_ACCELERATION_STRUCTURE           0
#define BINDING_ACCELERATION_STRUCTURE_PREVIOUS  1
#define BINDING_ACCELERATION_STRUCTURE_UNORDERED 2
#define BINDING_ACCELERATION_STRUCTURE_SSS       3
#define BINDING_SURFACE_DATA_BUFFER              4
#define BINDING_SURFACE_MAPPING_BUFFER           5
#define BINDING_SURFACE_MATERIAL_DATA_BUFFER     6
#define BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER 7
#define BINDING_VOLUME_MATERIAL_DATA_BUFFER      8
#define BINDING_LIGHT_DATA_BUFFER                9
#define BINDING_PREVIOUS_LIGHT_DATA_BUFFER       10
#define BINDING_LIGHT_MAPPING                    11
#define BINDING_BILLBOARDS_BUFFER                12
#define BINDING_BLUE_NOISE_TEXTURE               13
#define BINDING_BINDLESS_INDICES_BUFFER          14
#define BINDING_CONSTANTS                        15
#define BINDING_DEBUG_VIEW_TEXTURE               16
#define BINDING_GPU_PRINT_BUFFER                 17
#define BINDING_VALUE_NOISE_SAMPLER              18
#define BINDING_SAMPLER_READBACK_BUFFER          19
#define BINDING_LIGHT_IDENTITY_BUFFER            20
#define BINDING_WETNESS_ZONES 221
// Water generation passes bind their resources independently of the tracing bank.
#define BINDING_WATER_FOAM_SHEET_VERTICES 222
#define BINDING_WATER_FOAM_SURFACE_FIELD 223
#define BINDING_WATER_FOAM_SURFACE_HEIGHT 224
#define BINDING_SHORE_WETNESS_FIELD 235
#define BINDING_SHORE_WETNESS_HEIGHT 236
#define BINDING_WINDOW_ISLANDS                   222

// Sky provider (NUMOS, submodules/numos) outputs, at slots 200-217, clear of pass-local slots. The
// shader names are the ones numos/host/numos_host.slangh expects (its binding contract); passes that
// also bind their own resources (e.g. sparse rendering) assert no overlap with the range.
#define BINDING_SKY_ARGS                         200  // ConstantBuffer<AtmosphereArgs>, FrameOutputs::args
#define BINDING_ATMOSPHERE_TRANSMITTANCE_LUT     201
#define BINDING_ATMOSPHERE_MULTISCATTERING_LUT   202
#define BINDING_ATMOSPHERE_SKY_VIEW_LUT          203
#define BINDING_ATMOSPHERE_SKY_VIEW_SAMPLER      204
#define BINDING_ATMOSPHERE_CLOUD_SKY_TRANSMITTANCE_LUT 205
#define BINDING_ATMOSPHERE_CLOUD_SHADOW_MAP      206
#define BINDING_ATMOSPHERE_CLOUD_NOISE_SAMPLER   207
#define BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT   208
#define BINDING_NIGHT_SKY_STAR_CELLS             209
#define BINDING_NIGHT_SKY_STARS                  210
#define BINDING_NIGHT_SKY_MILKY_WAY              211
#define BINDING_NIGHT_SKY_STARLIGHT_LOW          212
#define BINDING_NIGHT_SKY_MOON                   213
// 214-217: free.
#define BINDING_ATMOSPHERE_MIN                   200
#define BINDING_ATMOSPHERE_MAX                   217

// Light BVH, StructuredBuffer<uint4>: 2 uint4 per compact node, then the light -> bit-trail table (4 per uint4).
// Read only when rtx.lightSampling.mode = LightBVH, and null otherwise. It also sits clear of pass-local slots.
#define BINDING_LIGHT_BVH_BUFFER                 218
// Light sampling statistics, RWStructuredBuffer<uint> counters that the direct integrator writes
// while rtx.lightSampling.logDirectStats is on. The host logs them as [LightSamplingStats].
#define BINDING_LIGHT_SAMPLING_STATS             219

#define COMMON_MAX_BINDING                       BINDING_LIGHT_IDENTITY_BUFFER
#define COMMON_NUM_BINDINGS                      (COMMON_MAX_BINDING + 1)

// Note: Used to represent a non-existent buffer
#define BINDING_INDEX_INVALID uint16_t(0xFFFF)

// Sentinel for an invalid surface index.  Equals the 21-bit maximum (SURFACE_INDEX_MAX_VALUE
// from instance_definitions.h) so that it fits inside the packed RayInteraction._surfaceAndFlags
// field.  The surfaceMapping buffer returns int32_t(-1) for unmapped surfaces; the 21-bit
// property setter truncates 0xFFFFFFFF to 0x1FFFFF automatically.
// This reserves the highest representable surface index as "invalid", reducing the usable
// range by one (max usable index = SURFACE_INDEX_MAX_VALUE - 1 = 2,097,150).
#define SURFACE_INDEX_INVALID 0x001FFFFFu

#define SAMPLER_FEEDBACK_INVALID           uint16_t(0xFFFF)
#define SAMPLER_FEEDBACK_MAX_TEXTURE_COUNT uint16_t(0xFFFF)

// Note: Light array may only be up to a size of 2^16-1, allowing the last index to be
// used for an invalid index similar to the max binding index for materials.
#define LIGHT_INDEX_INVALID (0xFFFF)

#ifdef __cplusplus

#define COMMON_RAYTRACING_BINDINGS \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE)            \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_UNORDERED)  \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_PREVIOUS)   \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_SSS)        \
  STRUCTURED_BUFFER(BINDING_SURFACE_DATA_BUFFER)                    \
  STRUCTURED_BUFFER(BINDING_SURFACE_MAPPING_BUFFER)                 \
  STRUCTURED_BUFFER(BINDING_SURFACE_MATERIAL_DATA_BUFFER)           \
  STRUCTURED_BUFFER(BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER)       \
  STRUCTURED_BUFFER(BINDING_WINDOW_ISLANDS)                       \
  TEXTURE2D(BINDING_SHORE_WETNESS_FIELD)                          \
  TEXTURE2D(BINDING_SHORE_WETNESS_HEIGHT)                         \
  STRUCTURED_BUFFER(BINDING_WETNESS_ZONES)                         \
  STRUCTURED_BUFFER(BINDING_VOLUME_MATERIAL_DATA_BUFFER)            \
  STRUCTURED_BUFFER(BINDING_LIGHT_DATA_BUFFER)                      \
  STRUCTURED_BUFFER(BINDING_PREVIOUS_LIGHT_DATA_BUFFER)             \
  STRUCTURED_BUFFER(BINDING_LIGHT_MAPPING)                          \
  STRUCTURED_BUFFER(BINDING_BILLBOARDS_BUFFER)                      \
  TEXTURE2DARRAY(BINDING_BLUE_NOISE_TEXTURE)                        \
  CONSTANT_BUFFER(BINDING_CONSTANTS)                                \
  RW_TEXTURE2D(BINDING_DEBUG_VIEW_TEXTURE)                          \
  RW_STRUCTURED_BUFFER(BINDING_GPU_PRINT_BUFFER)                    \
  SAMPLER3D(BINDING_VALUE_NOISE_SAMPLER)                            \
  RW_STRUCTURED_BUFFER(BINDING_SAMPLER_READBACK_BUFFER)             \
  STRUCTURED_BUFFER(BINDING_LIGHT_IDENTITY_BUFFER)             \
  STRUCTURED_BUFFER(BINDING_LIGHT_BVH_BUFFER)                       \
  RW_STRUCTURED_BUFFER(BINDING_LIGHT_SAMPLING_STATS)                \
  CONSTANT_BUFFER(BINDING_SKY_ARGS)                                 \
  TEXTURE2D(BINDING_ATMOSPHERE_TRANSMITTANCE_LUT)                   \
  TEXTURE2D(BINDING_ATMOSPHERE_MULTISCATTERING_LUT)                 \
  TEXTURE2D(BINDING_ATMOSPHERE_SKY_VIEW_LUT)                        \
  SAMPLER(BINDING_ATMOSPHERE_SKY_VIEW_SAMPLER)                      \
  TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_SKY_TRANSMITTANCE_LUT)         \
  TEXTURE3D(BINDING_ATMOSPHERE_CLOUD_SHADOW_MAP)                    \
  SAMPLER(BINDING_ATMOSPHERE_CLOUD_NOISE_SAMPLER)                   \
  TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT)                 \
  STRUCTURED_BUFFER(BINDING_NIGHT_SKY_STAR_CELLS)                   \
  STRUCTURED_BUFFER(BINDING_NIGHT_SKY_STARS)                        \
  STRUCTURED_BUFFER(BINDING_NIGHT_SKY_MILKY_WAY)                    \
  TEXTURE2D(BINDING_NIGHT_SKY_STARLIGHT_LOW)                        \
  TEXTURE2D(BINDING_NIGHT_SKY_MOON)

#endif
