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

// rtx.coreMaterial.gpuCheck. One thread per surface decodes the full records (surface material, Surface alpha and flag
// bits, extension tint) with the existing Slang structures and the core record with CoreSurfaceMaterial, and counts the
// surfaces where they disagree.

#define CORE_MATERIAL_CHECK_BINDING_SURFACES        0
#define CORE_MATERIAL_CHECK_BINDING_OLD_MATERIALS   1
#define CORE_MATERIAL_CHECK_BINDING_EXTENSIONS      2
#define CORE_MATERIAL_CHECK_BINDING_CORE_MATERIALS  3
#define CORE_MATERIAL_CHECK_BINDING_COUNTERS        4
#define CORE_MATERIAL_CHECK_BINDING_CORE_EXTENSIONS 5 // core side table
#define CORE_MATERIAL_CHECK_BINDING_LEGACY_SURFACES 6 // legacy records

#define CORE_MATERIAL_CHECK_THREADS_PER_GROUP       64

// The counter buffer (uint32) holds mismatching surfaces, OR of CORE_MATERIAL_CHECK_FIELD_*, surfaces checked and the first
// mismatching surface (0xFFFFFFFF = none).
#define CORE_MATERIAL_CHECK_COUNTER_MISMATCHES      0
#define CORE_MATERIAL_CHECK_COUNTER_FIELDS          1
#define CORE_MATERIAL_CHECK_COUNTER_SURFACES        2
#define CORE_MATERIAL_CHECK_COUNTER_FIRST           3
#define CORE_MATERIAL_CHECK_COUNTER_COUNT           4

#define CORE_MATERIAL_CHECK_FIELD_FLAGS             (1u << 0)
#define CORE_MATERIAL_CHECK_FIELD_SAMPLER           (1u << 1)
#define CORE_MATERIAL_CHECK_FIELD_TEXTURES          (1u << 2)
#define CORE_MATERIAL_CHECK_FIELD_FEEDBACK_STAMP    (1u << 3)
#define CORE_MATERIAL_CHECK_FIELD_ALBEDO_OPACITY    (1u << 4)
#define CORE_MATERIAL_CHECK_FIELD_EMISSIVE          (1u << 5)
#define CORE_MATERIAL_CHECK_FIELD_ROUGH_METAL       (1u << 6) // roughness, metallic, anisotropy, roughness floor
#define CORE_MATERIAL_CHECK_FIELD_DLSS_MASK         (1u << 7)
#define CORE_MATERIAL_CHECK_FIELD_SPECULAR_METAL    (1u << 8)
#define CORE_MATERIAL_CHECK_FIELD_EXTENSION_INDEX   (1u << 9)
#define CORE_MATERIAL_CHECK_FIELD_ALBEDO_TINT       (1u << 10)
#define CORE_MATERIAL_CHECK_FIELD_DECAL_SORT_ORDER  (1u << 11)
#define CORE_MATERIAL_CHECK_FIELD_ALPHA             (1u << 12) // alpha test / blend state and its derived mode
#define CORE_MATERIAL_CHECK_FIELD_SURFACE_FLAGS     (1u << 13) // emissive, matte, animated water, material changed, ...
#define CORE_MATERIAL_CHECK_FIELD_CLASS             (1u << 14)
#define CORE_MATERIAL_CHECK_FIELD_TRANSLUCENT       (1u << 15) // class 1 bytes 0-35 / spare bytes
#define CORE_MATERIAL_CHECK_FIELD_PORTAL            (1u << 16) // ray portal bytes 0-55
#define CORE_MATERIAL_CHECK_FIELD_CORE_EXTENSION    (1u << 17) // core side table entry contents
#define CORE_MATERIAL_CHECK_FIELD_LEGACY            (1u << 18) // legacy record vs the Surface words
#define CORE_MATERIAL_CHECK_FIELD_LEGACY_BITS       (1u << 19) // tail bits hasLegacy / hasTextureOps vs the Surface words

struct CoreMaterialCheckArgs
{
  uint32_t surfaceCount;
  uint32_t extensionCount;
  uint32_t coreExtensionCount; // core side table entries
  uint32_t legacySurfaceCount; // legacy records (0 = none this frame)
};
