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

// The layout of the 64 B per-surface core material record ("Core Material Buffer", rtx.coreMaterial.build).
// The host writer (RtSurfaceMaterial::writeCoreGPUData, rtx_materials.h, which
// static_asserts its struct offsets against these values) and the Slang decoder (core_surface_material.slangh) share it.
// Bytes 0-55 depend on the class, and bytes 56-63 are common to all classes.

#define CORE_MATERIAL_SIZE 64

// Class 0 (opaque PBR) and class 2 (skin SSS) layout, byte offsets. Half floats are IEEE binary16 bit patterns,
// encoded exactly as in the OpaqueSurfaceMaterial record (surface_material.h).
#define CORE_MATERIAL_OFFSET_FLAGS                    0  // u16, OPAQUE_SURFACE_MATERIAL_FLAG_* verbatim (2 type bits + 14 flags)
#define CORE_MATERIAL_OFFSET_SAMPLER                  2  // u16
#define CORE_MATERIAL_OFFSET_ALBEDO_OPACITY_TEXTURE   4  // u16
#define CORE_MATERIAL_OFFSET_NORMAL_TEXTURE           6  // u16
#define CORE_MATERIAL_OFFSET_ROUGHNESS_TEXTURE        8  // u16
#define CORE_MATERIAL_OFFSET_METALLIC_TEXTURE         10 // u16
#define CORE_MATERIAL_OFFSET_EMISSIVE_TEXTURE         12 // u16
#define CORE_MATERIAL_OFFSET_SAMPLER_FEEDBACK_STAMP   14 // u16
#define CORE_MATERIAL_OFFSET_ALBEDO_OPACITY           16 // 4 x f16
#define CORE_MATERIAL_OFFSET_EMISSIVE_COLOR           24 // 3 x f16
#define CORE_MATERIAL_OFFSET_EMISSIVE_INTENSITY       30 // f16 (pre-scaled, as in OpaqueSurfaceMaterial)
#define CORE_MATERIAL_OFFSET_ROUGHNESS                32 // f16
#define CORE_MATERIAL_OFFSET_METALLIC                 34 // f16
#define CORE_MATERIAL_OFFSET_ANISOTROPY               36 // f16
#define CORE_MATERIAL_OFFSET_ROUGHNESS_FLOOR          38 // f16
#define CORE_MATERIAL_OFFSET_DLSS_CONTROL_MASK        40 // u32
#define CORE_MATERIAL_OFFSET_SPECULAR_METAL           44 // u32: specularMetalParams (low 16) | specularMetalTint (high 16)
#define CORE_MATERIAL_OFFSET_ALBEDO_TINT              48 // 3 x f16 (1.0 when the material has no tint)
#define CORE_MATERIAL_OFFSET_DECAL_SORT_ORDER         54 // u8 (the surface record keeps 8 bits too)
#define CORE_MATERIAL_OFFSET_ALPHA                    56 // u32, CORE_MATERIAL_ALPHA_* (common to all classes)
#define CORE_MATERIAL_OFFSET_TAIL                     60 // u32, CORE_MATERIAL_TAIL_* (common to all classes)

// Class 1 (translucent): bytes 0-35 are the TranslucentSurfaceMaterial record verbatim, and 36-55 are zero.
// Class 3 is only a fallback, for ray portals and raytraced render targets. A ray portal carries bytes 0-55 of its
// full record verbatim, and a raytraced-render-target opaque material uses the class 0 layout.
#define CORE_MATERIAL_TRANSLUCENT_BYTES               36
#define CORE_MATERIAL_CLASS_BYTES                     56

// Alpha / surface dword (byte 56).
#define CORE_MATERIAL_ALPHA_MODE_SHIFT                0  // 2 bits, CORE_MATERIAL_ALPHA_MODE_*
#define CORE_MATERIAL_ALPHA_MODE_MASK                 0x3u
#define CORE_MATERIAL_ALPHA_TEST_TYPE_SHIFT           2  // 3 bits, AlphaTestType
#define CORE_MATERIAL_ALPHA_TEST_TYPE_MASK            0x7u
#define CORE_MATERIAL_ALPHA_TEST_REFERENCE_SHIFT      5  // 8 bits
#define CORE_MATERIAL_ALPHA_TEST_REFERENCE_MASK       0xFFu
#define CORE_MATERIAL_ALPHA_BLEND_TYPE_SHIFT          13 // 4 bits, BlendType
#define CORE_MATERIAL_ALPHA_BLEND_TYPE_MASK           0xFu
#define CORE_MATERIAL_ALPHA_FULLY_OPAQUE              (1u << 17)
#define CORE_MATERIAL_ALPHA_BLENDING_DISABLED         (1u << 18)
#define CORE_MATERIAL_ALPHA_EMISSIVE_BLEND            (1u << 19)
#define CORE_MATERIAL_ALPHA_PARTICLE                  (1u << 20)
#define CORE_MATERIAL_ALPHA_DECAL                     (1u << 21)
#define CORE_MATERIAL_ALPHA_INVERTED_BLEND            (1u << 22)
#define CORE_MATERIAL_SURFACE_EMISSIVE                (1u << 23)
#define CORE_MATERIAL_SURFACE_MATTE                   (1u << 24)
#define CORE_MATERIAL_SURFACE_ANIMATED_WATER          (1u << 25)
#define CORE_MATERIAL_SURFACE_MATERIAL_CHANGED        (1u << 26)
#define CORE_MATERIAL_SURFACE_MOTION_BLUR_MASK_OUT    (1u << 27)
#define CORE_MATERIAL_SURFACE_PRESERVE_PATH           (1u << 28)

// alphaMode is decal when the surface is a decal, then blend when blending is on, then test for an alpha test other
// than "always", and opaque otherwise.
#define CORE_MATERIAL_ALPHA_MODE_OPAQUE               0u
#define CORE_MATERIAL_ALPHA_MODE_TEST                 1u
#define CORE_MATERIAL_ALPHA_MODE_BLEND                2u
#define CORE_MATERIAL_ALPHA_MODE_DECAL                3u

// Tail dword (byte 60).
// The extension index is the entry of this surface's material in the core side table ("Core Material Extension
// Buffer", cb.coreMaterialExtTable), not the surface material extension cache index.
#define CORE_MATERIAL_TAIL_EXTENSION_INDEX_MASK       0x1FFFFFu // 21 bits; SURFACE_INDEX_INVALID (0x1FFFFF) = none
#define CORE_MATERIAL_TAIL_CLASS_SHIFT                21 // 2 bits, CORE_MATERIAL_CLASS_*
#define CORE_MATERIAL_TAIL_CLASS_MASK                 0x3u
#define CORE_MATERIAL_TAIL_TWO_SIDED                  (1u << 23) // copy of the instance's CULL_DISABLE flag
// Fixed-function state is split in two bits. HAS_TEXTURE_OPS covers texture-stage ops other than the defaults, or an
// argument reading TFactor or vertex colour (the TextureOps part of the legacy record, bytes 48-55). HAS_LEGACY covers
// everything else (texgen, texture transform, spritesheet, clip plane, eye, texture-factor blend).
#define CORE_MATERIAL_TAIL_HAS_LEGACY                 (1u << 24) // non-identity fixed-function state other than texture ops
#define CORE_MATERIAL_TAIL_HAS_ALBEDO_TINT            (1u << 25)
#define CORE_MATERIAL_TAIL_HAS_TEXTURE_OPS            (1u << 26) // non-default texture-stage ops
#define CORE_MATERIAL_TAIL_SPARE_SHIFT                27         // bits 27-31 spare (zero)

#define CORE_MATERIAL_CLASS_OPAQUE                    0u
#define CORE_MATERIAL_CLASS_TRANSLUCENT               1u
#define CORE_MATERIAL_CLASS_SKIN_SSS                  2u
#define CORE_MATERIAL_CLASS_FALLBACK                  3u

// A core side table entry is 64 B, one per unique opaque material that needs one: a subsurface extension
// (diffusion-profile SSS or thin opaque), POM, a thin film or a tangent texture. Tint-only extensions need none,
// because the tint is in the core record.
#define CORE_MATERIAL_EXT_SIZE                        64
#define CORE_MATERIAL_EXT_OFFSET_SUBSURFACE           0  // 32 B: the SubsurfaceMaterial record's first 32 B verbatim
                                                         // (or CORE_MATERIAL_EXT_NO_EXTENSION_* when the material has none)
#define CORE_MATERIAL_EXT_SUBSURFACE_BYTES            32
#define CORE_MATERIAL_EXT_OFFSET_TANGENT_TEXTURE      32 // u16
#define CORE_MATERIAL_EXT_OFFSET_HEIGHT_TEXTURE       34 // u16 (as in the full record: m_heightTextureIndex even without POM)
#define CORE_MATERIAL_EXT_OFFSET_DISPLACE_IN          36 // f16 (scaled, 0 without POM, as in the full record)
#define CORE_MATERIAL_EXT_OFFSET_DISPLACE_OUT         38 // f16
#define CORE_MATERIAL_EXT_OFFSET_THIN_FILM_THICKNESS  40 // f16 (normalized, as in the full record)
#define CORE_MATERIAL_EXT_USED_BYTES                  42 // 42-63 spare (zero)
// Bytes 0-31 of an entry whose material has no extension hold SURFACE_MATERIAL_EXTENSION_FLAG_NO_SUBSURFACE flags and
// the three subsurface texture indices invalid (0xFFFF). Everything else is zero.
#define CORE_MATERIAL_EXT_NO_EXTENSION_TEXTURE        0xFFFFu
