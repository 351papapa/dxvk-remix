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

// rtx.frameFeatures.validate. The validation variants of the Full kernels (FRAME_FEATURES_VALIDATE) OR these
// "feature seen" bits into LightSamplingStats after its 64 statistics slots. The host logs
// [FrameFeatures] VIOLATION when a seen bit's mask bit is clear.

#define FRAME_FEATURES_SEEN_SLOT              64 // uint index in LightSamplingStats
#define FRAME_FEATURES_SEEN_SLOTS             4  // slots reserved (one used)

#define FRAME_FEATURES_SEEN_THIN_OPAQUE       (1u << 0) // a primary hit on a thin-opaque subsurface material
#define FRAME_FEATURES_SEEN_THIN_FILM         (1u << 1) // a primary hit with thin-film thickness > 0
#define FRAME_FEATURES_SEEN_PORTAL            (1u << 2) // a primary ray resolved through a ray portal
#define FRAME_FEATURES_SEEN_POM               (1u << 3) // a primary hit on a POM surface with POM on
#define FRAME_FEATURES_SEEN_SSS_PROFILE       (1u << 4) // a primary hit on a diffusion-profile SSS material
#define FRAME_FEATURES_SEEN_UNORDERED         (1u << 5) // a primary ray with an unordered (particle / decal) interaction
#define FRAME_FEATURES_SEEN_LEGACY            (1u << 6) // a primary hit whose surface takes a legacy op
#define FRAME_FEATURES_SEEN_EYE               (1u << 7) // a primary hit on an eye surface
#define FRAME_FEATURES_SEEN_LIGHT_TYPE_SHIFT  8         // + light type (lightTypeSphere .. lightTypeDistant): a valid
                                                        // primary NEE light sample of that type (direct pass)
#define FRAME_FEATURES_SEEN_LIGHT_TYPE_COUNT  5
