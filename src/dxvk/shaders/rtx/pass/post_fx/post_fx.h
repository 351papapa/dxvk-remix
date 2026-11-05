/*
* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#define POST_FX_MOTION_BLUR_PREFILTER_PRIMARY_SURFACE_FLAGS_INPUT           0
#define POST_FX_MOTION_BLUR_PREFILTER_PRIMARY_SURFACE_FLAGS_FILTERED_OUTPUT 1

#define POST_FX_MOTION_BLUR_PRIMARY_SCREEN_SPACE_MOTION_INPUT 0
#define POST_FX_MOTION_BLUR_PRIMARY_SURFACE_FLAGS_INPUT       1
#define POST_FX_MOTION_BLUR_PRIMARY_LINEAR_VIEW_Z_INPUT       2
#define POST_FX_MOTION_BLUR_BLUE_NOISE_TEXTURE_INPUT          3
#define POST_FX_MOTION_BLUR_INPUT                             4
#define POST_FX_MOTION_BLUR_OUTPUT                            5
#define POST_FX_MOTION_BLUR_NEAREST_SAMPLER                   6
#define POST_FX_MOTION_BLUR_LINEAR_SAMPLER                    7

// Depth of field, pass 1: downsamples colour to half resolution with the CoC per texel,
// and reduces the per-tile maximum blur radii that classify tiles for the gather.
#define POST_FX_DOF_PREPARE_INPUT                              0
#define POST_FX_DOF_PREPARE_PRIMARY_LINEAR_VIEW_Z_INPUT        1
#define POST_FX_DOF_PREPARE_COLOR_COC_OUTPUT                   2
#define POST_FX_DOF_PREPARE_TILE_OUTPUT                        3
#define POST_FX_DOF_PREPARE_FOCUS_STATE_INPUT                  4
#define POST_FX_DOF_PREPARE_PRIMARY_SURFACE_FLAGS_INPUT        5

// Depth of field, pass 2: half resolution bokeh gather into a defocused base layer
// and a premultiplied near field layer. A tap costs one fetch of the colour and CoC input.
#define POST_FX_DOF_COLOR_COC_INPUT                            0
#define POST_FX_DOF_TILE_INPUT                                 1
#define POST_FX_DOF_NEAR_OUTPUT                                2
#define POST_FX_DOF_FAR_OUTPUT                                 3
#define POST_FX_DOF_LINEAR_SAMPLER                             4

// Depth of field, pass 3: full resolution composite. It recomputes the CoC per pixel, so in-focus
// pixels stay sharp, and composites the two half resolution layers.
#define POST_FX_DOF_RESOLVE_INPUT                              0
#define POST_FX_DOF_RESOLVE_PRIMARY_LINEAR_VIEW_Z_INPUT        1
#define POST_FX_DOF_RESOLVE_NEAR_INPUT                         2
#define POST_FX_DOF_RESOLVE_FAR_INPUT                          3
#define POST_FX_DOF_RESOLVE_OUTPUT                             4
#define POST_FX_DOF_RESOLVE_LINEAR_SAMPLER                     5
#define POST_FX_DOF_RESOLVE_FOCUS_STATE_INPUT                  6
#define POST_FX_DOF_RESOLVE_PRIMARY_SURFACE_FLAGS_INPUT        7

#define POST_FX_DOF_AF_PRIMARY_LINEAR_VIEW_Z_INPUT             0
#define POST_FX_DOF_AF_FOCUS_STATE_INPUT_OUTPUT                1

#define POST_FX_INPUT  0
#define POST_FX_OUTPUT 1

#define POST_FX_HIGHLIGHT_INPUT                       0
#define POST_FX_HIGHLIGHT_OBJECT_PICKING_INPUT        1
#define POST_FX_HIGHLIGHT_PRIMARY_CONE_RADIUS_INPUT   2
#define POST_FX_HIGHLIGHT_OUTPUT                      3
#define POST_FX_HIGHLIGHT_VALUES                      4

#define POST_FX_TILE_SIZE 8

// Depth of field constants, shared with rtx_postFx.cpp, which sizes the images and derives tileReach.

// Edge length of a classification tile in half resolution pixels. It is also the thread group edge
// of the prepare and gather passes, which run one group per tile.
#define POST_FX_DOF_TILE_SIZE    16
#define POST_FX_DOF_TILE_THREADS (POST_FX_DOF_TILE_SIZE * POST_FX_DOF_TILE_SIZE)

// How many tiles a near-field CoC may reach. The dilation window must cover
// ceil(maxGatherRadiusHalf / POST_FX_DOF_TILE_SIZE) tiles for classification to be conservative.
#define POST_FX_DOF_MAX_TILE_REACH   4
#define POST_FX_DOF_MAX_TILE_SAMPLES ((2 * POST_FX_DOF_MAX_TILE_REACH + 1) * (2 * POST_FX_DOF_MAX_TILE_REACH + 1))

// Largest gather radius the tile classification covers, in half resolution pixels.
// The host and shader both clamp to it, so no config can make the dilation unsound.
#define POST_FX_DOF_MAX_GATHER_RADIUS_HALF 64.0f

// Ceiling on gather taps per layer, applied in the shader and the UI so a bad config cannot hang the GPU.
#define POST_FX_DOF_MAX_SAMPLES 192
#define POST_FX_DOF_MIN_SAMPLES 8

// Floor that rtx_postFx.cpp raises rtx.dof.sampleCount to, so older configs with small counts
// do not give speckly bokeh. N taps fill a disc of radius sqrt(N / 0.8) half resolution pixels,
// so 32 covers 6.3, most of the default range. Small radii ask for fewer taps anyway.
#define POST_FX_DOF_MIN_EFFECTIVE_SAMPLES 32

// Blur radius in half resolution pixels below which a pixel counts as in focus.
// It is one full resolution pixel, below which a blur cannot be displayed.
#define POST_FX_DOF_MIN_RADIUS 0.5f

struct PostFxArgs {
  // Display image information
  uint2  imageSize;
  float2 invImageSize;

  // Camera Resolution
  float2 invMainCameraResolution;
  float2 inputOverOutputViewSize;

  // Post Fx Attributes
  // Motion Blur
  uint   motionBlurSampleCount;
  float  blurDiameterFraction;
  bool   enableMotionBlurNoiseSample;
  float  motionBlurMinimumVelocityThresholdInPixel;

  // Chromatic Aberration
  float2 chromaticAberrationScale;
  float  chromaticCenterAttenuationAmount;
  float  exposureFraction;

  // Vignette
  float  vignetteIntensity;
  float  vignetteRadius;
  float  vignetteSoftness;
  uint   frameIdx;

  float  motionBlurDynamicDeduction;
  bool   enableMotionBlurEmissive;
  float  jitterStrength;
  float  motionBlurDlfgDeduction;
};

// Shared by the three depth-of-field passes. Keep the 16B rows whole (see the asserts at the end).
struct PostFxDepthOfFieldArgs {
  // Full resolution colour image (m_finalOutput).
  uint2  imageSize;
  float2 invImageSize;

  // Half resolution working extent, ceil(imageSize / 2).
  uint2  halfImageSize;
  float2 invHalfImageSize;

  // Linear view Z is at render resolution, so pixel indices are scaled into it.
  uint2  linearViewZSize;
  float2 inputOverOutputViewSize;

  // Thin lens parameters. apertureTerm is f^2 / N, sensorToPixels projects
  // millimetres of blur disc through the 24 mm full-frame sensor height.
  float  focusDistance;
  float  focalLength;
  float  apertureTerm;
  float  worldUnitToMm;

  float  sensorToPixels;
  float  missLinearViewZ;
  // Blur radius of a normalised CoC of 1, clamped on the host to POST_FX_DOF_MAX_GATHER_RADIUS_HALF * 2.
  float  maxGatherRadius;
  float  maxGatherRadiusHalf;

  uint   sampleCount;
  uint   frameIdx;
  uint   autoFocusEnabled;
  // Tile dilation window half-width, in tiles; 0..POST_FX_DOF_MAX_TILE_REACH.
  uint   tileReach;

  float  autoFocusOffset;
  float  bokehMinIntensity;
  float  bokehFilterStrength;
  // Non-zero makes the resolve upsample the base layer blend with a colour-guided 2x2 bilateral filter.
  uint   edgeAwareUpsample;

  // Non-zero holds view model pixels at zero CoC. Prepare and resolve both read it,
  // because the resolve recomputes the blend from depth.
  uint   excludeViewModel;
  uint   pad0;
  uint   pad1;
  uint   pad2;
};

struct PostFxDofAutoFocusArgs {
  float2 focusPoint;
  float  missLinearViewZ;
  float  deltaTimeSecs;

  float  tauSeconds;
  uint   forceReset;
  float  regionRadius;
  float  deadZone;

  float  farTauScale;
  float  pad0;
  float  pad1;
  float  pad2;
};

struct PostFxMotionBlurPrefilterArgs {
  uint2 imageSize;
  int2  pixelStep;
};

#define POST_FX_HIGHLIGHTING_MAX_VALUES_POW 14
#define POST_FX_HIGHLIGHTING_MAX_VALUES     (1 << POST_FX_HIGHLIGHTING_MAX_VALUES_POW)
#define POST_FX_HIGHLIGHTING_INVALID_VALUE  0xFFFFFFFF

struct PostFxHighlightingArgs
{
  // Display image information
  uint2 imageSize;
  // If need to highlight an object under this pixel
  int2  pixel;
  // Highlighting params
  uint  desaturateNonHighlighted;
  float timeSinceStartMS;
  uint  highlightColorPacked;
  uint  valuesToHighlightCountPow;
};

#ifdef __cplusplus
// Every push constant struct here must be a whole number of 16B rows, because the shader uses
// scalar layout and a short row would shift every later member.
// PostFxArgs: 2 vector rows, then 4 scalar rows. Each `bool` is one byte in C++ and four in the shader,
// but C++ pads it to a 4B slot, so the layouts agree.
static_assert(sizeof(PostFxArgs) == 96,
              "PostFxArgs layout drift.");
static_assert(sizeof(PostFxArgs) % 16 == 0,
              "PostFxArgs must be a whole number of 16B rows.");

// 3 vector rows + 5 scalar rows.
static_assert(sizeof(PostFxDepthOfFieldArgs) == 128,
              "PostFxDepthOfFieldArgs layout drift.");
static_assert(sizeof(PostFxDepthOfFieldArgs) % 16 == 0,
              "PostFxDepthOfFieldArgs must be a whole number of 16B rows.");

static_assert(sizeof(PostFxDofAutoFocusArgs) == 48,
              "PostFxDofAutoFocusArgs layout drift.");
static_assert(sizeof(PostFxDofAutoFocusArgs) % 16 == 0,
              "PostFxDofAutoFocusArgs must be a whole number of 16B rows.");

static_assert(sizeof(PostFxMotionBlurPrefilterArgs) % 16 == 0,
              "PostFxMotionBlurPrefilterArgs must be a whole number of 16B rows.");
static_assert(sizeof(PostFxHighlightingArgs) % 16 == 0,
              "PostFxHighlightingArgs must be a whole number of 16B rows.");
#endif
