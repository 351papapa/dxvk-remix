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
#pragma once

#include "dxvk_format.h"
#include "dxvk_include.h"
#include "dxvk_context.h"
#include "rtx_objectpicking.h"
#include "rtx_resources.h"

#include "../spirv/spirv_code_buffer.h"
#include "../util/util_matrix.h"
#include "rtx_options.h"

namespace dxvk {

  class DxvkDevice;

  class DxvkPostFx {
  public:
    DxvkPostFx(DxvkDevice* device);
    ~DxvkPostFx();

    // Motion blur phase. Runs before tonemapping while the image is still in linear HDR space.
    // Reads m_finalOutput, writes back to m_finalOutput (via intermediate texture).
    void dispatchMotionBlur(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> nearestSampler,
      Rc<DxvkSampler> linearSampler,
      const uvec2& mainCameraResolution,
      const uint32_t frameIdx,
      const Resources::RaytracingOutput& rtOutput,
      const bool cameraCutDetected);

    // Depth of field phase. Runs before tonemapping while the image is still
    // in linear HDR space and reads the render-resolution linear view-Z.
    void dispatchDof(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      const uvec2& mainCameraResolution,
      const uint32_t frameIdx,
      const float missLinearViewZ,
      const Resources::RaytracingOutput& rtOutput,
      const float frameTimeMilliseconds,
      const bool cameraCutDetected);

    // Lens effects phase (chromatic aberration + vignette). Runs after tonemapping
    // so it operates on post-tonemap LDR data — these are display-space lens artifacts.
    // Reads and writes m_finalOutput in place.
    void dispatchLensEffects(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      const uvec2& mainCameraResolution,
      const uint32_t frameIdx,
      const Resources::RaytracingOutput& rtOutput);

    // NTSC/VHS composite. Runs after tonemapping and writes the processed
    // display-space result back to m_finalOutput.
    //
    // No frame index parameter on purpose: the tape model keys its random
    // streams off integer NTSC frame/field counters derived from GlobalTime, so
    // the look is frame-rate independent and does not freeze when the caller's
    // frame index is pinned to zero by rtx.rngSeedWithFrameIndex.
    void dispatchNtsc(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      const uvec2& mainCameraResolution,
      const Resources::RaytracingOutput& rtOutput);

    void dispatchHighlighting(
      Rc<RtxContext> ctx,
      const Resources::RaytracingOutput& rtOutput,
      std::vector<uint32_t>&& objectPickingValuesToHighlight,
      const std::optional<Vector2i>& pixelToHighlight,
      HighlightColor color);

    void showMotionBlurImguiSettings();
    void showDofImguiSettings();
    void showLensEffectsImguiSettings();
    void showLensRainImguiSettings();
    void dispatchLensRain(Rc<RtxContext> ctx, const Resources::RaytracingOutput& rtOutput);
    void showNtscImguiSettings();

    inline bool isPostFxEnabled() const { return enable(); }
    inline bool isMotionBlurEnabled() const { return enable() && enableMotionBlur() && motionBlurSampleCount() > 0 && exposureFraction() > 0.0f; }
    inline bool isDofEnabled() const { return enable() && dofEnable() && sampleCount() > 0; }
    inline bool isDofAutoFocusEnabled() const { return isDofEnabled() && autoFocusEnable(); }
    inline bool isChromaticAberrationEnabled() const { return enable() && enableLensEffects() && enableChromaticAberration() && chromaticAberrationAmount() > 0.0f; }
    inline bool isVignetteEnabled() const { return enable() && enableLensEffects() && enableVignette() && vignetteIntensity() > 0.0f; }
    inline const Resources::Resource& getDofFocusState() const { return m_dofFocusState; }

    RTX_OPTION_ARGS("rtx.postfx", bool, enable, true, "Enables optional post-processing effects in the stack.",
                    args.environment = "RTX_POST_FX_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.postfx", bool, enableMotionBlur, true, "Enables motion blur post-processing effect.",
                    args.environment = "RTX_POST_FX_MOTION_BLUR_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.postfx", bool, enableLensEffects, true, "Enables lens post-processing effects.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.postfx", bool, enableChromaticAberration, true, "Enables chromatic aberration post-processing effect.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.postfx", bool, enableVignette, true, "Enables vignette post-processing effect.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx.lensRain", bool, lensRainEnable, false, "Enable screen-space lens droplets driven by the active rain/storm weather. Runs after upscaling and tonemapping.");
    RTX_OPTION("rtx.lensRain", float, forceAmount, -1.0f, "-1: follow weather; 0..1: force a preview amount. Shelter still applies unless respectShelter is disabled.");
    RTX_OPTION("rtx.lensRain", bool, respectShelter, false, "Fade lens rain inside published no-rain cullzones or when the game disables interior precipitation, including forced previews.");
    RTX_OPTION("rtx.lensRain", float, drySeconds, 2.5f, "Time to remove 95 percent of the lens rain after entering shelter or rain stopping.");
    RTX_OPTION("rtx.lensRain", float, wetSeconds, 0.8f, "Time to reach 95 percent of the target lens rain when becoming exposed.");
    RTX_OPTION("rtx.lensRain", float, motionResponse, 0.025f, "Screen-height flow per meter of player or vehicle travel. Independent drops slide at different rates; camera motion is used only if no motion bridge has ever been published.");
    RTX_OPTION("rtx.lensRain", float, motionThreshold, 2.0f, "Player or vehicle speed in meters/second below which speed-driven sliding is suppressed.");
    RTX_OPTION("rtx.lensRain", float, amountScale, 1.0f, "Multiplier for weather-driven lens rain coverage.");
    RTX_OPTION("rtx.lensRain", float, sizeScale, 1.5f, "Independent drop radius multiplier.");
    RTX_OPTION("rtx.lensRain", float, density, 14.0f, "Controls independent droplet population and base radius. Smaller values produce larger, fewer drops; Size Scale changes radius independently.");
    RTX_OPTION("rtx.lensRain", int, dropCount, 0, "0: automatic population from Density and aspect ratio; 1..1024: independent simulated drop slots without changing radius. Visible count varies with lifetime, screen clipping and shelter. Negative values use automatic population.");
    RTX_OPTION("rtx.lensRain", float, refractPixels, 18.0f, "Maximum lens distortion in pixels at 1080p, scaled to output height.");
    RTX_OPTION("rtx.lensRain", float, slideSpeed, 1.0f, "Droplet lifecycle and downward sliding speed. Zero freezes this motion; camera-driven flow is controlled separately by motionResponse.");
    RTX_OPTION("rtx.lensRain", float, dispersion, 0.12f, "Fractional red/blue separation around the refracted sample.");
    RTX_OPTION("rtx.lensRain", float, blurPixels, 0.8f, "Small wet-lens blur radius in pixels at 1080p.");
    RTX_OPTION("rtx.postfx", bool, desaturateOthersOnHighlight, true, "If true, desaturare all objects that are not highlighted.");

    RTX_OPTION_ARGS("rtx.dof", bool, dofEnable, false,
                    "Enable the depth-of-field effect.",
                    args.environment = "RTX_DOF_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", bool, autoFocusEnable, false,
                    "Measure and smoothly track the depth-of-field focus distance from the screen.",
                    args.environment = "RTX_DOF_AUTO_FOCUS_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, autoFocusTau, 0.25f,
                    "Auto-focus smoothing time constant in seconds when focus moves to a nearer distance.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, autoFocusFarTauScale, 3.0f,
                    "Multiplier applied to the auto-focus smoothing time constant when focus moves to a farther distance.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, autoFocusDeadZone, 0.03f,
                    "Relative optical-power change below which auto-focus holds the current focus distance.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, autoFocusRegionRadius, 0.02f,
                    "Normalized radius (fraction of the smaller image dimension) of the disk sampled around the auto-focus point.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, autoFocusPointX, 0.5f,
                    "Normalized horizontal screen position used for auto-focus.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, autoFocusPointY, 0.5f,
                    "Normalized vertical screen position used for auto-focus.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, autoFocusOffset, 0.0f,
                    "An offset in world units added to the measured auto-focus distance.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, focusDistance, 5.0f,
                    "Manual depth-of-field focus distance in world units.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, focalLength, 100.0f,
                    "Lens focal length in millimeters. Longer lenses produce a shallower depth of field.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, fNumber, 2.8f,
                    "Lens aperture f-number. Higher values deepen the depth of field, lower values produce more blur.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, maxBlurRadius, 16.0f,
                    "The largest blur radius, in pixels at 1080p. It scales with the output height and is limited to "
                    "128 output pixels, so at higher resolutions the limit applies sooner (about 86 at 1600p and 59 at "
                    "2160p).",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, bokehMinIntensity, 1.0f,
                    "Intensity of the centre of the bokeh disc relative to its rim. 0: strongest rim emphasis (optical-vignetting look), 1: even disc.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", float, bokehFilterStrength, 0.25f,
                    "Width of the reconstruction tent used when the half resolution bokeh layers are upsampled. 0: plain bilinear, 1: a full half-resolution texel.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", bool, edgeAwareUpsample, true,
                    "Aligns the depth-of-field blur boundary to the upscaled colour edges. Without it the boundary "
                    "follows the render resolution and stair-steps along silhouettes whenever DLSS or another upscaler "
                    "is active. Costs a few extra texture reads per pixel. At native resolution it makes no difference.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", bool, excludeViewModel, true,
                    "Keeps the view model (the player's weapon and hands) sharp. The view model sits a few centimetres "
                    "from the lens, so a physically correct lens would blur it the most and spread its blur across much "
                    "of the screen. This normally costs nothing and often saves GPU time. Turn it off to frame a shot "
                    "with the weapon deliberately out of focus.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dof", uint, sampleCount, 96,
                    "The most half-resolution bokeh samples per layer. The number of samples grows with the blur radius "
                    "up to this limit, so it caps quality rather than setting a fixed cost per pixel. Values from 1 to "
                    "31 act as 32. 0 disables the effect.",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.ntsc", bool, ntscEnable, false,
                    "Enables the NTSC/VHS composite video look. Expensive: four full-resolution passes with roughly 200 "
                    "filter taps per pixel, so expect a noticeable frame time cost at high resolutions.",
                    args.environment = "RTX_NTSC_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx.ntsc", float, ntscLumaBW, 3.00f,
               "VHS luma bandwidth in MHz. About 3.0 matches SP tape speed and about 1.6 EP.");
    RTX_OPTION("rtx.ntsc", float, ntscColorBW, 425.00f, "VHS color-under bandwidth in kHz.");
    RTX_OPTION("rtx.ntsc", float, ntscRinging, 0.30f, "VHS playback edge-ringing gain.");
    RTX_OPTION("rtx.ntsc", float, ntscLumaNoise, 0.025f, "Luminance-dependent VHS tape noise amplitude.");
    RTX_OPTION("rtx.ntsc", float, ntscTapeDropoutRate, 0.50f, "Average VHS tape dropouts per 480-line frame.");
    RTX_OPTION("rtx.ntsc", float, ntscTapeDropoutLength, 15.0f, "Average VHS dropout length in microseconds.");
    RTX_OPTION("rtx.ntsc", float, ntscHeadSmear, 0.175f, "Worn-head symmetric luma smear strength.");
    RTX_OPTION("rtx.ntsc", float, ntscTapeTrail, 0.675f,
               "Strength of the luma trail to the right of bright edges, from 0 to 1.");

  private:
    Rc<vk::DeviceFn> m_vkd;
    Rc<DxvkBuffer> m_highlightingValues;
    Resources::Resource m_dofFocusState;
    // Half resolution depth-of-field working set. DxvkPostFx is not an RtxPass,
    // so it has no createTargetResource/releaseTargetResource hook: these are
    // created on demand in dispatchDof, rebuilt when the output extent changes,
    // and released by releaseDofResources() when depth of field is switched off
    // and from the destructor.
    Resources::Resource m_dofHalfColorCoC;   // rgb = colour, a = signed normalized CoC
    Resources::Resource m_dofHalfNear;       // premultiplied near field layer
    Resources::Resource m_dofHalfFar;        // defocused base layer
    Resources::Resource m_dofTile;           // rg = (max |CoC| radius, max near radius)
    VkExtent3D m_dofHalfExtent = { 0, 0, 0 };
    bool m_dofFocusStateReset = true;

    float m_lensRainAmount = 0.0f;
    Vector3 m_lensRainVelocity = Vector3(0.0f);
    uint32_t m_lensRainDropCount = 0;
    Rc<DxvkBuffer> m_lensRainStateBuffer;
    float m_lensRainAspect = 0.0f;
    Rc<DxvkBuffer> m_lensRainDropBuffer;
    Rc<DxvkBuffer> m_lensRainTileBuffer;

    uint64_t m_lensRainLastTimeMs = 0;
    uint64_t m_lensRainMotionFrame = 0;
    uint64_t m_lensRainMotionUpdateMs = 0;
    Vector3 m_lensRainGameVelocity = Vector3(0.0f);
    bool m_lensRainGameMotionSeen = false;
    bool m_lensRainHasMotionFrame = false;
    int m_lensRainMotionSource = -1;
    bool m_lensRainInitialized = false;

    void releaseDofResources();

    RTX_OPTION("rtx.postfx", bool,  enableMotionBlurNoiseSample, true, "Enable random distance sampling for every step along the motion vector. The random pattern is generated with interleaved gradient noise.");
    RTX_OPTION("rtx.postfx", bool,  enableMotionBlurEmissive, true, "Enable Motion Blur for Emissive surfaces. Disable this when the motion blur on emissive surfaces cause severe artifacts.");
    RTX_OPTION("rtx.postfx", uint,  motionBlurSampleCount, 4, "The number of samples along the motion vector. More samples could help to reduce motion blur noise.");
    RTX_OPTION("rtx.postfx", float, exposureFraction, 0.4f, "Simulate the camera exposure, the longer exposure will cause stronger motion blur.");
    RTX_OPTION("rtx.postfx", float, blurDiameterFraction, 0.02f, "The diameter of the circle that motion blur samplings occur. Motion vectors beyond this circle will be clamped.");
    RTX_OPTION("rtx.postfx", float, motionBlurMinimumVelocityThresholdInPixel, 1.0f, "The minimum motion vector distance that enable the motion blur. The unit is pixel size.");
    RTX_OPTION("rtx.postfx", float, motionBlurDynamicDeduction, 1.0f, "The deduction of motion blur for dynamic objects.");
    RTX_OPTION("rtx.postfx", float, motionBlurJitterStrength, 0.6f, "The jitter strength of every sample along the motion vector.");
    RTX_OPTION("rtx.postfx", float, chromaticAberrationAmount, 0.02f, "The strength of chromatic aberration.");
    RTX_OPTION("rtx.postfx", float, chromaticCenterAttenuationAmount, 0.975f, "Control the amount of chromatic aberration effect that attunuated when close to the center of screen.");
    RTX_OPTION("rtx.postfx", float, vignetteIntensity, 0.6f, "The darkness of vignette effect.");
    RTX_OPTION("rtx.postfx", float, vignetteRadius, 0.8f, "The radius that vignette effect starts. The unit is normalized screen space, 0 represents the center, 1 means the edge of the short edge of the rendering window. So, this setting can larger than 1 until reach to the long edge of the rendering window.");
    RTX_OPTION("rtx.postfx", float, vignetteSoftness, 0.2f, "The gradient that the color drop to black from the vignetteRadius to the edge of rendering window.");
  };

}
