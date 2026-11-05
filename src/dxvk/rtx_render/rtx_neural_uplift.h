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

#include <memory>
#include <vector>

#include "rtx_common_object.h"
#include "rtx_option.h"
#include "rtx_resources.h"

namespace dxvk {
  // Highest DLSSNR.Style the feature defines. It clamps anything higher to the last one,
  // so offering more would present duplicates of the last model.
  constexpr int kNeuralUpliftMaxModel = 2;

  // Highest NVSDK_NGX_DLSSNR_Hint_Render_Preset the SDK header defines (Preset_G).
  constexpr int kNeuralUpliftMaxPreset = 7;

  // Upper bound on rtx.neuralUplift.passCount. Chaining evaluations is a host-side idea rather
  // than a feature of DLSS-NR, and each extra pass costs a full evaluation plus its own
  // history/scratch allocation.
  constexpr int kNeuralUpliftMaxPassCount = 10;

  class DxvkDevice;
  class DxvkContext;
  class DxvkBarrierSet;
  class RtxContext;
  class NGXNeuralRenderingContext;

  /**
   * \brief DLSS-NR ("Neural Uplift" / 3D-Guided Neural Generation) image enhancement pass
   *
   * A neural post-process on a finished frame. It is not an upscaler - input and output are the
   * same resolution - so it runs downstream of whichever upscaler produced the frame, enhancing
   * it with the depth and motion vectors that upscaler already consumed.
   *
   * The model is LDR-domain, so RtxContext::dispatchDlssNR sandwiches this pass between a fast
   * tone map (m_finalOutput -> m_neuralRenderingInput) and an inverse tone map
   * (m_neuralRenderingOutput -> m_finalOutput, with highlight recovery). Depth, motion vectors
   * and the per-material control mask arrive at render resolution while the colour is at display
   * resolution, and the feature is told their real sizes through its per-buffer subrects rather
   * than having them resampled.
   *
   * The feature is created and evaluated through the driver's NGX core, which loads
   * nvngx_dlssnr.dll from the feature search path set up in NGXContext::initialize.
   */
  class DxvkNeuralUplift : public CommonDeviceObject, public RtxPass {
  public:
    // DLSSNR.Style: conditioning inputs to one network rather than three networks. The network
    // itself is chosen by `preset`.
    enum class Model : int {
      ModelA = 0,
      ModelB = 1,
      ModelC = 2,
    };

    // NVSDK_NGX_DLSSNR_Mask_Blend_Mode: how the control mask and the automatic character mask combine.
    enum class MaskBlendMode : int {
      None = 0,
      Mean = 1,
      Min = 2,
      Max = 3,
    };

    explicit DxvkNeuralUplift(DxvkDevice* device);
    ~DxvkNeuralUplift();

    DxvkNeuralUplift(const DxvkNeuralUplift&)                = delete;
    DxvkNeuralUplift(DxvkNeuralUplift&&) noexcept            = delete;
    DxvkNeuralUplift& operator=(const DxvkNeuralUplift&)     = delete;
    DxvkNeuralUplift& operator=(DxvkNeuralUplift&&) noexcept = delete;

    // Returns false only once NGX has been asked and reported DLSS-NR as unavailable.
    bool isSupported() const;

    // Used by RtxContext and Resources to decide whether the DLSS-NR resources (the control mask
    // and the neural-rendering output) need to exist this frame.
    bool useNeuralUplift() const {
      return isActive();
    }

    // Display-resolution feature size. Recorded on resolution change; the feature is rebuilt on
    // the next dispatch.
    void setNeuralUpliftSettings(const uint32_t displaySize[2]);

    /**
     * Enhances rtOutput.m_neuralRenderingInput into rtOutput.m_neuralRenderingOutput.
     *
     * Returns true when NGX produced a valid output image; the caller only runs the inverse tone
     * map (and therefore only consumes m_neuralRenderingOutput) when it does.
     */
    bool dispatch(Rc<RtxContext> ctx,
                  DxvkBarrierSet& barriers,
                  const Resources::RaytracingOutput& rtOutput,
                  bool resetHistory,
                  bool useRayReconstructionGuides);

    // The developer menu shows the Advanced Settings too. The user menu leaves them out, since a wrong network
    // preset disables the feature.
    void showImguiSettings(bool showDeveloperSettings);

    // Single status line for the developer panel.
    void showImguiStatusLine();

    void release();

    // Releases the NGX feature while the device is still alive.
    void onDestroy();

    RTX_OPTION_ARGS("rtx.neuralUplift", bool, enable, false,
                    "Enables DLSS-NR (DLSS 3D-Guided Neural Generation), a neural image enhancement applied to the "
                    "finished frame at display resolution, using the depth and motion vectors the upscaler used. It is "
                    "driven by the per-material DLSS control mask, combined with the automatic character mask when "
                    "rtx.neuralUplift.useAutoMask is on. NGX decides whether the GPU and driver support it.",
                    args.environment = "RTX_NEURAL_UPLIFT_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", float, intensity, 1.0f,
                    "Blends the enhanced image with the original (DLSSNR.Intensity). 1 is the full effect. Below 1, a "
                    "copy of the input is kept for the blend, which costs a little more.",
                    args.environment = "RTX_NEURAL_UPLIFT_INTENSITY",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", float, structuralStrength, 0.7f,
                    "Adjustments to details, shadows and materials (DLSSNR.LocalStructureStrength). Lower values\n"
                    "preserve the original structure; higher values apply stronger adjustments.",
                    args.environment = "RTX_NEURAL_UPLIFT_STRUCTURAL_STRENGTH",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", float, toneStrength, 0.3f,
                    "Adjustments to lightness, darkness and colour (DLSSNR.LocalToneStrength). Lower values keep the "
                    "original tone and colour, higher values adjust them more. This also sets how strongly the selected "
                    "model applies, so 0 makes every model look the same as the input.",
                    args.environment = "RTX_NEURAL_UPLIFT_TONE_STRENGTH",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", Model, model, Model::ModelA,
                    "Selects one of three DLSS-NR looks (DLSSNR.Style): 0: Model A, 1: Model B, 2: Model C. The result "
                    "depends on the content, so compare them in the scene you care about. These are variations of one "
                    "network. rtx.neuralUplift.preset chooses the network.",
                    args.environment = "RTX_NEURAL_UPLIFT_MODEL",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", bool, useAutoMask, false,
                    "Adds DLSS-NR's automatic character mask (DLSSNR.UseAutoMask) to the per-material control mask. "
                    "rtx.neuralUplift.maskBlendMode sets how the two combine. rtx.neuralUplift.skinStructureStrength "
                    "only applies with the automatic mask.",
                    args.environment = "RTX_NEURAL_UPLIFT_USE_AUTO_MASK",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", MaskBlendMode, maskBlendMode, MaskBlendMode::None,
                    "How the per-material control mask and the automatic character mask combine while "
                    "rtx.neuralUplift.useAutoMask is on (DLSSNR.MaskBlendMode). 0: None, 1: Mean, 2: Min, 3: Max. "
                    "None keeps the control mask's values. Mean averages the two masks, Min takes the lower value per pixel "
                    "and Max the higher. Has no effect while the automatic mask is off.",
                    args.environment = "RTX_NEURAL_UPLIFT_MASK_BLEND_MODE",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", float, skinStructureStrength, 0.5f,
                    "Structure adjustment for characters while rtx.neuralUplift.useAutoMask is on "
                    "(DLSSNR.SkinStructureStrength), kept separate so faces are not over-sharpened by "
                    "rtx.neuralUplift.structuralStrength. Lower values keep the original structure. Has no effect while "
                    "the automatic mask is off.",
                    args.environment = "RTX_NEURAL_UPLIFT_SKIN_STRUCTURE_STRENGTH",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", int, preset, 0,
                    "Network selection hint (DLSSNR.Hint.Render.Preset), 0 to 7. 0 lets the SDK choose, and 1 to 7 "
                    "select presets A to G. Which presets exist depends on the installed DLSS-NR version. Asking for a "
                    "missing one logs a fallback and uses the default. Unlike rtx.neuralUplift.model, which picks a "
                    "look within the network, this picks the network. Changing it recreates the feature and resets its "
                    "history.",
                    args.environment = "RTX_NEURAL_UPLIFT_PRESET",
                    args.minValue = 0, args.maxValue = kNeuralUpliftMaxPreset,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", int, passCount, 1,
                    "Runs DLSS-NR this many times in a row on each frame, 1 to 10. Each pass works on the previous "
                    "pass's output, so the effect stacks. Each extra pass costs a full evaluation, one image copy and "
                    "its own history memory. Changing this recreates the feature and resets its history.",
                    args.environment = "RTX_NEURAL_UPLIFT_PASS_COUNT",
                    args.minValue = 1, args.maxValue = kNeuralUpliftMaxPassCount,
                    args.flags = RtxOptionFlags::UserSetting);

    // Highlight recovery substitutes original pre-NR HDR for pixels that were very bright
    // and would land in the active tone mapper's shoulder. It lives in the inverse tone map
    // (DxvkToneMapping::dispatchInverseToneMapping), not in this pass.
    //   .x/.y = exposed HDR max-channel preserve range (default 1.0..8.0)
    //   .z/.w = exposed LDR max-channel shoulder range (default 0.75..0.99)
    RTX_OPTION_ARGS("rtx.neuralUplift", bool, enableHighlightRecovery, true,
                    "Restores highlights that DLSS-NR's low dynamic range processing compresses, when the image is "
                    "mapped back to HDR.",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION("rtx.neuralUplift", Vector4, highlightRecoveryThresholds, Vector4(1.0f, 8.0f, 0.75f, 0.99f),
                    "The edges of the highlight recovery blend: the exposed HDR maximum-channel range (x to y) and the "
                    "LDR maximum-channel range (z to w).");

    RTX_OPTION_ARGS("rtx.neuralUplift", bool, enableVolumetricControlMask, true,
                    "Weakens the DLSS-NR control mask through fog, volumetrics and alpha-blended surfaces, in "
                    "proportion to how much they hide.",
                    args.flags = RtxOptionFlags::UserSetting);

  protected:
    bool isEnabled() const override;
    void onDeactivation() override;

  private:
    // Creates or recreates the feature. The creation-time state is recorded before the attempt can fail,
    // so a failed attempt is not retried every frame.
    void initializeFeature(Rc<DxvkContext> ctx, int passes);

    // True when every pass holds a usable feature.
    bool isFeatureInitialized() const;

    void releaseFeature();

    // One context per pass: NGXNeuralRenderingContext owns a single NGX handle, and each pass
    // needs its own temporal history for the reason described on passCount.
    std::vector<std::unique_ptr<NGXNeuralRenderingContext>> m_contexts;

    uint32_t m_displaySize[2] = {};

    bool m_recreate = true;
    bool m_evaluatedLastFrame = false;

    // Creation-time state the feature was built around; a change to any of it rebuilds.
    int m_createdPreset = -1;
    int m_createdPassCount = 0;
    uint32_t m_createdDisplaySize[2] = {};

    // Forces DLSSNR.Reset on the next evaluation regardless of what the frame asked for. The
    // temporal history is only meaningful if the evaluation before this one produced it, and
    // a new feature has none, while a failed evaluation did not advance it. It stays set until an evaluation
    // succeeds. The effect options do not set it, since the feature resets itself when Style, UseAutoMask,
    // LocalToneStrength, LocalStructureStrength or SkinStructureStrength change.
    bool m_forceHistoryReset = true;

    // Diagnostics for the developer panel.
    uint32_t m_initCount = 0;
    uint32_t m_loggedEvaluationInit = 0;
    const char* m_statusReason = "not dispatched yet";
  };
} // namespace dxvk
