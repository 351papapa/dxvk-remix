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
#include "rtx_neural_uplift.h"

#include <algorithm>

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_context.h"
#include "rtx_imgui.h"
#include "rtx_ngx_wrapper.h"
#include "rtx_options.h"

namespace dxvk {

  DxvkNeuralUplift::DxvkNeuralUplift(DxvkDevice* device)
    : CommonDeviceObject(device)
    , RtxPass(device) {
  }

  DxvkNeuralUplift::~DxvkNeuralUplift() {
    release();
  }

  bool DxvkNeuralUplift::isSupported() const {
    NGXContext& ngxContext = m_device->getCommon()->metaNGXContext();

    // NGX comes up lazily, and before it has been asked an unchecked "no" would look the same as a real one.
    return !ngxContext.isDlssNeuralRenderingSupportChecked() || ngxContext.supportsDlssNeuralRendering();
  }

  bool DxvkNeuralUplift::isEnabled() const {
    return enable() && isSupported();
  }

  bool DxvkNeuralUplift::isFeatureInitialized() const {
    if (m_contexts.empty()) {
      return false;
    }

    for (const std::unique_ptr<NGXNeuralRenderingContext>& context : m_contexts) {
      if (!context || !context->isNeuralRenderingInitialized()) {
        return false;
      }
    }

    return true;
  }

  void DxvkNeuralUplift::releaseFeature() {
    // The features own GPU resources DXVK knows nothing about and so cannot keep alive, which is
    // why releasing one has to wait for the GPU to drain first.
    if (!m_contexts.empty()) {
      m_device->waitForIdle();
    }

    for (std::unique_ptr<NGXNeuralRenderingContext>& context : m_contexts) {
      if (context) {
        context->releaseNGXFeature();
      }
    }
    m_contexts.clear();

    m_evaluatedLastFrame = false;
  }

  void DxvkNeuralUplift::release() {
    releaseFeature();

    m_recreate = true;
    m_createdDisplaySize[0] = 0;
    m_createdDisplaySize[1] = 0;
    m_createdPassCount = 0;
  }

  void DxvkNeuralUplift::onDestroy() {
    release();
  }

  void DxvkNeuralUplift::onDeactivation() {
    releaseFeature();
    m_recreate = true;
    m_createdDisplaySize[0] = 0;
    m_createdDisplaySize[1] = 0;
    m_createdPassCount = 0;
  }

  void DxvkNeuralUplift::setNeuralUpliftSettings(const uint32_t displaySize[2]) {
    ScopedCpuProfileZone();

    m_displaySize[0] = displaySize[0];
    m_displaySize[1] = displaySize[1];

    m_recreate = true;
  }

  void DxvkNeuralUplift::initializeFeature(Rc<DxvkContext> ctx, int passes) {
    // Recorded before the attempt can fail. dispatch() diffs against these to decide whether to rebuild,
    // so a failed attempt must not look like a pending settings change, or it would retry (and waitForIdle)
    // every frame.
    m_createdPreset = preset();
    m_createdPassCount = passes;
    m_createdDisplaySize[0] = m_displaySize[0];
    m_createdDisplaySize[1] = m_displaySize[1];

    releaseFeature();

    if (m_displaySize[0] == 0 || m_displaySize[1] == 0) {
      m_statusReason = "display size not known yet";
      return;
    }

    // The preset option is a raw number because the SDK documents no mapping beyond "A-G";
    // clamp it to the range the header defines rather than handing NGX an arbitrary value.
    const uint32_t clampedPreset = static_cast<uint32_t>(std::clamp(preset(), 0, kNeuralUpliftMaxPreset));

    NGXContext& ngxContext = m_device->getCommon()->metaNGXContext();

    for (int i = 0; i < passes; ++i) {
      std::unique_ptr<NGXNeuralRenderingContext> context = ngxContext.createDlssNeuralRenderingContext();
      if (!context) {
        break;
      }

      context->initialize(ctx, m_displaySize, clampedPreset);
      if (!context->isNeuralRenderingInitialized()) {
        break;
      }

      m_contexts.push_back(std::move(context));
    }

    if (m_contexts.size() == static_cast<size_t>(passes)) {
      m_initCount++;
      m_statusReason = "active";
      // A feature this new has no history behind it, whatever the frame thinks about continuity.
      m_forceHistoryReset = true;
      return;
    }

    // All-or-nothing: a partial set of handles cannot serve the pass loop.
    releaseFeature();
    m_statusReason = ngxContext.supportsDlssNeuralRendering()
      ? "feature creation failed"
      : "NGX reports DLSS-NR as unavailable";
  }

  bool DxvkNeuralUplift::dispatch(Rc<RtxContext> ctx,
                                  DxvkBarrierSet& barriers,
                                  const Resources::RaytracingOutput& rtOutput,
                                  bool resetHistory,
                                  bool useRayReconstructionGuides) {
    ScopedGpuProfileZone(ctx, "Neural Uplift");
    ctx->setFramePassStage(RtxFramePassStage::DLSSNR);

    if (!isActive()) {
      return false;
    }

    const int passes = std::clamp(passCount(), 1, kNeuralUpliftMaxPassCount);

    m_recreate |= (m_createdPreset != preset())
      || (m_createdPassCount != passes)
      || (m_createdDisplaySize[0] != m_displaySize[0])
      || (m_createdDisplaySize[1] != m_displaySize[1]);

    if (m_recreate) {
      Rc<DxvkContext> dxvkContext = ctx;
      initializeFeature(dxvkContext, passes);
      m_recreate = false;
    }

    m_evaluatedLastFrame = false;

    if (!isFeatureInitialized()) {
      return false;
    }

    // Resources::updateDlssNeuralRenderingResources allocates these lazily in the onFrameBegin sweep that
    // activates this pass, so they normally exist by now. AliasedResource::resource() dereferences
    // unconditionally, so the empty case is still checked.
    if (rtOutput.m_neuralRenderingInput.empty() || rtOutput.m_neuralRenderingOutput.empty()) {
      m_statusReason = "colour resources not allocated yet";
      return false;
    }

    // The colour is read and written by separate resources, so no staging copy is needed for a
    // single pass; ReadWrite is only registered when a later pass copies the output back in.
    const Resources::Resource& colorInput = rtOutput.m_neuralRenderingInput.resource(
      passes > 1 ? Resources::AccessType::ReadWrite : Resources::AccessType::Read);
    const Resources::Resource& colorOutput = rtOutput.m_neuralRenderingOutput.resource(
      Resources::AccessType::Write);

    if (colorInput.image == nullptr || colorInput.view == nullptr ||
        colorOutput.image == nullptr || colorOutput.view == nullptr) {
      m_statusReason = "colour resources unavailable";
      return false;
    }

    const Resources::Resource* motionVectors = useRayReconstructionGuides
      ? &rtOutput.m_primaryScreenSpaceMotionVectorDLSSRR
      : &rtOutput.m_primaryScreenSpaceMotionVector;
    const Resources::Resource* depth = useRayReconstructionGuides
      ? &rtOutput.m_primaryDepthDLSSRR.resource(Resources::AccessType::Read)
      : &rtOutput.m_primaryDepth;

    if (motionVectors->image == nullptr) {
      motionVectors = nullptr;
    }
    if (depth->image == nullptr) {
      depth = nullptr;
    }

    const Resources::Resource* controlMask =
      rtOutput.m_controlMask.image != nullptr ? &rtOutput.m_controlMask : nullptr;

    // NGXNeuralRenderingContext reads every input's extent before deciding what to bind, so all of them have to exist.
    if (depth == nullptr || motionVectors == nullptr || controlMask == nullptr) {
      m_statusReason = "a required input is missing";
      return false;
    }

    const VkExtent3D outputExtent = colorOutput.image->info().extent;

    // Transitioned once, outside the pass loop. They move from their steady state to SHADER_READ and stay
    // readable for every pass, so a later transition would name the wrong source state.
    const Resources::Resource* readOnlyInputs[] = { motionVectors, depth, controlMask };

    for (const Resources::Resource* input : readOnlyInputs) {
      if (input == nullptr || input->view == nullptr) {
        continue;
      }

      barriers.accessImage(
        input->image,
        input->view->imageSubresources(),
        input->image->info().layout,
        input->image->info().stages,
        input->image->info().access,
        input->image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);

#ifdef REMIX_DEVELOPMENT
      ctx->cacheResourceAliasingImageView(input->view);
#endif
    }

#ifdef REMIX_DEVELOPMENT
    ctx->cacheResourceAliasingImageView(colorInput.view);
    ctx->cacheResourceAliasingImageView(colorOutput.view);
#endif

    NGXNeuralRenderingContext::NGXNeuralRenderingBuffers buffers;
    buffers.pInColor = &colorInput;
    buffers.pOutColor = &colorOutput;
    buffers.pMotionVectors = motionVectors;
    buffers.pDepth = depth;
    buffers.pControlMask = controlMask;

    float jitterOffset[2] = { 0.0f, 0.0f };
    m_device->getCommon()->getSceneManager().getCamera().getJittering(jitterOffset);

    const bool autoMask = useAutoMask();

    NGXNeuralRenderingContext::NGXNeuralRenderingSettings settings = {};
    settings.jitterOffset[0] = jitterOffset[0];
    settings.jitterOffset[1] = jitterOffset[1];
    // The path tracer's screen-space motion vectors are already in absolute render pixels, which
    // is the same convention DLSS consumes them under, so the scale is fixed at 1.
    settings.motionVectorScale[0] = 1.0f;
    settings.motionVectorScale[1] = 1.0f;
    settings.intensity = intensity();
    settings.toneStrength = toneStrength();
    settings.structuralStrength = structuralStrength();
    settings.model = static_cast<uint32_t>(
      std::clamp(static_cast<int>(model()), 0, kNeuralUpliftMaxModel));
    settings.useAutoMask = autoMask;
    settings.maskBlendMode = autoMask ? static_cast<uint32_t>(maskBlendMode()) : 0u;
    // The skin classification that this applies through only exists while the automatic mask is on.
    settings.skinStructureStrength = autoMask ? skinStructureStrength() : 0.0f;

    const VkImageSubresourceLayers copyLayers = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    bool evaluated = false;

    for (int pass = 0; pass < passes; ++pass) {
      // Reset only on a real discontinuity. Each pass owns its own NGX handle, so pass k's history is
      // always that same pass's previous frame.
      settings.resetAccumulation = resetHistory || m_forceHistoryReset;

      if (pass > 0) {
        // Feed the previous pass's output back in. DxvkContext::copyImage does its own layout
        // transitions from each image's steady-state layout, so these barriers only have to make
        // the previous NGX write visible to the transfer and vice versa.
        barriers.accessImage(
          colorOutput.image,
          colorOutput.view->imageSubresources(),
          colorOutput.image->info().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT,
          colorOutput.image->info().layout,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT);

        barriers.accessImage(
          colorInput.image,
          colorInput.view->imageSubresources(),
          colorInput.image->info().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT,
          colorInput.image->info().layout,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_WRITE_BIT);

        barriers.recordCommands(ctx->getCommandList());

        ctx->copyImage(
          colorInput.image, copyLayers, { 0, 0, 0 },
          colorOutput.image, copyLayers, { 0, 0, 0 },
          outputExtent);

        barriers.accessImage(
          colorInput.image,
          colorInput.view->imageSubresources(),
          colorInput.image->info().layout,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_WRITE_BIT,
          colorInput.image->info().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT);

        barriers.accessImage(
          colorOutput.image,
          colorOutput.view->imageSubresources(),
          colorOutput.image->info().layout,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT,
          colorOutput.image->info().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT);
      } else {
        barriers.accessImage(
          colorInput.image,
          colorInput.view->imageSubresources(),
          colorInput.image->info().layout,
          colorInput.image->info().stages,
          colorInput.image->info().access,
          colorInput.image->info().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT);

        barriers.accessImage(
          colorOutput.image,
          colorOutput.view->imageSubresources(),
          colorOutput.image->info().layout,
          colorOutput.image->info().stages,
          colorOutput.image->info().access,
          colorOutput.image->info().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT);
      }

      barriers.recordCommands(ctx->getCommandList());

      // pass selects which handle this call lands on - see passCount for why each pass has one.
      evaluated = m_contexts[pass]->evaluateNeuralRendering(ctx, buffers, settings);

      // A failed pass leaves the output holding whatever the last good pass wrote, so there is no
      // point spending the remaining passes on it.
      if (!evaluated) {
        break;
      }
    }

    barriers.accessImage(
      colorOutput.image,
      colorOutput.view->imageSubresources(),
      colorOutput.image->info().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      colorOutput.image->info().layout,
      colorOutput.image->info().stages,
      colorOutput.image->info().access);

    barriers.accessImage(
      colorInput.image,
      colorInput.view->imageSubresources(),
      colorInput.image->info().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT,
      colorInput.image->info().layout,
      colorInput.image->info().stages,
      colorInput.image->info().access);

    barriers.recordCommands(ctx->getCommandList());

    // NGX captured the raw VkImageView, so the view is tracked as well as the image. A view references its
    // image, not the other way around.
    ctx->getCommandList()->trackResource<DxvkAccess::None>(colorInput.view);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(colorInput.image);
    ctx->getCommandList()->trackResource<DxvkAccess::None>(colorOutput.view);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(colorOutput.image);

    // A rejected evaluation left the history where the previous one put it, so the next frame
    // would reproject across the gap. Held until an evaluation actually succeeds.
    if (evaluated && m_loggedEvaluationInit != m_initCount) {
      Logger::info(str::format("[DLSS-NR] Evaluated ", passes, " pass(es) at ", m_displaySize[0], "x", m_displaySize[1],
                               " via driver NGX, preset hint ", m_createdPreset, ", style ", int(model())));
      m_loggedEvaluationInit = m_initCount;
    }
    m_forceHistoryReset = !evaluated;
    m_evaluatedLastFrame = evaluated;
    m_statusReason = evaluated ? "active" : "evaluate failed";

    return evaluated;
  }

  void DxvkNeuralUplift::showImguiStatusLine() {
    NGXContext& ngxContext = m_device->getCommon()->metaNGXContext();

    ImGui::Text("DLSS-NR: %s - preset %d, inits %u", m_statusReason, preset(), m_initCount);

    // Reported, never used to hide the controls.
    if (!isSupported()) {
      ImGui::TextWrapped("NGX reports DLSS-NR as unavailable on this GPU or driver, or nvngx_dlssnr.dll was not found. "
                         "NGX looks for nvngx_dlssnr.dll in: %s.", ngxContext.getFeatureSearchPathText().c_str());
    }
  }

  void DxvkNeuralUplift::showImguiSettings(bool showDeveloperSettings) {
    static auto modelCombo = RemixGui::ComboWithKey<Model>(
      "Model",
      { {
        { Model::ModelA, "Model A" },
        { Model::ModelB, "Model B" },
        { Model::ModelC, "Model C" },
      } });

    static auto maskBlendModeCombo = RemixGui::ComboWithKey<MaskBlendMode>(
      "Mask Blend Mode",
      { {
        { MaskBlendMode::None, "None" },
        { MaskBlendMode::Mean, "Mean" },
        { MaskBlendMode::Min, "Min" },
        { MaskBlendMode::Max, "Max" },
      } });

    RemixGui::Checkbox("Enable DLSS-NR", &enableObject());

    showImguiStatusLine();

    if (!enable()) {
      return;
    }

    ImGui::Indent();

    constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;

    modelCombo.getKey(&modelObject());
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.Style. Conditioning inputs to one network rather than three networks -\n"
                        "the network itself is chosen by Network Preset under Advanced Settings.");
    }

    RemixGui::DragFloat("Structure Intensity", &structuralStrengthObject(), 0.01f, 0.0f, 1.0f, "%.3f", sliderFlags);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.LocalStructureStrength. Adjustments to details, shadows and materials.");
    }

    RemixGui::DragFloat("Tone Intensity", &toneStrengthObject(), 0.01f, 0.0f, 1.0f, "%.3f", sliderFlags);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.LocalToneStrength. Adjustments to lightness, darkness and colour. This is\n"
                        "also how far the selected Model is blended in from neutral, so 0 makes any\n"
                        "Model a no-op.");
    }

    RemixGui::DragFloat("Overall Intensity", &intensityObject(), 0.01f, 0.0f, 1.0f, "%.3f", sliderFlags);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.Intensity. Blend of the enhanced image against the original. Below 1 the\n"
                        "feature keeps an extra copy of the input to blend against, so it also costs\n"
                        "slightly more.");
    }

    RemixGui::Checkbox("Enable Auto Mask", &useAutoMaskObject());
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.UseAutoMask. Adds the feature's own per-pixel character mask to the\n"
                        "per-material DLSS control mask. Mask Blend Mode sets how the two combine.");
    }

    const bool autoMaskEnabled = useAutoMask();
    ImGui::BeginDisabled(!autoMaskEnabled);
    maskBlendModeCombo.getKey(&maskBlendModeObject());
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.MaskBlendMode. None keeps the control mask's values. Mean averages the two\n"
                        "masks, Min takes the lower value per pixel and Max the higher.");
    }
    RemixGui::DragFloat("Character Intensity", &skinStructureStrengthObject(), 0.01f, 0.0f, 1.0f, "%.3f", sliderFlags);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.SkinStructureStrength. Applied only through the automatic mask, which is the\n"
                        "only thing that classifies skin.");
    }
    ImGui::EndDisabled();

    ImGui::Separator();
    if (showDeveloperSettings && RemixGui::CollapsingHeader("Advanced Settings")) {
      ImGui::Indent();

      RemixGui::Checkbox("Highlight Recovery", &enableHighlightRecoveryObject());
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Recovers highlights compressed by the LDR-domain processing. Applied by the\n"
                          "inverse tone map on the way back to HDR.");
      }
      // Thresholds: (exposed HDR max-channel min/max, LDR max-channel min/max)
      RemixGui::DragFloat4("Highlight Recovery Thresholds", &highlightRecoveryThresholdsObject(), 0.01f, 0.0f, 32.0f, "%.3f", sliderFlags);

      RemixGui::Checkbox("Volumetric Improvement", &enableVolumetricControlMaskObject());

      RemixGui::DragInt("Pass Count", &passCountObject(), 1.0f, 1, kNeuralUpliftMaxPassCount, "%d", sliderFlags);
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Runs the pass this many times over, each one enhancing what the last produced.\n"
                          "Costs a full evaluation and one image copy per extra pass, plus its own\n"
                          "temporal history. Changing this recreates the feature and resets the history.");
      }

      RemixGui::DragInt("Network Preset", &presetObject(), 1.0f, 0, kNeuralUpliftMaxPreset, "%d", sliderFlags);
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("DLSSNR.Hint.Render.Preset - the network selector, distinct from Model above.\n"
                          "0 leaves the choice to the SDK; 1-7 pin presets A-G. Asking for one the\n"
                          "installed build does not ship falls back to its default with a log line.\n"
                          "Changing it recreates the feature and resets the temporal history.");
      }

      ImGui::Unindent();
    }

    ImGui::Unindent();
  }

} // namespace dxvk
