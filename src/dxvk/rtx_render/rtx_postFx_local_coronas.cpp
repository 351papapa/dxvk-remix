#include <algorithm>
#include <cmath>
#include <utility>

#include <remix/remix_local_coronas.h>
#include "rtx_postFx.h"
#include "rtx_context.h"
#include "rtx_imgui.h"
#include "rtx_shader_manager.h"
#include "dxvk_scoped_annotation.h"
#include "rtx/pass/local_coronas.h"
#include <rtx_shaders/local_corona_prepare.h>
#include <rtx_shaders/local_corona_composite.h>

namespace dxvk {
  namespace {
    static_assert(CORONA_SOURCE_LIMIT == remixapi::local_coronas::limit);
    static_assert(sizeof(LocalCoronaInput) == sizeof(remixapi::local_coronas::Source));
    static_assert(offsetof(LocalCoronaInput, falloff) == offsetof(remixapi::local_coronas::Source, falloff));
    static_assert(sizeof(LocalCoronaSprite) == 32);
    static_assert(sizeof(LocalCoronaArgs) == 32);

    class LocalCoronaPrepareShader : public ManagedShader {
      SHADER_SOURCE(LocalCoronaPrepareShader, VK_SHADER_STAGE_COMPUTE_BIT, local_corona_prepare)
      BINDLESS_ENABLED()
      PUSH_CONSTANTS(LocalCoronaArgs)
      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        STRUCTURED_BUFFER(CORONA_INPUT)
        RW_STRUCTURED_BUFFER(CORONA_SPRITES)
      END_PARAMETER()
    };

    class LocalCoronaCompositeShader : public ManagedShader {
      SHADER_SOURCE(LocalCoronaCompositeShader, VK_SHADER_STAGE_COMPUTE_BIT, local_corona_composite)
      PUSH_CONSTANTS(LocalCoronaArgs)
      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(CORONA_SPRITES)
        RW_TEXTURE2D(CORONA_COLOR)
      END_PARAMETER()
    };
  }

  bool DxvkPostFx::prepareLocalCoronas() {
    if (ShaderManager::getInstance() == nullptr) return false;
    // Lazy, only on an opt-in submission. Register both pipelines before the
    // caller checks pending compilation and takes over the host raster draw.
    return LocalCoronaPrepareShader::getShader() != nullptr && LocalCoronaCompositeShader::getShader() != nullptr;
  }

  bool DxvkPostFx::setLocalCoronas(uint32_t frame, uint64_t sceneEpoch, std::vector<remixapi::local_coronas::Source>&& sources, bool suppressed, bool available) {
    using State = remixapi::local_coronas::State;
    const State decision = remixapi::local_coronas::submissionState(suppressed,
      m_localCoronasSubmitted && m_localCoronasFrame == frame && m_localCoronasEpoch == sceneEpoch,
      m_localCoronasDispatched && m_localCoronasDispatchFrame == frame, available);
    if (decision == State::Duplicate) {
      recordLocalCoronaStatus(decision, frame, uint32_t(m_localCoronas.size()));
      return true;
    }
    if (decision == State::Late || decision == State::Unavailable) {
      recordLocalCoronaStatus(decision, frame, uint32_t(sources.size()));
      return false;
    }
    m_localCoronasSubmitted = true;
    recordLocalCoronaStatus(suppressed ? State::Suppressed : State::Accepted, frame, uint32_t(sources.size()));
    m_localCoronasFrame = frame;
    m_localCoronasEpoch = sceneEpoch;
    m_localCoronas = std::move(sources);
    m_localCoronasConsumed = suppressed;
    if (!m_localCoronas.empty()) {
      ONCE(Logger::info(str::format("[LocalCoronas] First nonempty source snapshot: frame=", frame,
        " count=", m_localCoronas.size(), " sceneEpoch=", sceneEpoch)));
    }
    return true;
  }

  void DxvkPostFx::dispatchLocalCoronas(Rc<RtxContext> ctx, const Resources::RaytracingOutput& rtOutput) {
    const uint32_t frame = ctx->getDevice()->getCurrentFrameId();
    m_localCoronasDispatched = true;
    m_localCoronasDispatchFrame = frame;
    if (m_localCoronasConsumed) {
      return;
    }
    m_localCoronasConsumed = true;
    if (m_localCoronasFrame != frame || m_localCoronasEpoch != ctx->getSceneManager().clearEpoch()) {
      recordLocalCoronaStatus(remixapi::local_coronas::State::Stale, m_localCoronasFrame, uint32_t(m_localCoronas.size()));
      return;
    }
    if (m_localCoronas.empty()) {
      recordLocalCoronaStatus(remixapi::local_coronas::State::Idle, frame, 0);
      return;
    }
    ScopedCpuProfileZone();
    ScopedGpuProfileZone(ctx, "Local Corona Cores");
    LocalCoronaArgs args {};
    args.width = rtOutput.m_finalOutputExtent.width;
    args.height = rtOutput.m_finalOutputExtent.height;
    args.count = uint32_t(m_localCoronas.size()); // Validated whole batch, never truncate.
    args.sizeScale = std::clamp(localCoronasSize(), 0.01f, 4.0f);
    args.brightness = std::clamp(localCoronasBrightness(), 0.0f, 20.0f);
    args.endpointBias = 0.05f * RtxOptions::getMeterToWorldUnitScale();
    if (m_localCoronaInputs == nullptr) {
      DxvkBufferCreateInfo info {};
      info.size = remixapi::local_coronas::limit * sizeof(LocalCoronaInput);
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      m_localCoronaInputs = ctx->getDevice()->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "Local corona sources");
      info.size = remixapi::local_coronas::limit * sizeof(LocalCoronaSprite);
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      m_localCoronaSprites = ctx->getDevice()->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "Visible local coronas");
    }

    ctx->setFramePassStage(RtxFramePassStage::PostFX);
    ctx->writeToBuffer(m_localCoronaInputs, 0, args.count * sizeof(LocalCoronaInput), m_localCoronas.data());
    ctx->bindCommonRayTracingResources(rtOutput);
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx->pushConstants(0, sizeof(args), &args);
    ctx->bindResourceBuffer(CORONA_INPUT, DxvkBufferSlice(m_localCoronaInputs));
    ctx->bindResourceBuffer(CORONA_SPRITES, DxvkBufferSlice(m_localCoronaSprites));
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LocalCoronaPrepareShader::getShader());
    ctx->dispatch((args.count + 31) / 32, 1, 1);

    ctx->bindResourceView(CORONA_COLOR, rtOutput.m_finalOutput.view(Resources::AccessType::ReadWrite), nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LocalCoronaCompositeShader::getShader());
    ctx->dispatch((args.width + 15) / 16, (args.height + 15) / 16, 1);
    recordLocalCoronaStatus(remixapi::local_coronas::State::Dispatched, frame, args.count);
    ONCE(Logger::info("[LocalCoronas] Native local corona pass dispatched for a current-frame source snapshot."));
  }

  void DxvkPostFx::showLocalCoronaSettings() {
    constexpr auto flags = ImGuiSliderFlags_AlwaysClamp;
    RemixGui::DragFloat("Core brightness", &localCoronasBrightnessObject(), 0.02f, 0.0f, 20.0f, "%.2f", flags);
    RemixGui::DragFloat("Core size", &localCoronasSizeObject(), 0.02f, 0.01f, 4.0f, "%.2f", flags);
    ImGui::TextWrapped("Requires the host's RTX PostFX core backend. Disabling this filter hides v2 host cores. Starbursts and emitting fixtures are independent.");
    const uint64_t status = localCoronaStatus();
    ImGui::Text("Frame %u; sources %u / %u", uint32_t(status >> 32), uint32_t(status >> 8) & 0xffffffu, remixapi::local_coronas::limit);
    ImGui::TextWrapped("%s", remixapi::local_coronas::stateName(static_cast<remixapi::local_coronas::State>(status & 255)));
    ImGui::TextWrapped("Dispatch records CPU command submission, not GPU completion. Screen-space cores do not contribute geometry to reflections or GI.");
  }
}
