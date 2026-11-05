#include <algorithm>
#include <cmath>
#include <utility>

#include <remix/remix_local_starbursts.h>
#include "rtx_postFx.h"
#include "rtx_asset_data_manager.h"
#include "rtx_context.h"
#include "rtx_imgui.h"
#include "rtx_shader_manager.h"
#include "rtx_texture_manager.h"
#include "rtx_water.h"
#include "dxvk_scoped_annotation.h"
#include "rtx/pass/local_starbursts.h"
#include <rtx_shaders/local_starburst_prepare.h>
#include <rtx_shaders/local_starburst_composite.h>

namespace dxvk {
  namespace {
    static_assert(sizeof(LocalStarburstInput) == sizeof(remixapi_LocalStarburstSource));
    static_assert(offsetof(LocalStarburstInput, aspect) == offsetof(remixapi_LocalStarburstSource, aspect));
    static_assert(sizeof(LocalStarburstSprite) == 32);
    static_assert(sizeof(LocalStarburstArgs) == 32);

    class LocalStarburstPrepareShader : public ManagedShader {
      SHADER_SOURCE(LocalStarburstPrepareShader, VK_SHADER_STAGE_COMPUTE_BIT, local_starburst_prepare)
      BINDLESS_ENABLED()
      PUSH_CONSTANTS(LocalStarburstArgs)
      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        STRUCTURED_BUFFER(STARBURST_INPUT)
        RW_STRUCTURED_BUFFER(STARBURST_SPRITES)
      END_PARAMETER()
    };

    class LocalStarburstCompositeShader : public ManagedShader {
      SHADER_SOURCE(LocalStarburstCompositeShader, VK_SHADER_STAGE_COMPUTE_BIT, local_starburst_composite)
      PUSH_CONSTANTS(LocalStarburstArgs)
      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(STARBURST_SPRITES)
        RW_TEXTURE2D(STARBURST_COLOR)
        SAMPLER2D(STARBURST_MASK)
      END_PARAMETER()
    };
  }

  void DxvkPostFx::setLocalStarbursts(uint32_t frame, uint64_t sceneEpoch, std::vector<remixapi_LocalStarburstSource>&& sources) {
    m_localStarsFrame = frame;
    m_localStarsEpoch = sceneEpoch;
    m_localStars = std::move(sources);
    if (!m_localStars.empty()) {
      ONCE(Logger::info(str::format("[Starbursts] First nonempty source snapshot: frame=", frame,
        " count=", m_localStars.size(), " sceneEpoch=", sceneEpoch)));
    }
  }

  void DxvkPostFx::dispatchLocalStarbursts(Rc<RtxContext> ctx, const Resources::RaytracingOutput& rtOutput) {
    if (m_localStars.empty() || m_localStarsFrame != ctx->getDevice()->getCurrentFrameId() ||
        m_localStarsEpoch != ctx->getSceneManager().clearEpoch() || !enable() || !enabled() ||
        maskTexture().empty() || !(brightness() > 0) || !(angularScale() > 0)) {
      return;
    }
    ScopedCpuProfileZone();
    ScopedGpuProfileZone(ctx, "Local Starbursts");
    if (m_localStarPath != maskTexture()) {
      m_localStarPath = maskTexture();
      m_localStarMask = {};
      const auto asset = AssetDataManager::get().findAsset(WaterSystem::resolveAssetPath(m_localStarPath));
      if (asset != nullptr) {
        m_localStarMask = TextureRef(ctx->getCommonObjects()->getTextureManager().preloadTextureAsset(asset, ColorSpace::AUTO, true));
      }
      if (m_localStarMask.isImageEmpty()) {
        Logger::warn("[Starbursts] Cannot load mask: " + m_localStarPath);
      }
    }
    if (m_localStarMask.isImageEmpty()) {
      return;
    }

    LocalStarburstArgs args {};
    args.width = rtOutput.m_finalOutputExtent.width;
    args.height = rtOutput.m_finalOutputExtent.height;
    args.count = std::min(uint32_t(m_localStars.size()), uint32_t(std::clamp(sourceBudget(), 1, int(REMIXAPI_LOCAL_STARBURST_LIMIT))));
    args.sizeScale = std::clamp(angularScale(), 0.01f, 4.0f);
    args.brightness = std::clamp(brightness(), 0.0f, 20.0f);
    args.endpointBias = std::clamp(endpointBias(), 0.0f, 0.50f) * RtxOptions::getMeterToWorldUnitScale();
    if (args.count < m_localStars.size()) {
      std::partial_sort(m_localStars.begin(), m_localStars.begin() + args.count, m_localStars.end(),
        [](const auto& a, const auto& b) {
          const auto score = [](const auto& s) {
            return s.opacity * (0.2126f * s.color[0] + 0.7152f * s.color[1] + 0.0722f * s.color[2]);
          };
          const float left = score(a);
          const float right = score(b);
          return left != right ? left > right : a.sourceId < b.sourceId;
        });
    }
    if (m_localStarInputs == nullptr) {
      DxvkBufferCreateInfo info {};
      info.size = REMIXAPI_LOCAL_STARBURST_LIMIT * sizeof(LocalStarburstInput);
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      m_localStarInputs = ctx->getDevice()->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "Local starburst sources");
      info.size = REMIXAPI_LOCAL_STARBURST_LIMIT * sizeof(LocalStarburstSprite);
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      m_localStarSprites = ctx->getDevice()->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "Visible local starbursts");
    }

    ctx->setFramePassStage(RtxFramePassStage::PostFX);
    ctx->writeToBuffer(m_localStarInputs, 0, args.count * sizeof(LocalStarburstInput), m_localStars.data());
    ctx->bindCommonRayTracingResources(rtOutput);
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx->pushConstants(0, sizeof(args), &args);
    ctx->bindResourceBuffer(STARBURST_INPUT, DxvkBufferSlice(m_localStarInputs));
    ctx->bindResourceBuffer(STARBURST_SPRITES, DxvkBufferSlice(m_localStarSprites));
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LocalStarburstPrepareShader::getShader());
    ctx->dispatch((args.count + 31) / 32, 1, 1);

    ctx->bindResourceView(STARBURST_COLOR, rtOutput.m_finalOutput.view(Resources::AccessType::ReadWrite), nullptr);
    ctx->bindResourceSampler(STARBURST_MASK, ctx->getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE));
    ctx->bindResourceView(STARBURST_MASK, m_localStarMask.getImageView(), nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LocalStarburstCompositeShader::getShader());
    ctx->dispatch((args.width + 15) / 16, (args.height + 15) / 16, 1);
    ONCE(Logger::info("[Starbursts] Native local starburst pass dispatched for a current-frame source snapshot."));
  }

  void DxvkPostFx::showLocalStarburstSettings() {
    constexpr auto flags = ImGuiSliderFlags_AlwaysClamp;
    RemixGui::DragFloat("Brightness", &brightnessObject(), 0.02f, 0.0f, 20.0f, "%.2f", flags);
    RemixGui::DragFloat("Star size", &angularScaleObject(), 0.02f, 0.01f, 4.0f, "%.2f", flags);
    RemixGui::DragInt("Source budget", &sourceBudgetObject(), 1.0f, 1, REMIXAPI_LOCAL_STARBURST_LIMIT, "%d", flags);
    RemixGui::DragFloat("Occlusion endpoint bias (m)", &endpointBiasObject(), 0.002f, 0.0f, 0.50f, "%.3f", flags);
    RemixGui::InputText("Star mask texture", &maskTextureObject());
  }
}
