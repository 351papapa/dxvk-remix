#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "rtx_postFx.h"
#include "rtx_context.h"
#include "rtx_asset_data_manager.h"
#include "rtx_texture_manager.h"
#include "rtx_shader_manager.h"
#include "rtx_imgui.h"
#include "rtx_water.h"
#include "dxvk_scoped_annotation.h"
#include "../util/util_global_time.h"
#include "rtx/pass/sun_flare.h"
#include <rtx_shaders/sun_flare_state.h>
#include <rtx_shaders/sun_flare.h>

namespace dxvk {
  namespace {
    static_assert(sizeof(SunFlareStateArgs) == 48);
    static_assert(sizeof(SunFlareArgs) == 48);

    class SunFlareStateShader : public ManagedShader {
      SHADER_SOURCE(SunFlareStateShader, VK_SHADER_STAGE_COMPUTE_BIT, sun_flare_state)
      BINDLESS_ENABLED()
      PUSH_CONSTANTS(SunFlareStateArgs)
      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        RW_TEXTURE2D(SUN_FLARE_STATE)
      END_PARAMETER()
    };

    class SunFlareShader : public ManagedShader {
      SHADER_SOURCE(SunFlareShader, VK_SHADER_STAGE_COMPUTE_BIT, sun_flare)
      PUSH_CONSTANTS(SunFlareArgs)
      BEGIN_PARAMETER()
        TEXTURE2D(SUN_FLARE_STATE)
        RW_TEXTURE2D(SUN_FLARE_COLOR)
        SAMPLER2D(SUN_FLARE_VC_STAR)
        SAMPLER2D(SUN_FLARE_VC_HEX)
        SAMPLER2D(SUN_FLARE_VC_CIRCLE)
        SAMPLER2D(SUN_FLARE_VC_RING)
        SAMPLER2D(SUN_FLARE_SA_STAR)
      END_PARAMETER()
    };
  }

  void DxvkPostFx::dispatchSunFlare(Rc<RtxContext> ctx, const Resources::RaytracingOutput& rtOutput) {
    if (!enable() || !sunFlareEnable()) {
      m_sunFlareInitialized = false;
      return;
    }
    ScopedCpuProfileZone();
    ScopedGpuProfileZone(ctx, "Solar Lens Flare");
    ctx->setFramePassStage(RtxFramePassStage::PostFX);

    if (sunFlareTextureDirectory().empty()) {
      m_sunFlareInitialized = false;
      return;
    }
    if (m_sunFlareLoadedDirectory != sunFlareTextureDirectory()) {
      m_sunFlareLoadedDirectory = sunFlareTextureDirectory();
      constexpr std::array<const char*, 5> names = {
        "vc_coronastar.dds", "vc_coronahex.dds", "vc_coronacircle.dds", "vc_coronaringa.dds", "sa_coronastar.dds"
      };
      for (size_t i = 0; i < names.size(); ++i) {
        m_sunFlareTextures[i] = {};
        const std::string path = m_sunFlareLoadedDirectory + "/" + names[i];
        auto asset = AssetDataManager::get().findAsset(WaterSystem::resolveAssetPath(path));
        if (asset != nullptr) {
          m_sunFlareTextures[i] = TextureRef(ctx->getCommonObjects()->getTextureManager().preloadTextureAsset(asset, ColorSpace::AUTO, true));
        } else {
          Logger::warn("[SunFlare] Texture missing: " + path);
        }
      }
      m_sunFlareInitialized = false;
    }

    const TextureRef* fallback = nullptr;
    for (const auto& texture : m_sunFlareTextures) {
      if (!texture.isImageEmpty()) {
        fallback = &texture;
        break;
      }
    }
    if (fallback == nullptr) {
      m_sunFlareInitialized = false;
      return;
    }

    if (m_sunFlareState.image == nullptr) {
      Rc<DxvkContext> baseContext = ctx;
      m_sunFlareState = Resources::createImageResource(baseContext, "Sun visibility and camera water state",
        VkExtent3D { 2, 1, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
      m_sunFlareInitialized = false;
    }

    const RtCamera& camera = ctx->getSceneManager().getCamera();
    const uint64_t timeMs = GlobalTime::get().absoluteTimeMs();
    const float frameTime = GlobalTime::get().deltaTime();
    const bool reset = !m_sunFlareInitialized || camera.isCameraCut()
      || timeMs < m_sunFlareLastTimeMs || timeMs - m_sunFlareLastTimeMs > 250;
    SunFlareStateArgs stateArgs {};
    stateArgs.resetHistory = reset;
    stateArgs.visibilitySamples = 8;
    stateArgs.suppressUnderwater = sunFlareHideUnderwater();
    stateArgs.cloudOcclusion = sunFlareCloudOcclusion();
    const uint64_t waterHash = parseWaterMediumHash(WaterSystem::cameraMediumMaterialHash());
    stateArgs.cameraMediumState = waterHash == 0 ? 0u
      : (waterHash == ctx->getSceneManager().getStartInMediumMaterialSourceHash() ? 2u : 1u);
    stateArgs.deltaTime = std::isfinite(frameTime) ? std::clamp(frameTime, 0.0f, 0.1f) : 0.0f;
    stateArgs.fadeInSeconds = std::clamp(sunFlareFadeIn(), 0.001f, 5.0f);
    stateArgs.fadeOutSeconds = std::clamp(sunFlareFadeOut(), 0.001f, 5.0f);
    stateArgs.waterTransition = std::clamp(sunFlareWaterTransition(), 0.001f, 2.0f);
    stateArgs.maxRayDistance = 10000.0f;
    stateArgs.waterProbeDistance = 100.0f;
    stateArgs.edgeFade = 0.03f;
    ctx->bindCommonRayTracingResources(rtOutput);
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx->pushConstants(0, sizeof(stateArgs), &stateArgs);
    ctx->bindResourceView(SUN_FLARE_STATE, m_sunFlareState.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SunFlareStateShader::getShader());
    ctx->dispatch(1, 1, 1);
    m_sunFlareLastTimeMs = timeMs;
    m_sunFlareInitialized = true;

    SunFlareArgs args {};
    args.width = rtOutput.m_finalOutputExtent.width;
    args.height = rtOutput.m_finalOutputExtent.height;
    args.style = static_cast<uint32_t>(std::clamp(sunFlareStyle(), 0, 1));
    args.textureSet = static_cast<uint32_t>(std::clamp(sunFlareTextureSet(), 0, 1));
    args.brightness = std::clamp(sunFlareBrightness(), 0.0f, 20.0f);
    args.sizeScale = std::clamp(sunFlareSize(), 0.05f, 8.0f);
    args.vcAspectRatio = static_cast<float>(args.width) / std::max(args.height, 1u);
    args.timeSeconds = static_cast<float>(timeMs % 1000000u) * 0.001f;
    args.scaleWithResolution = sunFlareResolutionScale();
    args.saFlicker = sunFlareFlicker();
    args.frameIndex = m_sunFlareFrame++;
    ctx->pushConstants(0, sizeof(args), &args);
    ctx->bindResourceView(SUN_FLARE_STATE, m_sunFlareState.view, nullptr);
    ctx->bindResourceView(SUN_FLARE_COLOR, rtOutput.m_finalOutput.view(Resources::AccessType::ReadWrite), nullptr);
    constexpr uint32_t bindings[] = { SUN_FLARE_VC_STAR, SUN_FLARE_VC_HEX, SUN_FLARE_VC_CIRCLE, SUN_FLARE_VC_RING, SUN_FLARE_SA_STAR };
    const auto sampler = ctx->getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    for (size_t i = 0; i < m_sunFlareTextures.size(); ++i) {
      const auto& texture = m_sunFlareTextures[i].isImageEmpty() ? *fallback : m_sunFlareTextures[i];
      ctx->bindResourceView(bindings[i], texture.getImageView(), nullptr);
      ctx->bindResourceSampler(bindings[i], sampler);
    }
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SunFlareShader::getShader());
    ctx->dispatch((args.width + 7) / 8, (args.height + 7) / 8, 1);
  }

  void DxvkPostFx::showSunFlareImguiSettings() {
    constexpr auto flags = ImGuiSliderFlags_AlwaysClamp;
    RemixGui::Combo("Effect layout", &sunFlareStyleObject(), "Dense (26 ghosts)\0Classic (7 ghosts)\0");
    RemixGui::Combo("Texture set", &sunFlareTextureSetObject(), "Mixed shapes\0Soft star\0");
    RemixGui::DragFloat("Brightness", &sunFlareBrightnessObject(), 0.02f, 0.0f, 20.0f, "%.2f", flags);
    RemixGui::DragFloat("Ghost size", &sunFlareSizeObject(), 0.02f, 0.05f, 8.0f, "%.2f", flags);
    RemixGui::Checkbox("Scale with output resolution", &sunFlareResolutionScaleObject());
    RemixGui::Checkbox("Brightness flicker", &sunFlareFlickerObject());
    RemixGui::Checkbox("Hide underwater", &sunFlareHideUnderwaterObject());
    RemixGui::Checkbox("Cloud occlusion", &sunFlareCloudOcclusionObject());
    RemixGui::DragFloat("Fade in (seconds)", &sunFlareFadeInObject(), 0.01f, 0.001f, 5.0f, "%.3f", flags);
    RemixGui::DragFloat("Fade out (seconds)", &sunFlareFadeOutObject(), 0.01f, 0.001f, 5.0f, "%.3f", flags);
    RemixGui::DragFloat("Water exit fade (seconds)", &sunFlareWaterTransitionObject(), 0.01f, 0.001f, 2.0f, "%.3f", flags);
    RemixGui::InputText("Texture directory", &sunFlareTextureDirectoryObject());
    if (ImGui::Button("Reload flare textures")) {
      m_sunFlareLoadedDirectory.clear();
    }
    ImGui::TextWrapped("Follows the runtime atmosphere sun. Water visibility uses camera immersion even when underwater fog is disabled. Disable the game's original flare and the older Im3D flare plugin when using this effect.");
  }
}
