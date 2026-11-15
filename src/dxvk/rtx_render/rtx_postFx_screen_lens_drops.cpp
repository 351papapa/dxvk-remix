// Copyright 2026 Remix Plus contributors
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include "rtx_postFx.h"
#include "rtx_context.h"
#include "dxvk_scoped_annotation.h"
#include "rtx/pass/post_fx/lens_rain.h"

namespace dxvk {
  void DxvkPostFx::reportScreenLensDropTiming(const char* operation, uint32_t count, uint64_t microseconds) {
    if (m_screenLensDropTimings && m_screenLensDropTimingSamples.fetch_add(1) < 64) {
      Logger::info(str::format("[ScreenLensDrops] ", operation, " count=", count, " cpuUs=", microseconds));
    }
  }

  std::vector<LensRainDrop> DxvkPostFx::screenLensDropUploads(float aspect) const {
    std::vector<LensRainDrop> result;
    result.reserve(m_screenLensDrops.frame().drops.size());
    const float size = std::isfinite(gameDropSizeScale()) ? std::clamp(gameDropSizeScale(), 0.25f, 3.0f) : 1.0f;
    const float absorption = std::isfinite(bloodAbsorption()) ? std::clamp(bloodAbsorption(), 0.0f, 4.0f) : 1.0f;
    for (const auto& source : m_screenLensDrops.frame().drops) {
      LensRainDrop drop = {};
      drop.centerRadii = { source.u * aspect, source.v, source.radiusX * size, source.radiusY * size };
      drop.axisOpacityStrength = { source.axisX, source.axisY, source.opacity, source.variant == 0 ? 0.9f : 1.1f };
      drop.absorptionKind = source.kind == 2 ? float4(0.7f * absorption, 4.2f * absorption, 5.3f * absorption, 2.0f)
        : float4(0.0f, 0.0f, 0.0f, 1.0f);
      result.push_back(drop);
    }
    return result;
  }

  bool DxvkPostFx::dispatchPendingScreenLensDrops(Rc<RtxContext> ctx,
      const Rc<DxvkImage>& target, bool targetGammaEncoded) {
    if (!m_screenLensDrops.availableForPresent(ctx->getDevice()->getCurrentFrameId())) {
      m_screenLensLateSource.reset();
      m_screenLensLateOutput.reset();
      return false;
    }
    if (m_screenLensDrops.frame().drops.empty()) { m_screenLensDrops.consume(); return false; }
    if (screenLensCameraInWater(ctx)) {
      m_screenLensDrops.consume();
      m_screenLensLateSource.reset();
      m_screenLensLateOutput.reset();
      return false;
    }
    if (target == nullptr) { return false; }
    const auto& targetInfo = target->info();
    const auto* format = imageFormatInfo(targetInfo.format);
    const auto extent = target->mipLevelExtent(0);
    const bool multisampled = targetInfo.sampleCount != VK_SAMPLE_COUNT_1_BIT;
    // D3D9 scene backbuffers have transfer and sampled usage. MSAA writeback uses the
    // existing framebuffer blit, so its color-attachment usage is required as well.
    if (targetInfo.type != VK_IMAGE_TYPE_2D || extent.width == 0 || extent.height == 0 || extent.depth != 1
        || targetInfo.numLayers != 1 || format->aspectMask != VK_IMAGE_ASPECT_COLOR_BIT
        || format->flags.test(DxvkFormatFlag::SampledUInt) || format->flags.test(DxvkFormatFlag::SampledSInt)
        || !(targetInfo.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        || (multisampled ? !(targetInfo.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
                        : !(targetInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))) {
      ONCE(Logger::warn("Lens drops: current present target is not a supported color backbuffer; snapshot not consumed."));
      return false;
    }
    ScopedCpuProfileZone();
    ScopedGpuProfileZone(ctx, "Native Lens Drops Late Target");
    ctx->setFramePassStage(RtxFramePassStage::PostFX);
    const auto allocate = [&](Resources::Resource& resource, VkFormat imageFormat, bool output, const char* name) {
      if (resource.image != nullptr && resource.image->info().format == imageFormat
          && resource.image->info().extent.width == extent.width && resource.image->info().extent.height == extent.height) {
        return;
      }
      DxvkImageCreateInfo info;
      info.type = VK_IMAGE_TYPE_2D;
      info.flags = 0;
      info.format = imageFormat;
      info.extent = extent;
      info.numLayers = 1;
      info.mipLevels = 1;
      info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      info.layout = VK_IMAGE_LAYOUT_UNDEFINED;
      info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | (output ? VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        : VK_IMAGE_USAGE_TRANSFER_DST_BIT);
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      info.access = VK_ACCESS_SHADER_READ_BIT | (output ? VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT);
      resource.image = ctx->getDevice()->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXRenderTarget, name);
      DxvkImageViewCreateInfo view;
      view.type = VK_IMAGE_VIEW_TYPE_2D;
      view.format = imageFormat;
      view.usage = output ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT;
      view.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
      view.minLevel = 0; view.numLevels = 1; view.minLayer = 0; view.numLayers = 1;
      resource.view = ctx->getDevice()->createImageView(resource.image, view);
      ctx->changeImageLayout(resource.image, VK_IMAGE_LAYOUT_GENERAL);
    };
    allocate(m_screenLensLateSource, targetInfo.format, false, "Native lens drops current-target copy");
    // The common SDR/HDR backbuffers fit RGBA16F. Preserve a host's 32-bit float target if present.
    const VkFormat outputFormat = targetInfo.format == VK_FORMAT_R32G32B32A32_SFLOAT
      ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
    allocate(m_screenLensLateOutput, outputFormat, true, "Native lens drops late result");
    const VkImageSubresourceLayers layer = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    if (multisampled) {
      const VkImageResolve resolve = { layer, {0, 0, 0}, layer, {0, 0, 0}, extent };
      ctx->resolveImage(m_screenLensLateSource.image, target, resolve, targetInfo.format);
    } else {
      ctx->copyImage(m_screenLensLateSource.image, layer, {0, 0, 0}, target, layer, {0, 0, 0}, extent);
    }
    LensRainArgs args = {};
    args.imageSize = { extent.width, extent.height };
    args.invImageSize = { 1.0f / extent.width, 1.0f / extent.height };
    args.tileCount = { 32u, std::clamp(uint32_t(std::ceil(32.0f * extent.height / extent.width)), 8u, 64u) };
    args.aspect = float(extent.width) / extent.height;
    args.refractPixels = std::clamp(refractPixels(), 0.0f, 48.0f);
    args.dispersion = std::clamp(dispersion(), 0.0f, 0.5f);
    args.blurPixels = std::clamp(blurPixels(), 0.0f, 4.0f);
    const auto drops = screenLensDropUploads(args.aspect);
    args.externalCount = uint32_t(drops.size());
    args.tileDropCapacity = LENS_RAIN_TOTAL_DROPS;
    // Encoding describes values after texture sampling. SRGB image sampling and
    // destination blit encoding are handled by Vulkan and cancel in this round trip.
    args.colorFlags = { targetGammaEncoded ? 1u : 0u, 0, 0, 0 };
    dispatchLensRainImage(ctx, m_screenLensLateSource, m_screenLensLateOutput, args, drops);
    // Blit handles format conversion. For MSAA it selects a fullscreen framebuffer pass
    // and writes each destination sample; it does not issue an illegal vkCmdBlitImage.
    RtxContext::blitImageHelper(ctx, m_screenLensLateOutput.image, target, VK_FILTER_NEAREST);
    m_screenLensDrops.consume();
    return true;
  }
}
