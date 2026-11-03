#include "rtx_water.h"
#include "rtx_water_medium.h"
#include "rtx_pre_exposure.h"
#include "rtx_host_calibration.h"
#include "rtx_context.h"
#include "rtx_shader_manager.h"
#include "rtx_asset_data_manager.h"
#include "rtx_texture_manager.h"
#include "dxvk_scoped_annotation.h"
#include "rtx/pass/water_caustics.h"
#include <rtx_shaders/water_caustics.h>
#include <rtx_shaders/water_caustics_local.h>
#include <rtx_shaders/water_caustics_solar.h>
#include <rtx_shaders/water_caustics_local_solar.h>
#include <rtx_shaders/water_caustics_composite.h>
#include <rtx_shaders/water_caustics_envelope.h>
#include <algorithm>
#include "../util/util_global_time.h"
#include "rtx/pass/water_foam.h"
#include <rtx_shaders/water_foam.h>
#include <rtx_shaders/water_foam_update.h>
#include <rtx_shaders/water_foam_update_culled.h>
#include <cstddef>
#include <cmath>
#include <cstring>
#include "rtx_water_foam_args.h"

namespace dxvk {
namespace {
static_assert(sizeof(WaterCausticsArgs) == 128);
static_assert(offsetof(WaterCausticsArgs, solarContrast) == 124);
static_assert(offsetof(WaterCausticsArgs, solarSurfaceHeight) == 16);
static_assert(offsetof(WaterCausticsArgs, solarProjectionEnabled) == 20);
static_assert(sizeof(WaterVolumeCausticsArgs) == 64);
static_assert(offsetof(WaterCausticsArgs, poolScale) == 44,
  "WaterCausticsArgs::poolScale must reuse the old padding word at offset 44");
static_assert(offsetof(WaterCausticsArgs, dispersion) == 120,
  "WaterCausticsArgs::dispersion must reuse the old localPadding0 word at offset 120");
static_assert(offsetof(WaterCausticsArgs, localLightIndices) == 80);
static_assert(offsetof(WaterCausticsArgs, localLightCount) == 112);
static_assert(offsetof(WaterCausticsArgs, fadeStart) == 116);
static_assert(offsetof(WaterCausticsArgs, lightDirection) == 48);
static_assert(offsetof(WaterCausticsArgs, lightValid) == 60);
static_assert(offsetof(WaterCausticsArgs, lightIrradiance) == 64);
// The push constant layout is a contract with both water_foam shaders. Reusing
// the former padding slot for skyAmbient must not have moved anything, so pin the
// size and the new member's offset here rather than trusting a manual count.
static_assert(sizeof(WaterFoamArgs) == 128);
static_assert(offsetof(WaterFoamArgs, runupWidth) == 72);
static_assert(offsetof(WaterFoamArgs, waveResponse) == 76);
static_assert(offsetof(WaterFoamArgs, lifetime) == 80);
static_assert(offsetof(WaterFoamArgs, wetnessLifetime) == 84);
static_assert(offsetof(WaterFoamArgs, textureScale) == 88);
static_assert(offsetof(WaterFoamArgs, time) == 92);
static_assert(offsetof(WaterFoamArgs, crestStrength) == 96);
static_assert(offsetof(WaterFoamArgs, crestThreshold) == 100);
static_assert(offsetof(WaterFoamArgs, oceanWaveStrength) == 104);
static_assert(offsetof(WaterFoamArgs, waveInputAvailable) == 108);
static_assert(offsetof(WaterFoamArgs, skyAmbient) == 15 * sizeof(uint32_t),
  "WaterFoamArgs::skyAmbient must occupy the former padding slot at byte offset 60");
static_assert(offsetof(WaterFoamArgs, sourceEnabled) == 16 * sizeof(uint32_t),
  "WaterFoamArgs::sourceEnabled must sit at byte offset 64");
static_assert(offsetof(WaterFoamArgs, excludeCharacters) == 17 * sizeof(uint32_t),
  "WaterFoamArgs::excludeCharacters must sit at byte offset 68");
static_assert(offsetof(WaterFoamArgs, reset) == 14 * sizeof(uint32_t),
  "WaterFoamArgs::reset must stay at byte offset 56");
struct WaterCausticSun {
  uint32_t index = kNewLightIdx;
  Vector3 direction;
  Vector3 irradiance;
};

WaterCausticSun selectWaterCausticSun(RtxContext* ctx) {
  WaterCausticSun sun {};
  float strongestFlux = 0.0f;
  for (const auto* light : ctx->getCommonObjects()->getSceneManager().getLightManager().getLinearizedLights()) {
    if (light->getType() != RtLightType::Distant || light->getBufferIdx() == kNewLightIdx) { continue; }
    const auto& distant = light->getDistantLight();
    const auto direction = -distant.getDirection();
    const auto radiance = distant.getRadiance();
    const float height = RtxOptions::zUp() ? direction.z : direction.y;
    const float flux = height * (0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z);
    if (height <= 0.0f || !std::isfinite(flux) || flux <= strongestFlux) { continue; }
    strongestFlux = flux;
    sun.index = light->getBufferIdx();
    sun.direction = direction;
    // Integrating the projected distant-light cone gives pi times stored radiance.
    const float encodeScale = 3.14159265f * PreExposure::get() * (light->isSkyProviderLight ? 1.0f : HostCalibration::lightScale());
    sun.irradiance = radiance * encodeScale;
  }
  return sun;
}

float finiteCausticSetting(float value, float minimum, float maximum) {
  return std::isfinite(value) ? std::clamp(value, minimum, maximum) : minimum;
}

class WaterCausticEnvelopeShader : public ManagedShader {
  SHADER_SOURCE(WaterCausticEnvelopeShader, VK_SHADER_STAGE_COMPUTE_BIT, water_caustics_envelope)
  BEGIN_PARAMETER()
    SAMPLER2D(0)
    RW_TEXTURE2D(1)
  END_PARAMETER()
};

Rc<DxvkImageView> createWaterCausticMipChain(RtxContext* ctx, const Rc<DxvkImageView>& source) {
  const auto& sourceInfo = source->info();
  const auto sourceExtent = source->mipLevelExtent(0);
  if (sourceInfo.type != VK_IMAGE_VIEW_TYPE_2D || sourceInfo.aspect != VK_IMAGE_ASPECT_COLOR_BIT
      || sourceInfo.numLayers != 1u || sourceExtent.width == 0u || sourceExtent.height == 0u) {
    Logger::warn("[Water volume caustics] Unsupported envelope source view");
    return nullptr;
  }

  // POT dimensions make every mip reduction an area-preserving 2x2 average.
  VkExtent3D extent {1u, 1u, 1u};
  while (extent.width < sourceExtent.width) { extent.width <<= 1u; }
  while (extent.height < sourceExtent.height) { extent.height <<= 1u; }
  const uint32_t maxDimension = ctx->getDevice()->adapter()->deviceProperties().limits.maxImageDimension2D;
  if (extent.width > maxDimension || extent.height > maxDimension) {
    Logger::warn("[Water volume caustics] Envelope exceeds device image limits");
    return nullptr;
  }
  uint32_t mipLevels = 1u;
  for (uint32_t size = std::max(extent.width, extent.height); size > 1u; size >>= 1u) {
    ++mipLevels;
  }
  Rc<DxvkContext> baseContext = ctx;
  auto resource = Resources::createImageResource(baseContext, "Water caustic shaped envelope", extent, VK_FORMAT_R16_SFLOAT,
    1u, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 0,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VkClearColorValue {}, mipLevels);
  resource.image->addShaderReadStages(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  auto mipInfo = resource.view->info();
  mipInfo.numLevels = 1u;
  const auto mipView = ctx->getDevice()->createImageView(resource.image, mipInfo);
  // Resource initialization ended the render pass. Source uploads are visible
  // before shaping, and generated levels declare their fragment reads.
  ctx->emitMemoryBarrier(0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
  ctx->bindResourceView(0, source, nullptr);
  ctx->bindResourceSampler(0, ctx->getResourceManager().getSampler(
    VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT));
  ctx->bindResourceView(1, mipView, nullptr);
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, WaterCausticEnvelopeShader::getShader());
  ctx->dispatch((extent.width + 7u) / 8u, (extent.height + 7u) / 8u, 1u);
  ctx->emitMemoryBarrier(0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
  ctx->generateMipmaps(resource.view, VK_FILTER_LINEAR);
  Logger::info(str::format("[Water volume caustics] Shaped envelope cache: ", extent.width, "x", extent.height,
    " levels=", mipLevels, " sourceCookie=", source->cookie()));
  return resource.view;
}

class WaterFoamShader : public ManagedShader {
  SHADER_SOURCE(WaterFoamShader, VK_SHADER_STAGE_COMPUTE_BIT, water_foam)
  BINDLESS_ENABLED()
  PUSH_CONSTANTS(WaterFoamArgs)
  BEGIN_PARAMETER()
    COMMON_RAYTRACING_BINDINGS
    RW_TEXTURE2D(WATER_FOAM_COLOR)
    TEXTURE2D(WATER_FOAM_FIELD)
    SAMPLER2D(WATER_FOAM_PATTERN)
    SAMPLERCUBE(WATER_FOAM_SKY)
    SAMPLER(WATER_FOAM_ENVIRONMENT_SAMPLER)
    TEXTURE2D(BINDING_WATER_FOAM_SURFACE_HEIGHT)
  END_PARAMETER()
};
class WaterFoamUpdateShader : public ManagedShader {
  SHADER_SOURCE(WaterFoamUpdateShader, VK_SHADER_STAGE_COMPUTE_BIT, water_foam_update)
  BINDLESS_ENABLED()
  PUSH_CONSTANTS(WaterFoamArgs)
  BEGIN_PARAMETER()
    COMMON_RAYTRACING_BINDINGS
    TEXTURE2D(WATER_FOAM_FIELD)
    RW_TEXTURE2D(WATER_FOAM_NEXT)
    STRUCTURED_BUFFER(WATER_FOAM_WAVES)
    STRUCTURED_BUFFER(WATER_FOAM_SHORE)
    STRUCTURED_BUFFER(WATER_FOAM_BEACH_MASK)
    TEXTURE2D(WATER_FOAM_PREVIOUS_HEIGHT)
    // The water height / validity output. COMMON_RAYTRACING_BINDINGS does NOT cover this
    // binding, and calling bindResourceView alone does not declare anything either: the
    // RW_TEXTURE2D macro is what tells the shader manager this is a STORAGE_IMAGE 2D view
    // written by this pass, which is what makes the resource tracker see the write and
    // order the next frame's read against it. Omitting it left binding 224 present in the
    // SPIR-V and bound on the CPU while the tracker knew nothing about it.
    RW_TEXTURE2D(BINDING_WATER_FOAM_SURFACE_HEIGHT)
  END_PARAMETER()
};
class WaterCausticsShader : public ManagedShader {
  SHADER_SOURCE(WaterCausticsShader, VK_SHADER_STAGE_COMPUTE_BIT, water_caustics)
  BINDLESS_ENABLED()
  PUSH_CONSTANTS(WaterCausticsArgs)
  BEGIN_PARAMETER()
    COMMON_RAYTRACING_BINDINGS
    TEXTURE2D(CAUSTICS_PRIMARY_POSITION)
    TEXTURE2D(CAUSTICS_SECONDARY_POSITION)
    TEXTURE2D(CAUSTICS_PRIMARY_NORMAL)
    TEXTURE2D(CAUSTICS_SECONDARY_NORMAL)
    TEXTURE2D(CAUSTICS_PRIMARY_ALBEDO)
    TEXTURE2D(CAUSTICS_SECONDARY_ALBEDO)
    TEXTURE2D(CAUSTICS_FLAGS)
    TEXTURE2D(CAUSTICS_MATERIAL)
    TEXTURE2D(CAUSTICS_PDF)
    TEXTURE2D(CAUSTICS_PRIMARY_CONE)
    TEXTURE2D(CAUSTICS_SECONDARY_CONE)
    TEXTURE2D(CAUSTICS_ACTIVE_PIXELS)
    SAMPLER2D(CAUSTICS_PATTERN)
    STRUCTURED_BUFFER(CAUSTICS_SURFACE_INPUT)
    SAMPLER2D(CAUSTICS_ENVELOPE)
    RW_TEXTURE2D(CAUSTICS_CACHE)
  END_PARAMETER()
};
class WaterCausticsCompositeShader : public ManagedShader {
  SHADER_SOURCE(WaterCausticsCompositeShader, VK_SHADER_STAGE_COMPUTE_BIT, water_caustics_composite)
  BINDLESS_ENABLED()
  PUSH_CONSTANTS(WaterCausticsArgs)
  BEGIN_PARAMETER()
    COMMON_RAYTRACING_BINDINGS
    TEXTURE2D(CAUSTICS_ACTIVE_PIXELS)
    RW_TEXTURE2D(CAUSTICS_CACHE)
    RW_TEXTURE2D(CAUSTICS_PRIMARY_RADIANCE)
    RW_TEXTURE2D(CAUSTICS_SECONDARY_RADIANCE)
  END_PARAMETER()
};
}

void WaterSystem::prepareCausticTexture(RtxContext* ctx) {
  if (m_causticTexturePath != causticsTexture()) {
    m_causticTexturePath = causticsTexture();
    m_causticTexture = {};
    auto asset = AssetDataManager::get().findAsset(resolveAssetPath(m_causticTexturePath));
    if (asset != nullptr) {
      m_causticTexture = TextureRef(ctx->getCommonObjects()->getTextureManager().preloadTextureAsset(asset, ColorSpace::AUTO, true));
      Logger::info("[WaterCaustics] Receiver pattern requested: " + m_causticTexturePath);
    } else {
      Logger::warn("[WaterCaustics] Pattern missing: " + m_causticTexturePath);
    }
  }
}

void WaterSystem::prepareVolumeCaustics(RtxContext* ctx) {
  WaterVolumeCausticsArgs args {};
  uint64_t sourceCookie = 0;
  Rc<DxvkImageView> pattern;
  Rc<DxvkBuffer> surfaceInput;
  WaterLightSurface surface;
  bool active = false;
  const float contrast = finiteCausticSetting(volumetricCausticsStrength(), 0.0f, 1.0f);
  const bool requested = enabled() && causticsEnabled() && contrast > 0.0f && causticsStrength() > 0.0f;
  if (requested && RtxOptions::zUp() && gpuSurface() && m_gpuSurface != nullptr) {
    surface.valid = m_gpuSurface->surfaceHeightAt(ctx->getSceneManager().getCamera().getPosition(), surface.height);
    if (surface.valid) {
      prepareCausticTexture(ctx);
      pattern = m_causticTexture.getImageView();
      surfaceInput = m_gpuSurface->waterLevelBuffer();
      const auto sun = selectWaterCausticSun(ctx);
      if (pattern != nullptr && surfaceInput != nullptr && sun.index != kNewLightIdx) {
        sourceCookie = pattern->cookie();
        if (m_volumeCausticPattern != nullptr && m_volumeCausticSourceCookie == sourceCookie) {
          pattern = m_volumeCausticPattern;
        } else {
          pattern = createWaterCausticMipChain(ctx, pattern);
        }
      } else {
        pattern = nullptr;
      }
      active = m_cameraMedium.active && pattern != nullptr;
      if (active) {
        args.strength = contrast;
        args.scale = finiteCausticSetting(causticsScale(), 0.5f, 16.0f);
        args.speed = finiteCausticSetting(causticsSpeed(), 0.0f, 3.0f);
        args.maxDepth = finiteCausticSetting(causticsMaxDepth(), 0.5f, 30.0f);
        args.randomized = causticsRandomized();
        args.sunIndex = sun.index;
        args.surfaceHeight = surface.height;
        args.fadeStart = finiteCausticSetting(causticsFadeStart(), 0.0f, 0.9f);
        args.receiverStrength = finiteCausticSetting(causticsStrength(), 0.0f, 8.0f);
        args.textureMaxLod = float(pattern->info().numLevels - 1u);
        const auto extent = pattern->mipLevelExtent(0);
        args.textureSize = {float(extent.width), float(extent.height)};
      }
    }
  }
  if (pattern == nullptr) {
    sourceCookie = 0;
  }
  // The receiver shares this cache above water, independently of volume activation.
  // Light buffer indices can be reordered each frame without changing the mask.
  auto previousKey = m_volumeCausticsArgs;
  auto currentKey = args;
  previousKey.sunIndex = currentKey.sunIndex = 0;
  m_volumeCausticsChanged = active != m_volumeCausticsActive
    || (active && (std::memcmp(&previousKey, &currentKey, sizeof(args)) != 0 || pattern != m_volumeCausticPattern
      || surfaceInput != m_volumeCausticSurface));
  if (m_volumeCausticsChanged) {
    Logger::info(str::format("[Water volume caustics] active=", active, " contrast=", args.strength,
      " sunIndex=", args.sunIndex, " scale=", args.scale, " speed=", args.speed));
  }
  m_volumeCausticsArgs = args;
  m_volumeCausticsActive = active;
  m_causticLightSurface = surface;
  m_volumeCausticPattern = pattern;
  m_volumeCausticSourceCookie = sourceCookie;
  m_volumeCausticSurface = surfaceInput;
}

void WaterSystem::bindVolumeCausticsResources(RtxContext* ctx) {
  ctx->bindResourceView(VOLUME_WATER_CAUSTICS_PATTERN, m_volumeCausticPattern, nullptr);
  ctx->bindResourceSampler(VOLUME_WATER_CAUSTICS_PATTERN,
    ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT));
  ctx->bindResourceBuffer(VOLUME_WATER_CAUSTICS_SURFACE_INPUT, DxvkBufferSlice(m_volumeCausticSurface));
}

void WaterSystem::prepareCaustics(RtxContext* ctx, const Resources::RaytracingOutput& output) {
  ScopedCpuProfileZone();
  m_causticsPrepared = false;
  if (!enabled() || ((!causticsEnabled() || causticsStrength() <= 0.0f) && (!reflectedCausticsEnabled() || reflectedCausticsStrength() <= 0.0f))) {
    m_causticRadiance.reset();
    return;
  }
  prepareCausticTexture(ctx);
  if (m_causticTexture.isImageEmpty()) { return; }
  const auto extent = output.m_compositeOutputExtent;
  if (m_causticRadiance.image == nullptr || m_causticRadiance.image->info().extent.width != extent.width || m_causticRadiance.image->info().extent.height != extent.height) {
    Rc<DxvkContext> baseContext = ctx;
    m_causticRadiance = Resources::createImageResource(baseContext, "Water receiver caustics", extent, VK_FORMAT_R16G16B16A16_SFLOAT);
    Logger::info("[WaterCaustics] Receiver texture resident; caustics passes ready");
  }
  ScopedGpuProfileZone(ctx, "Water Receiver Caustics");
  ctx->bindCommonRayTracingResources(output);
  ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
  WaterCausticsArgs args {};
  args.strength = causticsEnabled() ? std::clamp(causticsStrength(), 0.0f, 8.0f) : 0.0f;
  args.randomized = causticsRandomized() ? 1u : 0u;
  args.reflectedStrength = reflectedCausticsEnabled() ? std::clamp(reflectedCausticsStrength(), 0.0f, 8.0f) : 0.0f;
  args.reflectedHeight = std::clamp(reflectedCausticsHeight(), 0.5f, 15.0f);
  args.scale = std::clamp(causticsScale(), 0.5f, 16.0f);
  // Per-domain tiling. The shader picks between args.scale (shared-protocol ocean) and
  // args.poolScale (every other animated water surface) from the surface the
  // refracted ray hit. When the split is off both carry the ocean value, so the
  // shader needs no extra branch and the behaviour is unchanged.
  args.poolScale = causticsScalePerDomain()
    ? std::clamp(causticsPoolScale(), 0.2f, 16.0f)
    : args.scale;
  // Chromatic dispersion. The shader evaluates the pattern three times and drives
  // R / G / B separately, so 0 keeps the old single-tap cost exactly.
  args.dispersion = std::clamp(causticsDispersion(), 0.0f, 2.0f);
  args.speed = std::clamp(causticsSpeed(), 0.0f, 3.0f);
  args.maxDepth = std::clamp(causticsMaxDepth(), 0.5f, 30.0f);
  args.solarContrast = finiteCausticSetting(volumetricCausticsStrength(), 0.0f, 1.0f);
  args.solarSurfaceHeight = m_causticLightSurface.height;
  args.solarProjectionEnabled = m_causticLightSurface.valid && m_volumeCausticSurface != nullptr
    && m_volumeCausticPattern != nullptr;
  if (m_volumeCausticPattern != nullptr) {
    const auto envelopeExtent = m_volumeCausticPattern->mipLevelExtent(0);
    args.solarPadding = {float(m_volumeCausticPattern->info().numLevels - 1u),
      float(std::max(envelopeExtent.width, envelopeExtent.height))};
  }
  args.fadeStart = std::clamp(causticsFadeStart(), 0.0f, 0.9f);
  args.localLightRange = std::clamp(causticsLocalLightRange(), 5.0f, 200.0f);
  const uint32_t localLimit = static_cast<uint32_t>(std::clamp(causticsLocalLights(), 0, 8));
  const auto cameraPosition = ctx->getCommonObjects()->getSceneManager().getCamera().getPosition();
  float localScores[8] {};
  if (localLimit > 0) {
    for (const auto* light : ctx->getCommonObjects()->getSceneManager().getLightManager().getLinearizedLights()) {
      if (light->getType() >= RtLightType::Distant || light->getBufferIdx() == kNewLightIdx) { continue; }
      const auto offset = light->getPosition() - cameraPosition;
      const float distanceSquared = dot(offset, offset);
      if (distanceSquared >= args.localLightRange * args.localLightRange) { continue; }
      float area = 0.0f;
      switch (light->getType()) {
      case RtLightType::Sphere: {
        const float radius = light->getSphereLight().getRadius();
        area = 3.14159265f * radius * radius;
        break;
      }
      case RtLightType::Rect: {
        const auto size = light->getRectLight().getDimensions();
        area = size.x * size.y;
        break;
      }
      case RtLightType::Disk: {
        const auto size = light->getDiskLight().getHalfDimensions();
        area = 3.14159265f * size.x * size.y;
        break;
      }
      case RtLightType::Cylinder:
        area = 2.0f * light->getCylinderLight().getRadius() * light->getCylinderLight().getAxisLength();
        break;
      default: break;
      }
      const auto radiance = light->getRadiance();
      const float score = (0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z) * area / std::max(1.0f, distanceSquared);
      if (!std::isfinite(score) || score <= 0.0f) { continue; }
      for (uint32_t slot = 0; slot < localLimit; ++slot) {
        if (score <= localScores[slot]) { continue; }
        for (uint32_t next = localLimit - 1; next > slot; --next) {
          localScores[next] = localScores[next - 1];
          args.localLightIndices[next] = args.localLightIndices[next - 1];
        }
        localScores[slot] = score;
        args.localLightIndices[slot] = light->getBufferIdx();
        args.localLightCount = std::min(localLimit, args.localLightCount + 1);
        break;
      }
    }
  }
  const auto sun = selectWaterCausticSun(ctx);
  args.lightDirection = {sun.direction.x, sun.direction.y, sun.direction.z};
  args.lightIrradiance = {sun.irradiance.x, sun.irradiance.y, sun.irradiance.z};
  args.lightValid = sun.index != kNewLightIdx;
  ctx->pushConstants(0, sizeof(args), &args);
  ctx->bindResourceView(CAUSTICS_PRIMARY_POSITION, output.getCurrentPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Read), nullptr);
  ctx->bindResourceView(CAUSTICS_SECONDARY_POSITION, output.m_secondaryWorldPositionWorldTriangleNormal.view(Resources::AccessType::Read), nullptr);
  ctx->bindResourceView(CAUSTICS_PRIMARY_NORMAL, output.m_primaryWorldShadingNormal.view, nullptr);
  ctx->bindResourceView(CAUSTICS_SECONDARY_NORMAL, output.m_secondaryWorldShadingNormal.view, nullptr);
  ctx->bindResourceView(CAUSTICS_PRIMARY_ALBEDO, output.m_primaryAlbedo.view, nullptr);
  ctx->bindResourceView(CAUSTICS_SECONDARY_ALBEDO, output.m_secondaryAlbedo.view, nullptr);
  ctx->bindResourceView(CAUSTICS_FLAGS, output.m_sharedFlags.view, nullptr);
  ctx->bindResourceView(CAUSTICS_MATERIAL, output.m_sharedMaterialData0.view, nullptr);
  ctx->bindResourceView(CAUSTICS_PDF, output.m_sharedIntegrationSurfacePdf.view(Resources::AccessType::Read), nullptr);
  ctx->bindResourceView(CAUSTICS_PRIMARY_CONE, output.m_primaryConeRadius.view, nullptr);
  ctx->bindResourceView(CAUSTICS_SECONDARY_CONE, output.m_secondaryConeRadius.view(Resources::AccessType::Read), nullptr);
  ctx->bindResourceView(CAUSTICS_ACTIVE_PIXELS, output.m_sparseRenderingActiveLocalPixelCoords.view, nullptr);
  ctx->bindResourceView(CAUSTICS_ENVELOPE, m_volumeCausticPattern, nullptr);
  ctx->bindResourceSampler(CAUSTICS_ENVELOPE, ctx->getResourceManager().getSampler(
    VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT));
  ctx->bindResourceView(CAUSTICS_PATTERN, m_causticTexture.getImageView(), nullptr);
  ctx->bindResourceSampler(CAUSTICS_PATTERN, ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT));
  ctx->bindResourceView(CAUSTICS_CACHE, m_causticRadiance.view, nullptr);
  ctx->bindResourceBuffer(CAUSTICS_SURFACE_INPUT, m_volumeCausticSurface != nullptr
    ? DxvkBufferSlice(m_volumeCausticSurface) : DxvkBufferSlice());
  const bool solarProjection = args.solarProjectionEnabled && args.lightValid;
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, solarProjection
    ? (args.localLightCount > 0
      ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterCausticsShader, water_caustics_local_solar)
      : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterCausticsShader, water_caustics_solar))
    : (args.localLightCount > 0
      ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterCausticsShader, water_caustics_local)
      : WaterCausticsShader::getShader()));
  ctx->dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
  m_causticsPrepared = true;
}

void WaterSystem::compositeCaustics(RtxContext* ctx, const Resources::RaytracingOutput& output) {
  if (!m_causticsPrepared) { return; }
  ScopedGpuProfileZone(ctx, "Water Caustics Composite");
  ctx->bindCommonRayTracingResources(output);
  ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
  ctx->bindResourceView(CAUSTICS_ACTIVE_PIXELS, output.m_sparseRenderingActiveLocalPixelCoords.view, nullptr);
  ctx->bindResourceView(CAUSTICS_CACHE, m_causticRadiance.view, nullptr);
  ctx->bindResourceView(CAUSTICS_PRIMARY_RADIANCE, output.m_primaryDirectDiffuseRadiance.view(Resources::AccessType::ReadWrite), nullptr);
  ctx->bindResourceView(CAUSTICS_SECONDARY_RADIANCE, output.m_secondaryCombinedDiffuseRadiance.view(Resources::AccessType::ReadWrite), nullptr);
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, WaterCausticsCompositeShader::getShader());
  const auto extent = output.m_compositeOutputExtent;
  ctx->dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
}
void WaterSystem::fillShoreWetnessArgs(WetnessArgs& args) {
  // Freeze both resource index and domain before ray tracing. compositeFoam swaps
  // the simulation buffers later, while later passes still use this frame's CB.
  m_shoreWetnessReady = enabled() && m_foamReady && shoreWetnessStrength() > 0.0f;
  m_shoreWetnessIndex = m_foamIndex;
  args.shoreOriginX = m_foamOrigin.x;
  args.shoreOriginY = m_foamOrigin.y;
  args.shoreInvSpacing = 1.0f / 0.4f;
  args.shoreStrength = m_shoreWetnessReady ? std::clamp(shoreWetnessStrength(), 0.0f, 1.0f) : 0.0f;
}

void WaterSystem::bindShoreWetnessResources(RtxContext* ctx) {
  ctx->bindResourceView(BINDING_SHORE_WETNESS_FIELD,
    m_shoreWetnessReady ? m_foamField[m_shoreWetnessIndex].view : nullptr, nullptr);
  ctx->bindResourceView(BINDING_SHORE_WETNESS_HEIGHT,
    m_shoreWetnessReady ? m_foamHeight[m_shoreWetnessIndex].view : nullptr, nullptr);
}

void WaterSystem::compositeFoam(RtxContext* ctx, const Resources::RaytracingOutput& output) {
  if (!enabled() || (!shoreFoamEnabled() && shoreWetnessStrength() <= 0.0f)) { m_foamReady = false; return; }
  // Contact foam master switch. The player is excluded separately, in the shader,
  // by surface class, so shoreline / rock / pier foam is unaffected by that.
  const bool seedContact = shoreFoamEnabled() && contactFoam();
  if (m_foamCompositePath != foamCompositeTexture()) {
    m_foamCompositePath = foamCompositeTexture();
    m_foamCompositeTexture = {};
    auto asset = AssetDataManager::get().findAsset(resolveAssetPath(m_foamCompositePath));
    if (asset != nullptr) {
      m_foamCompositeTexture = TextureRef(ctx->getCommonObjects()->getTextureManager().preloadTextureAsset(asset, ColorSpace::AUTO, true));
    }
  }
  constexpr uint32_t fieldSize = 257;
  if (m_foamField[0].image == nullptr) {
    Rc<DxvkContext> base = ctx;
    m_foamField[0] = Resources::createImageResource(base, "Shore foam density A", {fieldSize, fieldSize, 1}, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_foamField[1] = Resources::createImageResource(base, "Shore foam density B", {fieldSize, fieldSize, 1}, VK_FORMAT_R16G16B16A16_SFLOAT);
    // Water and terrain each keep a 32-bit height and separate validity.
    m_foamHeight[0] = Resources::createImageResource(base, "Shore foam water and terrain A", {fieldSize, fieldSize, 1}, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_foamHeight[1] = Resources::createImageResource(base, "Shore foam water and terrain B", {fieldSize, fieldSize, 1}, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_foamReady = false;
  }
  ScopedGpuProfileZone(ctx, "Water Contact Noise Foam");
  ctx->bindCommonRayTracingResources(output);
  ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
  // Every field of WaterFoamArgs is assigned here, in one block, through a fill
  // whose signature is regression-tested. `WaterFoamArgs args {}` zero-initialises,
  // so any field left out silently becomes 0 in the shader: that is exactly how all
  // intersection foam once disappeared, when the strength assignment was deleted by
  // accident and the display shader's `saturate(density * erosion * pattern *
  // args.strength)` became identically 0 with no error and no log.
  //
  // Source gating is a separate field and never lineWidth = 0: a zero width made
  // the update shader evaluate contact / lineWidth as 0/0, which read as "every
  // cell touches the contact" and saturated the whole field in under a second.
  WaterFoamBuilder::WaterFoamFill fill {};
  const auto position = ctx->getCommonObjects()->getSceneManager().getCamera().getPosition();
  const float centerY = RtxOptions::zUp() ? position.y : position.z;
  fill.width = output.m_finalOutputExtent.width;
  fill.height = output.m_finalOutputExtent.height;
  fill.lineWidth = std::clamp(shoreFoamWidth(), 0.1f, 2.0f);
  fill.strength = shoreFoamEnabled() ? std::clamp(shoreFoamStrength(), 0.0f, 1.0f) : 0.0f;
  fill.maxDistance = 100.0f;
  fill.maxMotion = 0.35f * std::clamp(GlobalTime::get().realDeltaTime(), 0.001f, 0.1f);
  fill.oceanOnly = shoreFoamOceanOnly() ? 1u : 0u;
  fill.fieldSize = fieldSize;
  fill.originX = std::floor(position.x / 0.4f) * 0.4f - (fieldSize / 2) * 0.4f;
  fill.originY = std::floor(centerY / 0.4f) * 0.4f - (fieldSize / 2) * 0.4f;
  fill.previousX = m_foamOrigin.x;
  fill.previousY = m_foamOrigin.y;
  fill.spacing = 0.4f;
  const float waterTime = m_gpuSurface ? m_gpuSurface->simulationTime() : -1.0f;
  fill.time = waterTime >= 0.0f ? waterTime
    : (static_cast<uint32_t>(GlobalTime::get().absoluteTimeMs()) & ((1U << 24U) - 1U)) / 1000.f;
  const float elapsed = fill.time - m_foamSnapshotTime;
  fill.reset = m_foamReady && elapsed >= 0.0f && elapsed < 0.5f ? 0u : 1u;
  fill.dt = fill.reset == 0u ? std::clamp(elapsed, 0.0f, 0.1f) : 0.0f;
  fill.runupWidth = std::clamp(shoreFoamRunupWidth(), 0.0f, 6.0f);
  fill.waveResponse = std::clamp(shoreFoamWaveResponse(), 0.0f, 8.0f);
  fill.lifetime = std::clamp(shoreFoamLifetime(), 0.5f, 15.0f);
  fill.wetnessLifetime = std::clamp(shoreWetnessLifetime(), 1.0f, 120.0f);
  fill.crestStrength = shoreFoamEnabled() ? std::clamp(crestFoamStrength(), 0.0f, 6.0f) : 0.0f;
  fill.crestThreshold = std::clamp(crestFoamThreshold(), 0.01f, 0.4f);
  const Rc<DxvkBuffer> waveInput = m_gpuSurface ? m_gpuSurface->waveInputBuffer() : nullptr;
  fill.waveInputAvailable = waveInput.ptr() != nullptr ? 1u : 0u;
  fill.shoreHeight = beachWaves() ? std::clamp(beachWaveHeight(), 0.0f, 2.0f) : 0.0f;
  fill.shoreSpacing = std::clamp(beachWaveLength(), 2.0f, 100.0f);
  fill.shorePeriod = std::clamp(beachWavePeriod(), 1.0f, 30.0f);
  fill.shoreDepth = std::clamp(beachWaveDepth(), 0.25f, 12.0f);
  fill.oceanWaveStrength = waveInput.ptr() != nullptr ? m_gpuSurface->oceanWaveStrength() : 0.0f;
  fill.textureScale = std::clamp(shoreFoamTextureScale(), 0.25f, 12.0f);
  fill.skyAmbient = std::clamp(shoreFoamSkyAmbient(), 0.0f, 2.0f);
  // Two independent flags, on purpose. Overloading one with both meanings made an
  // earlier diagnostic unsound: with strength zeroed by the missing assignment, no
  // amount of toggling the player exclusion could show anything, so "still no foam
  // after disabling the exclusion" did not rule out a classification problem.
  fill.sourceEnabled = seedContact ? kWATER_FOAM_SOURCE_ON : kWATER_FOAM_SOURCE_OFF;
  fill.excludeCharacters = contactFoamPlayerExclusion()
    ? kWATER_FOAM_EXCLUDE_CHARACTERS_ON : kWATER_FOAM_EXCLUDE_CHARACTERS_OFF;
  WaterFoamArgs args = WaterFoamBuilder::makeWaterFoamArgs(fill);
  // Log the values actually uploaded, not the options they came from. A zero here
  // is what a dead display chain looks like from the CPU side, and reading it back
  // costs far less than inferring it from the screen.
  if ((m_foamArgsLogCounter++ % 600u) == 0u) {
    Logger::info(str::format(
      "[WaterFoam] upload width=", args.width, " height=", args.height,
      " strength=", args.strength, " lineWidth=", args.lineWidth,
      " sourceEnabled=", args.sourceEnabled, " excludeCharacters=", args.excludeCharacters,
      " reset=", args.reset, " dt=", args.dt, " maxMotion=", args.maxMotion,
      " fieldSize=", args.fieldSize, " spacing=", args.spacing,
      " oceanOnly=", args.oceanOnly, " skyAmbient=", args.skyAmbient,
      " runupWidth=", args.runupWidth, " waveResponse=", args.waveResponse,
      " lifetime=", args.lifetime, " wetnessLifetime=", args.wetnessLifetime,
      " crestStrength=", args.crestStrength, " crestThreshold=", args.crestThreshold,
      " analyticWaves=", args.waveInputAvailable, " waveGain=", args.oceanWaveStrength,
      " textureScale=", args.textureScale, " gpuWaterClock=", waterTime >= 0.0f,
      " fullySet=", WaterFoamBuilder::waterFoamArgsIsFullySet(args),
      " sizeofArgs=", sizeof(WaterFoamArgs)));
  }
  ctx->pushConstants(0, sizeof(args), &args);
  const uint32_t next = m_foamIndex ^ 1u;
  ctx->bindResourceView(WATER_FOAM_FIELD, m_foamField[m_foamIndex].view, nullptr);
  ctx->bindResourceView(WATER_FOAM_NEXT, m_foamField[next].view, nullptr);
  ctx->bindResourceView(BINDING_WATER_FOAM_SURFACE_HEIGHT, m_foamHeight[next].view, nullptr);
  ctx->bindResourceView(WATER_FOAM_PREVIOUS_HEIGHT, m_foamHeight[m_foamIndex].view, nullptr);
  ctx->bindResourceBuffer(WATER_FOAM_WAVES, waveInput != nullptr ? DxvkBufferSlice(waveInput) : DxvkBufferSlice());
  const Rc<DxvkBuffer> shoreDepth = m_gpuSurface ? m_gpuSurface->shoreDepthBuffer() : nullptr;
  ctx->bindResourceBuffer(WATER_FOAM_SHORE, shoreDepth != nullptr ? DxvkBufferSlice(shoreDepth) : DxvkBufferSlice());
  const Rc<DxvkBuffer> beachMask = m_gpuSurface ? m_gpuSurface->beachMaskBuffer() : nullptr;
  ctx->bindResourceBuffer(WATER_FOAM_BEACH_MASK, beachMask != nullptr ? DxvkBufferSlice(beachMask) : DxvkBufferSlice());
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, shoreFoamFrustumCulling()
    ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamUpdateShader, water_foam_update_culled)
    : WaterFoamUpdateShader::getShader());
  ctx->dispatch((fieldSize + 7) / 8, (fieldSize + 7) / 8, 1);
  m_foamIndex = next; m_foamReady = true;
  m_foamOrigin = Vector2(args.originX, args.originY);
  // Keep the material display's domain and clock matched to this GPU snapshot.
  {
    const float camUpNow = RtxOptions::zUp() ? position.z : position.y;
    m_foamSnapshotCameraUp = camUpNow;
    m_foamSnapshotTime = args.time;
  }

  // When the material path owns display, stop here. The density above has already been
  // advanced and the origin recorded, which is all the display surface needs; the
  // screen-space composite below is exactly the fake lighting this path replaces, so
  // running it as well would paint an unlit white line straight back over the lit foam.
  // Deliberately not an early return before the update: the contact detection and
  // density transport must keep running, they are what feeds the geometry.
  if (foamLineMaterialPath() || fill.strength <= 0.0f) {
    if ((m_materialPathLogCounter++ % 600u) == 0u) {
      Logger::info(str::format("[WaterFoam] material path owns display; screen-space composite skipped."
        " origin=(", args.originX, ",", args.originY, ") spacing=", args.spacing,
        " strength=", args.strength));
    }
    return;
  }

  ctx->bindResourceView(WATER_FOAM_FIELD, m_foamField[next].view, nullptr);
  ctx->bindResourceView(BINDING_WATER_FOAM_SURFACE_HEIGHT, m_foamHeight[next].view, nullptr);
  ctx->bindResourceView(WATER_FOAM_PATTERN, m_foamCompositeTexture.getImageView(), nullptr);
  ctx->bindResourceSampler(WATER_FOAM_PATTERN, ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT));
  ctx->bindResourceView(WATER_FOAM_SKY, ctx->getResourceManager().getSkyProbe(ctx).view, nullptr);
  ctx->bindResourceSampler(WATER_FOAM_SKY, ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE));
  ctx->bindResourceSampler(WATER_FOAM_ENVIRONMENT_SAMPLER, ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT));
  if (m_foamCompositeTexture.isImageEmpty()) { return; }
  ctx->bindResourceView(WATER_FOAM_COLOR, output.m_finalOutput.view(Resources::AccessType::ReadWrite), nullptr);
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamShader::getShader());
  ctx->dispatch((args.width + 7) / 8, (args.height + 7) / 8, 1);
}
}
