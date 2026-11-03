// Adapted from xoxor4d/dxvk-remix, commit 7f5210e1a7cd1356bf0271230f82e6132b08ad8c.
// Modified for device-owned resources, frame validity and configurable world orientation and units.
// The original zlib/libpng license is retained in ThirdPartyLicenses.txt.

#include <algorithm>
#include <cmath>
#include <limits>

#include <rtx_shaders/rain_map_trace.h>
#include <rtx_shaders/rain_map_relief.h>

#include "rtx_rain_map.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_scene_manager.h"
#include "rtx_shader_manager.h"
#include "dxvk_scoped_annotation.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/rain_map/rain_map_binding_indices.h"
#include "rtx/pass/raytrace_args.h"

namespace dxvk {

  namespace {

    class RainMapTraceShader : public ManagedShader {
      SHADER_SOURCE(RainMapTraceShader, VK_SHADER_STAGE_COMPUTE_BIT, rain_map_trace)
      BINDLESS_ENABLED()
      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        RW_TEXTURE2D(RAIN_MAP_BINDING_HEIGHT_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(RainMapTraceShader);

    class RainMapReliefShader : public ManagedShader {
      SHADER_SOURCE(RainMapReliefShader, VK_SHADER_STAGE_COMPUTE_BIT, rain_map_relief)
      BINDLESS_ENABLED()
      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        TEXTURE2D(RAIN_MAP_BINDING_HEIGHT_INPUT)
        RW_TEXTURE2D(RAIN_MAP_BINDING_MAP_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(RainMapReliefShader);

  } // anonymous namespace

  void RtxRainMap::prepare(RtxContext& ctx, RaytraceArgs& constants) {
    ScopedCpuProfileZone();
    WetnessArgs& args = constants.wetnessArgs;
    m_enabled = WetnessOptions::enabled() && WetnessOptions::rainMapEnabled();
    m_preparedFrame = constants.frameIdx;
    m_traced = false;
    args.rainMapEnabled = m_enabled;
    const auto finite = [](float value, float fallback, float low, float high) {
      return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
    };
    const float units = finite(RtxOptions::getMeterToWorldUnitScale(), 100.0f, 0.0001f, 1000000.0f);
    args.rainOcclusionMinHeight = finite(WetnessOptions::rainOcclusionMinHeight(), 2.5f, 0.0f, 1000.0f) * units;
    args.rainOcclusionFeather = finite(WetnessOptions::rainOcclusionFeather(), 1.5f, 0.0f, 1000.0f) * units;
    if (!m_enabled) {
      return;
    }

    const uint32_t resolution = (std::clamp(WetnessOptions::rainMapResolution(), 64u, 2048u) + 7u) & ~7u;
    const float extent = finite(WetnessOptions::rainMapExtent(), 128.0f, 8.0f, 10000.0f) * units;
    const float texel = extent / float(resolution);
    const Vector3 camera = ctx.getSceneManager().getCamera().getPosition();
    if (!std::isfinite(camera.x) || !std::isfinite(camera.y) || !std::isfinite(camera.z)) {
      m_enabled = false;
      args.rainMapEnabled = false;
      return;
    }
    const float groundY = RtxOptions::zUp() ? camera.y : camera.z;
    const float height = RtxOptions::zUp() ? camera.z : camera.y;

    args.rainMapResolution = resolution;
    args.rainMapOriginX = std::floor(camera.x / texel) * texel - extent * 0.5f;
    args.rainMapOriginY = std::floor(groundY / texel) * texel - extent * 0.5f;
    args.rainMapExtent = extent;
    args.rainMapTexel = texel;
    args.rainMapTraceTop = height + finite(WetnessOptions::rainMapTraceAbove(), 100.0f, 1.0f, 10000.0f) * units;
    args.rainMapTraceBottom = height - finite(WetnessOptions::rainMapTraceBelow(), 60.0f, 1.0f, 10000.0f) * units;
  }

  void RtxRainMap::ensureResources(RtxContext& ctx, uint32_t resolution) {
    if (m_map.isValid() && m_resolution == resolution) {
      return;
    }
    Rc<DxvkContext> context = &ctx;
    const VkExtent3D extent = { resolution, resolution, 1 };
    m_height = Resources::createImageResource(context, "Rain map height", extent, VK_FORMAT_R32G32_SFLOAT);
    m_map = Resources::createImageResource(context, "Rain map", extent, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_resolution = resolution;
    m_traced = false;
  }

  void RtxRainMap::dispatch(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput) {
    const RaytraceArgs& constants = rtOutput.m_raytraceArgs;
    if (!m_enabled || !constants.wetnessArgs.rainMapEnabled || m_preparedFrame != constants.frameIdx
        || m_preparedFrame != m_device->getCurrentFrameId()) {
      return;
    }
    ScopedGpuProfileZone(&ctx, "Rain map");
    ensureResources(ctx, constants.wetnessArgs.rainMapResolution);
    const uint32_t groups = (m_resolution + 7u) / 8u;

    ctx.bindCommonRayTracingResources(rtOutput);
    // The common descriptor must not sample the image written by the relief dispatch.
    ctx.bindResourceView(BINDING_RAIN_MAP, placeholderView(ctx), nullptr);
    ctx.bindResourceView(RAIN_MAP_BINDING_HEIGHT_OUTPUT, m_height.view, nullptr);
    ctx.getCommandList()->trackResource<DxvkAccess::Write>(m_height.image);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, RainMapTraceShader::getShader());
    ctx.dispatch(groups, groups, 1);

    ctx.bindResourceView(RAIN_MAP_BINDING_HEIGHT_INPUT, m_height.view, nullptr);
    ctx.bindResourceView(RAIN_MAP_BINDING_MAP_OUTPUT, m_map.view, nullptr);
    ctx.getCommandList()->trackResource<DxvkAccess::Read>(m_height.image);
    ctx.getCommandList()->trackResource<DxvkAccess::Write>(m_map.image);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, RainMapReliefShader::getShader());
    ctx.dispatch(groups, groups, 1);
    m_dispatchedFrame = constants.frameIdx;
    m_traced = true;
  }

  Rc<DxvkImageView> RtxRainMap::placeholderView(RtxContext& ctx) {
    if (!m_placeholder.isValid()) {
      Rc<DxvkContext> context = &ctx;
      VkClearColorValue miss = {};
      miss.float32[1] = -std::numeric_limits<float>::max();
      m_placeholder = Resources::createImageResource(context, "Rain map unavailable", VkExtent3D { 1, 1, 1 },
        VK_FORMAT_R32G32B32A32_SFLOAT, 1, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 0,
        VK_IMAGE_USAGE_STORAGE_BIT, miss);
    }
    return m_placeholder.view;
  }

  Rc<DxvkImageView> RtxRainMap::view(RtxContext& ctx) {
    // Particle spawning precedes prepare() and reads the previous constants and their matching map.
    if (m_enabled && m_traced && m_dispatchedFrame == m_preparedFrame && m_map.isValid()) {
      ctx.getCommandList()->trackResource<DxvkAccess::Read>(m_map.image);
      return m_map.view;
    }
    return placeholderView(ctx);
  }

} // namespace dxvk
