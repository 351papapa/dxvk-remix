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
#include "d3d9_raster_replay.h"

#include <cmath>
#include <limits>
#include <optional>

#include "../../d3d9/d3d9_caps.h"
#include "../../d3d9/d3d9_rtx.h"
#include "../../d3d9/d3d9_spec_constants.h"
#include "../../d3d9/d3d9_state.h"
#include "../../d3d9/d3d9_util.h"
#include "../../dxso/dxso_util.h"
#include "../../dxvk/dxvk_buffer.h"
#include "../../dxvk/dxvk_context.h"
#include "../../dxvk/rtx_render/rtx_frontend_interface.h"
#include "../../dxvk/rtx_render/rtx_materials.h"
#include "../../dxvk/rtx_render/rtx_replacement_material_texture_type.h"
#include "../../dxvk/rtx_render/rtx_scene_types.h"
#include "d3d9_draw_call.h"
#include "../../util/log/log.h"
#include "../../util/util_once.h"
#include "../../util/util_string.h"

namespace dxvk {
  namespace frontends {
    namespace d3d9 {

      namespace {
        // The sign of a projection's [2][3] term, as RtCamera's jitter reads it: -1 below zero, else 1.
        float projectionSign(const Matrix4& viewToProjection) {
          return viewToProjection[2][3] < 0.f ? -1.f : 1.f;
        }

        void issue(DxvkContext& ctx, const DrawParameters& params) {
          if (params.indexCount == 0) {
            ctx.DxvkContext::draw(params.vertexCount, params.instanceCount, params.vertexOffset, 0);
          } else {
            ctx.DxvkContext::drawIndexed(params.indexCount, params.instanceCount, params.firstIndex, params.vertexOffset, 0);
          }
        }

        // A D3D9 constant buffer that the replays override. The D3D9 device binds each one at its slot once, when it
        // creates the buffer, so the buffer in the slot is the device's.
        template<typename T>
        class ConstantBuffer {
        public:
          ConstantBuffer(DxvkContext& ctx, DxsoProgramType stage, DxsoConstantBuffers type)
            : m_ctx(ctx)
            , m_buffer(ctx.getShaderResourceSlot(computeResourceSlotId(stage, DxsoBindingType::ConstantBuffer, type)).bufferSlice.buffer()) { }

          // The contents as the last draw set them.
          const T& current() const {
            return *static_cast<const T*>(m_buffer->mapPtr(0));
          }

          // New contents, which the next draw reads.
          T& map() {
            const DxvkBufferSliceHandle slice = m_buffer->allocSlice();
            m_ctx.invalidateBuffer(m_buffer, slice);
            return *static_cast<T*>(slice.mapPtr);
          }

        private:
          DxvkContext& m_ctx;
          Rc<DxvkBuffer> m_buffer;
        };

        ConstantBuffer<D3D9RtxVertexCaptureData> vertexCaptureConstants(DxvkContext& ctx) {
          return ConstantBuffer<D3D9RtxVertexCaptureData>(ctx, DxsoProgramType::VertexShader, DxsoConstantBuffers::VSVertexCaptureData);
        }

        ConstantBuffer<D3D9FixedFunctionVS> fixedFunctionConstants(DxvkContext& ctx) {
          return ConstantBuffer<D3D9FixedFunctionVS>(ctx, DxsoProgramType::VertexShader, DxsoConstantBuffers::VSFixedFunction);
        }

        ConstantBuffer<D3D9SharedPS> sharedPixelConstants(DxvkContext& ctx) {
          return ConstantBuffer<D3D9SharedPS>(ctx, DxsoProgramType::PixelShader, DxsoConstantBuffers::PSShared);
        }
      }

      void replaySky(RtxContext& rtx, const DrawParameters& params, const DrawCallState& state) {
        frontend::RasterService service { rtx };
        if (!service.rendersGameSky()) {
          return;
        }
        DxvkContext& ctx = service.context();
        ConstantBuffer<D3D9RtxVertexCaptureData> vsConstants = vertexCaptureConstants(ctx);
        ConstantBuffer<D3D9FixedFunctionVS> ffConstants = fixedFunctionConstants(ctx);
        // A shader draw takes the passes' matrices through the vertex capture constants, a fixed-function draw through
        // its transform constants.
        const bool vs = state.usesVertexShader;
        std::optional<D3D9RtxVertexCaptureData> prevVs;
        std::optional<D3D9FixedFunctionVS> prevFf;
        if (vs) {
          prevVs.emplace(vsConstants.current());
        } else {
          prevFf.emplace(ffConstants.current());
        }
        const DxvkScInfo& specConstants = ctx.getSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS);
        const uint32_t prevJitter = specConstants.specConstants[D3D9SpecConstantId::ClipSpaceJitterEnabled] ? 1 : 0;
        const uint32_t prevCustomTransform = specConstants.specConstants[D3D9SpecConstantId::CustomVertexTransformEnabled] ? 1 : 0;
        const uint32_t colorSlot = state.getMaterialData().getColorTextureSlot(0);
        Rc<DxvkImageView> prevColorView;

        frontend::RasterDrawInfo info;
        info.materialKey = state.getMaterialData().getHash();
        info.worldToView = vs ? state.getTransformData().worldToView : prevFf->View;
        info.viewToProjection = vs ? state.getTransformData().viewToProjection : prevFf->Projection;
        info.minDepth = state.minZ;
        info.maxDepth = state.maxZ;

        service.replay(frontend::RasterTarget::Sky, info, [&](const frontend::RasterPass& pass) {
          const bool matte = pass.kind == frontend::RasterPassKind::SkyMatte;
          // A replacement texture is bound for the whole replay, in place of a bound colour texture.
          if (matte && pass.texture != nullptr && colorSlot < DxvkLimits::MaxNumResourceSlots &&
              ctx.getShaderResourceSlot(colorSlot).imageView != nullptr) {
            prevColorView = ctx.getShaderResourceSlot(colorSlot).imageView;
            ctx.bindResourceView(colorSlot, pass.texture, nullptr);
          }
          if (vs) {
            ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ClipSpaceJitterEnabled, matte ? 1 : prevJitter);
            ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::CustomVertexTransformEnabled,
                                matte ? prevCustomTransform : 1);
            D3D9RtxVertexCaptureData constants = *prevVs;
            if (matte) {
              const float sign = projectionSign(state.getTransformData().viewToProjection);
              constants.jitterX = pass.clipJitter.x * sign;
              constants.jitterY = pass.clipJitter.y * sign;
            } else {
              constants.customWorldToProjection = pass.viewToProjection * pass.worldToView;
            }
            vsConstants.map() = constants;
          } else {
            D3D9FixedFunctionVS constants = *prevFf;
            if (matte) {
              const float sign = projectionSign(constants.Projection);
              const Vector2 jitter { pass.clipJitter.x * sign, pass.clipJitter.y * sign };
              if (std::abs(jitter.x) >= std::numeric_limits<float>::min() || std::abs(jitter.y) >= std::numeric_limits<float>::min()) {
                constants.Projection[2][0] += jitter.x;
                constants.Projection[2][1] += jitter.y;
              }
            } else {
              constants.View = pass.worldToView;
              constants.WorldView = pass.worldToView * prevFf->World;
              constants.Projection = pass.viewToProjection;
            }
            ffConstants.map() = constants;
          }
          issue(ctx, params);
        });

        if (vs) {
          ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ClipSpaceJitterEnabled, prevJitter);
          ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::CustomVertexTransformEnabled, prevCustomTransform);
          vsConstants.map() = *prevVs;
        } else {
          ffConstants.map() = *prevFf;
        }
        if (prevColorView != nullptr) {
          ctx.bindResourceView(colorSlot, prevColorView, nullptr);
        }
      }

      TerrainReplay replayTerrain(RtxContext& rtx, const DrawParameters& params, DrawCallState& state) {
        TerrainReplay replay;
        if (!frontend::terrainBakeActive() || !state.testCategoryFlags(InstanceCategories::Terrain)) {
          return replay;
        }
        if (state.usesVertexShader && !D3D9Rtx::useVertexCapture()) {
          ONCE(Logger::warn(str::format("[RTX Terrain Baker] Terrain texture corresponds to a draw call with programmable Vertex Shader usage. Vertex capture must be enabled to support baking of such draw calls. Ignoring the draw call.")));
          return replay;
        }
        frontend::RasterService service { rtx };
        DxvkContext& ctx = service.context();
        ConstantBuffer<D3D9RtxVertexCaptureData> vsConstants = vertexCaptureConstants(ctx);
        ConstantBuffer<D3D9FixedFunctionVS> ffConstants = fixedFunctionConstants(ctx);
        ConstantBuffer<D3D9SharedPS> psConstants = sharedPixelConstants(ctx);
        const bool vs = state.usesVertexShader;
        const LegacyMaterialData& legacy = state.getMaterialData();
        const uint32_t colorSlot = legacy.getColorTextureSlot(0);
        // A material channel other than albedo reads the albedo replacement through a second texture stage, which the
        // draw must leave unused.
        uint32_t secondarySlot = kInvalidResourceSlot;
        if (legacy.getColorTextureSlot(kTerrainBakerSecondaryTextureStage) == kInvalidResourceSlot) {
          const auto shaderSampler = RemapStateSamplerShader(static_cast<uint8_t>(kTerrainBakerSecondaryTextureStage));
          secondarySlot = computeResourceSlotId(DxsoProgramType::PixelShader, DxsoBindingType::Image, shaderSampler.second);
        }
        Rc<DxvkImageView> prevColorView;
        if (colorSlot < DxvkLimits::MaxNumResourceSlots && ctx.getShaderResourceSlot(colorSlot).imageView != nullptr) {
          prevColorView = ctx.getShaderResourceSlot(colorSlot).imageView;
        }

        frontend::RasterDrawInfo info;
        info.materialKey = legacy.getHash();
        info.objectToWorld = state.getTransformData().objectToWorld;
        info.geometry = &state.getGeometryData();
        info.texcoords = state.hasTextureCoordinates();
        info.secondTexture = legacy.getColorTexture2().isValid();
        info.programmablePixelShader = state.usesPixelShader;
        // Shader model 1 pixel shaders and fixed-function texturing read the replacement texture category.
        info.channelDecode = !state.usesPixelShader || state.programmablePixelShaderInfo.majorVersion() <= 1;
        info.secondTextureSlot = secondarySlot != kInvalidResourceSlot;

        std::optional<D3D9RtxVertexCaptureData> prevVs;
        std::optional<D3D9FixedFunctionVS> prevFf;
        std::optional<D3D9SharedPS> prevShared;
        DxvkScInfo prevSpecConstants {};
        Rc<DxvkSampler> prevSecondarySampler;
        bool replaced = false;
        const frontend::RasterResult result = service.replay(frontend::RasterTarget::TerrainCascades, info,
                                                             [&](const frontend::RasterPass& pass) {
          if (!prevShared.has_value()) {
            if (vs) {
              prevVs.emplace(vsConstants.current());
            } else {
              prevFf.emplace(ffConstants.current());
            }
            prevShared.emplace(psConstants.current());
            prevSpecConstants = ctx.getSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS);
            // The bake writes unfogged texture colour.
            ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::FogEnabled, false);
            ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::VertexFogMode, D3DFOG_NONE);
            ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::PixelFogMode, D3DFOG_NONE);
            if (vs) {
              ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::CustomVertexTransformEnabled, true);
            }
            ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ReplacementTextureCategory,
                                static_cast<uint32_t>(ReplacementMaterialTextureCategory::AlbedoOpacity));
          }

          // The first pass of a channel binds its textures, which its other cascades keep.
          if (pass.index == 0) {
            if (pass.texture != nullptr) {
              replaced = true;
              ctx.bindResourceView(colorSlot, pass.texture, nullptr);
            }
            ctx.setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ReplacementTextureCategory,
                                static_cast<uint32_t>(pass.channel));
            if (pass.channel != frontend::RasterChannel::Color) {
              ctx.bindResourceView(secondarySlot, pass.secondTexture, nullptr);
              prevSecondarySampler = ctx.getShaderResourceSlot(secondarySlot).sampler;
              ctx.bindResourceSampler(secondarySlot, ctx.getShaderResourceSlot(colorSlot).sampler);
            }
          }

          // The second texture stage's scale and offsets turn a height sample into the baked height.
          D3D9SharedPS& sharedState = psConstants.map();
          for (int i = 0; i < caps::TextureStageCount; ++i) {
            sharedState.Stages[i] = prevShared->Stages[i];
          }
          sharedState.Stages[kTerrainBakerSecondaryTextureStage].texturePreOffset = pass.preOffset;
          sharedState.Stages[kTerrainBakerSecondaryTextureStage].textureScale = pass.scale;
          sharedState.Stages[kTerrainBakerSecondaryTextureStage].texturePostOffset = pass.postOffset;

          if (vs) {
            D3D9RtxVertexCaptureData& constants = vsConstants.map();
            constants = *prevVs;
            constants.customWorldToProjection = pass.viewToProjection * (pass.worldToView * prevVs->normalTransform);
          } else {
            D3D9FixedFunctionVS& constants = ffConstants.map();
            constants = *prevFf;
            constants.InverseView = pass.viewToWorld;
            constants.View = pass.worldToView;
            constants.WorldView = pass.worldToView * prevFf->World;
            constants.Projection = pass.viewToProjection;
            // The bake writes unlit texture colour.
            for (auto& light : constants.Lights) {
              light.Diffuse = Vector4(0.f);
              light.Specular = Vector4(0.f);
              light.Ambient = Vector4(1.f);
            }
          }
          issue(ctx, params);
        });

        if (prevShared.has_value()) {
          ctx.setSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS, prevSpecConstants);
          psConstants.map() = *prevShared;
          if (vs) {
            vsConstants.map() = *prevVs;
          } else {
            ffConstants.map() = *prevFf;
          }
          if (replaced && secondarySlot != kInvalidResourceSlot) {
            // The second texture stage was unused before the bake, so it gets a null view again.
            ctx.bindResourceView(secondarySlot, nullptr, nullptr);
            if (prevSecondarySampler.ptr()) {
              ctx.bindResourceSampler(secondarySlot, prevSecondarySampler);
            }
          }
        }

        if (result.textureTransform.has_value()) {
          state.modifyTransformData().textureTransform = *result.textureTransform;
        }
        if (result.material != nullptr) {
          state.modifyTransformData().texgenMode = TexGenMode::CascadedViewPositions;
          state.modifyMaterialData() = LegacyMaterialData::fromBakedAlbedo(result.material->getOpaqueMaterialData().getAlbedoOpacityTexture(),
                                                                           result.sampler);
        }
        // The draw's colour texture goes back into the colour slot of the material the draw has after the bake.
        if (replaced && result.passes > 0 && prevColorView != nullptr) {
          ctx.bindResourceView(state.getMaterialData().getColorTextureSlot(0), prevColorView, nullptr);
        }
        replay.material = result.material;
        replay.cascadesChanged = result.cascadesChanged;
        replay.channelPasses = result.channelPasses;
        return replay;
      }

    } // namespace d3d9
  } // namespace frontends
} // namespace dxvk
