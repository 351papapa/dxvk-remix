/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <unordered_map>

#include "../../util/log/log.h"
#include "../../util/util_string.h"
#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "rtx_retained_batch_policy.h"
#include "rtx_retained_world_batch.h"
#include "rtx_retained_world_batch_cpu.h"

namespace dxvk {
namespace {
  using Cpu = RetainedWorldBatchCpu;
  using Input = RetainedWorldBatcher::ParentInput;
  using Member = RetainedWorldBatcher::Member;
  using Plan = RetainedWorldBatcher::Plan;
  constexpr size_t kMaxBytes = 128 * 1024 * 1024;
  constexpr size_t kMaxBatches = 256;
  constexpr size_t kMaxParts = 65536;
  enum class BuildResult { Success, Input, Plan, Pack, Allocation, Map, Exception };

  struct FullPrepareTrace {
    using Clock = std::chrono::steady_clock;

    FullPrepareTrace(uint32_t frame, uint64_t attempt) noexcept
      : m_frame(frame), m_attempt(attempt) {
      m_boundaries[0] = Clock::now();
    }

    void mark(size_t boundary) noexcept {
      m_boundaries[boundary] = Clock::now();
    }

    void complete() noexcept {
      m_success = true;
    }

    ~FullPrepareTrace() noexcept {
      if (!m_success) {
        return;
      }
      // The trace outlives try-local work, including the swapped-out old cache.
      const auto end = Clock::now();
      const auto milliseconds = [](auto first, auto last) {
        return std::chrono::duration<double, std::milli>(last - first).count();
      };
      try {
        Logger::info(str::format("[WorldBatchWork] frame=", m_frame, " prepareAttempt=", m_attempt,
          " success=helperPrepare fullFixedProofMs=", milliseconds(m_boundaries[0], m_boundaries[1]),
          " fullCandidatesAndGroupingMs=", milliseconds(m_boundaries[1], m_boundaries[2]),
          " fullCoverageBudgetMs=", milliseconds(m_boundaries[2], m_boundaries[3]),
          " fullFixedStageMs=", milliseconds(m_boundaries[3], m_boundaries[4]),
          " fullNewPackMs=", milliseconds(m_boundaries[4], m_boundaries[5]),
          " fullMembershipIndexAndCleanupMs=", milliseconds(m_boundaries[5], end),
          " totalMs=", milliseconds(m_boundaries[0], end)));
      } catch (...) {
      }
    }

    std::array<Clock::time_point, 6> m_boundaries {};
    uint32_t m_frame = 0;
    uint64_t m_attempt = 0;
    bool m_success = false;
  };

  template<typename T>
  bool sameRepresentation(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
  }

  std::array<float, 16> matrixArray(const Matrix4& value) {
    std::array<float, 16> result;
    for (size_t column = 0; column < 4; ++column) {
      for (size_t row = 0; row < 4; ++row) {
        result[column * 4 + row] = value[column][row];
      }
    }
    return result;
  }

  AlphaStateDesc alphaState(const ExternalDrawState& state) {
    return state.alphaState.has_value() ? *state.alphaState :
      deriveAlphaState(FixedFunctionAlphaInput::fromLegacyMaterial(state.drawCall.getMaterialData()),
                       *state.resolvedMaterial, state.drawCall.getCategoryFlags());
  }

  LegacySurfaceDesc legacySurface(const ExternalDrawState& state) {
    return state.legacySurface.has_value() ? *state.legacySurface :
      deriveLegacySurface(state.drawCall.getMaterialData(), state.drawCall.getTransformData());
  }

  bool eligible(const ExternalDrawState& state) {
    if (!state.frontendResolved || state.assetKeys || !state.inlineSubmeshes || state.inlineSubmeshes->size() != 1 ||
        !state.resolvedMaterial || state.inlineSubmeshParts || state.deforming || state.optionalParticleDesc ||
        state.materialOverride || !state.gpuInstancingTransforms.empty() || state.sortOrder ||
        state.cameraType != CameraType::Main || state.drawCall.cameraType != CameraType::Main ||
        state.drawCall.usesVertexShader || state.drawCall.newGeometryEveryFrame ||
        state.drawCall.isUsingRaytracedRenderTarget || state.drawCall.isEye() || state.drawCall.materialIndependentGeometry ||
        state.drawCall.getSkinningState().numBones || state.drawCall.getSkinningState().numBonesPerVertex ||
        !state.drawCall.getSkinningState().pBoneMatrices.empty()) {
      return false;
    }
    CategoryFlags allowed;
    allowed.set(InstanceCategories::IgnoreLights);
    allowed.set(InstanceCategories::IgnoreAntiCulling);
    allowed.set(InstanceCategories::IgnoreMotionBlur);
    allowed.set(InstanceCategories::IgnoreOpacityMicromap);
    allowed.set(InstanceCategories::IgnoreAlphaChannel);
    allowed.set(InstanceCategories::IgnoreBakedLighting);
    if ((state.categories.raw() & ~allowed.raw()) || (state.drawCall.getCategoryFlags().raw() & ~allowed.raw())) {
      return false;
    }
    const MaterialData& material = *state.resolvedMaterial;
    if (material.getType() != MaterialDataType::Opaque || material.getIgnored() || material.getParticleSystemDesc()) {
      return false;
    }
    const auto& opaque = material.getOpaqueMaterialData();
    if (!retained_batch_policy::hasNoEffectiveAuthoredEmission(opaque.getEnableEmission(), opaque.getEmissiveIntensity()) ||
        opaque.getSpriteSheetFPS() || opaque.getSpriteSheetRows() > 1 ||
        opaque.getSpriteSheetCols() > 1 || opaque.getHeightTexture().isValid()) {
      return false;
    }
    const auto alpha = alphaState(state);
    const auto legacy = legacySurface(state);
    const auto& transforms = state.drawCall.getTransformData();
    const auto& geometry = state.inlineSubmeshes->front();
    return alpha.isBlendingDisabled && alpha.isFullyOpaque && alpha.alphaTestType == AlphaTestType::kAlways &&
      !alpha.isDecal && !alpha.isParticle && !alpha.emissiveBlend &&
      !transforms.instancesToObject && !transforms.instancesFromHost && !transforms.enableClipPlane &&
      transforms.texgenMode == TexGenMode::None && !legacy.enableClipPlane && legacy.texgenMode == TexGenMode::None &&
      !geometry.numBonesPerVertex && !geometry.skinningRetarget && !geometry.blendWeightBuffer.defined() &&
      !geometry.blendIndicesBuffer.defined() && geometry.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST &&
      Cpu::validTransform(matrixArray(transforms.objectToWorld));
  }

  bool sameTexture(const TextureRef& a, const TextureRef& b) {
    return a.isValid() == b.isValid() && a.getHostEncoding() == b.getHostEncoding() &&
      (!a.isValid() || (a.getUniqueKey() == b.getUniqueKey() && a.getImageView() == b.getImageView() &&
                        a.getManagedTexture().ptr() == b.getManagedTexture().ptr()));
  }

  bool sameMaterial(const MaterialData& a, const MaterialData& b) {
    if (a.usesLegacyDefaults() != b.usesLegacyDefaults()) {
      return false;
    }
    const auto& ma = a.getOpaqueMaterialData();
    const auto& mb = b.getOpaqueMaterialData();
    if (ma.getSamplerOverride().ptr() != mb.getSamplerOverride().ptr() ||
        ma.getIgnoreAlphaChannel() != mb.getIgnoreAlphaChannel()) {
      return false;
    }
    // Compare authored values and resources without cache metadata or dirty flags.
    return
      sameRepresentation(ma.getAnisotropyConstant(), mb.getAnisotropyConstant()) &&
      sameRepresentation(ma.getEmissiveIntensity(), mb.getEmissiveIntensity()) &&
      sameRepresentation(ma.getAlbedoConstant(), mb.getAlbedoConstant()) &&
      sameRepresentation(ma.getOpacityConstant(), mb.getOpacityConstant()) &&
      sameRepresentation(ma.getRoughnessConstant(), mb.getRoughnessConstant()) &&
      sameRepresentation(ma.getMetallicConstant(), mb.getMetallicConstant()) &&
      sameRepresentation(ma.getEmissiveColorConstant(), mb.getEmissiveColorConstant()) &&
      sameRepresentation(ma.getEnableEmission(), mb.getEnableEmission()) &&
      sameRepresentation(ma.getSkyLitParticle(), mb.getSkyLitParticle()) &&
      sameRepresentation(ma.getSpriteSheetRows(), mb.getSpriteSheetRows()) &&
      sameRepresentation(ma.getSpriteSheetCols(), mb.getSpriteSheetCols()) &&
      sameRepresentation(ma.getSpriteSheetFPS(), mb.getSpriteSheetFPS()) &&
      sameRepresentation(ma.getEnableThinFilm(), mb.getEnableThinFilm()) &&
      sameRepresentation(ma.getAlphaIsThinFilmThickness(), mb.getAlphaIsThinFilmThickness()) &&
      sameRepresentation(ma.getThinFilmThicknessConstant(), mb.getThinFilmThicknessConstant()) &&
      sameRepresentation(ma.getUseLegacyAlphaState(), mb.getUseLegacyAlphaState()) &&
      sameRepresentation(ma.getBlendEnabled(), mb.getBlendEnabled()) &&
      sameRepresentation(ma.getBlendType(), mb.getBlendType()) &&
      sameRepresentation(ma.getInvertedBlend(), mb.getInvertedBlend()) &&
      sameRepresentation(ma.getAlphaTestType(), mb.getAlphaTestType()) &&
      sameRepresentation(ma.getAlphaTestReferenceValue(), mb.getAlphaTestReferenceValue()) &&
      sameRepresentation(ma.getDisplaceIn(), mb.getDisplaceIn()) &&
      sameRepresentation(ma.getDisplaceOut(), mb.getDisplaceOut()) &&
      sameRepresentation(ma.getSubsurfaceTransmittanceColor(), mb.getSubsurfaceTransmittanceColor()) &&
      sameRepresentation(ma.getSubsurfaceMeasurementDistance(), mb.getSubsurfaceMeasurementDistance()) &&
      sameRepresentation(ma.getSubsurfaceSingleScatteringAlbedo(), mb.getSubsurfaceSingleScatteringAlbedo()) &&
      sameRepresentation(ma.getSubsurfaceVolumetricAnisotropy(), mb.getSubsurfaceVolumetricAnisotropy()) &&
      sameRepresentation(ma.getSubsurfaceDiffusionProfile(), mb.getSubsurfaceDiffusionProfile()) &&
      sameRepresentation(ma.getSubsurfaceRadius(), mb.getSubsurfaceRadius()) &&
      sameRepresentation(ma.getSubsurfaceRadiusScale(), mb.getSubsurfaceRadiusScale()) &&
      sameRepresentation(ma.getSubsurfaceMaxSampleRadius(), mb.getSubsurfaceMaxSampleRadius()) &&
      sameRepresentation(ma.getFilterMode(), mb.getFilterMode()) &&
      sameRepresentation(ma.getWrapModeU(), mb.getWrapModeU()) &&
      sameRepresentation(ma.getWrapModeV(), mb.getWrapModeV()) &&
      sameRepresentation(ma.getEnableDLSSControlMask(), mb.getEnableDLSSControlMask()) &&
      sameRepresentation(ma.getDLSSControlMaskIntensity(), mb.getDLSSControlMaskIntensity()) &&
      sameRepresentation(ma.getDLSSControlMaskToneStrength(), mb.getDLSSControlMaskToneStrength()) &&
      sameRepresentation(ma.getDLSSControlMaskStructuralStrength(), mb.getDLSSControlMaskStructuralStrength()) &&
      sameRepresentation(ma.getRoughnessEncoding(), mb.getRoughnessEncoding()) &&
      sameRepresentation(ma.getSpecularMetalStrength(), mb.getSpecularMetalStrength()) &&
      sameRepresentation(ma.getSpecularMetalParams(), mb.getSpecularMetalParams()) &&
      sameRepresentation(ma.getSpecularMetalTint(), mb.getSpecularMetalTint()) &&
      sameRepresentation(ma.getAlbedoTint(), mb.getAlbedoTint()) &&
      sameTexture(ma.getAlbedoOpacityTexture(), mb.getAlbedoOpacityTexture()) &&
      sameTexture(ma.getNormalTexture(), mb.getNormalTexture()) &&
      sameTexture(ma.getTangentTexture(), mb.getTangentTexture()) &&
      sameTexture(ma.getHeightTexture(), mb.getHeightTexture()) &&
      sameTexture(ma.getRoughnessTexture(), mb.getRoughnessTexture()) &&
      sameTexture(ma.getMetallicTexture(), mb.getMetallicTexture()) &&
      sameTexture(ma.getEmissiveColorTexture(), mb.getEmissiveColorTexture()) &&
      sameTexture(ma.getSubsurfaceTransmittanceTexture(), mb.getSubsurfaceTransmittanceTexture()) &&
      sameTexture(ma.getSubsurfaceThicknessTexture(), mb.getSubsurfaceThicknessTexture()) &&
      sameTexture(ma.getSubsurfaceSingleScatteringAlbedoTexture(), mb.getSubsurfaceSingleScatteringAlbedoTexture()) &&
      sameTexture(ma.getSubsurfaceRadiusTexture(), mb.getSubsurfaceRadiusTexture()) &&
      sameTexture(ma.getSecondaryTexture(), mb.getSecondaryTexture());
  }

  bool sameBlendMode(const DxvkBlendMode& a, const DxvkBlendMode& b) {
    return a.enableBlending == b.enableBlending && a.colorSrcFactor == b.colorSrcFactor &&
      a.colorDstFactor == b.colorDstFactor && a.colorBlendOp == b.colorBlendOp &&
      a.alphaSrcFactor == b.alphaSrcFactor && a.alphaDstFactor == b.alphaDstFactor &&
      a.alphaBlendOp == b.alphaBlendOp && a.writeMask == b.writeMask;
  }

  bool sameState(const ExternalDrawState& a, const ExternalDrawState& b) {
    const auto& ga = a.inlineSubmeshes->front();
    const auto& gb = b.inlineSubmeshes->front();
    const auto& ma = *a.resolvedMaterial;
    const auto& mb = *b.resolvedMaterial;
    const auto& la = a.drawCall.getMaterialData();
    const auto& lb = b.drawCall.getMaterialData();
    const auto aa = alphaState(a);
    const auto ab = alphaState(b);
    const auto sa = legacySurface(a);
    const auto sb = legacySurface(b);
    return a.cameraType == b.cameraType && a.doubleSided == b.doubleSided &&
      a.categories.raw() == b.categories.raw() && a.drawCall.getCategoryFlags().raw() == b.drawCall.getCategoryFlags().raw() &&
      a.drawCall.cameraType == b.drawCall.cameraType && a.drawCall.wetnessTag == b.drawCall.wetnessTag &&
      la.getColorTexture().getImageHash() == lb.getColorTexture().getImageHash() &&
      sameMaterial(ma, mb) && sameBlendMode(la.blendMode, lb.blendMode) &&
      (ma.getSamplerOverride().ptr() || la.getSampler().ptr() == lb.getSampler().ptr()) &&
      !std::strcmp(firstAlphaStateDifference(aa, ab), "none") &&
      !std::strcmp(firstLegacySurfaceDifference(sa, sb), "none") &&
      ga.cullMode == gb.cullMode && ga.frontFace == gb.frontFace && ga.forceCullBit == gb.forceCullBit &&
      a.drawCall.getCullMode() == b.drawCall.getCullMode() &&
      ga.normalBuffer.defined() == gb.normalBuffer.defined() &&
      ga.texcoordBuffer.defined() == gb.texcoordBuffer.defined() && ga.color0Buffer.defined() == gb.color0Buffer.defined();
  }

  bool sameBuffer(const RasterBuffer& a, const RasterBuffer& b, bool index = false) {
    return a.defined() == b.defined() && (!a.defined() ||
      (a.buffer().ptr() == b.buffer().ptr() && a.offset() == b.offset() && a.length() == b.length() &&
       a.offsetFromSlice() == b.offsetFromSlice() && a.stride() == b.stride() &&
       (index ? a.indexType() == b.indexType() : a.vertexFormat() == b.vertexFormat())));
  }

  bool sameGeometry(const RasterGeometry& a, const RasterGeometry& b) {
    return a.vertexCount == b.vertexCount && a.indexCount == b.indexCount && a.topology == b.topology &&
      a.getHashForRule<rules::FullGeometryHash>() == b.getHashForRule<rules::FullGeometryHash>() &&
      sameBuffer(a.positionBuffer, b.positionBuffer) && sameBuffer(a.normalBuffer, b.normalBuffer) &&
      sameBuffer(a.texcoordBuffer, b.texcoordBuffer) && sameBuffer(a.color0Buffer, b.color0Buffer) &&
      sameBuffer(a.indexBuffer, b.indexBuffer, true);
  }

  uint32_t formatBytes(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R32G32B32_SFLOAT: return 12;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_B8G8R8A8_UNORM: return 4;
    default: return 0;
    }
  }

  bool readable(const RasterBuffer& buffer, uint32_t count, uint32_t bytes) {
    if (!buffer.defined() || !count || !bytes || buffer.stride() < bytes || buffer.isPendingGpuWrite() ||
        !(buffer.buffer()->memFlags() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      return false;
    }
    const uint64_t end = uint64_t(buffer.offsetFromSlice()) + uint64_t(count - 1) * buffer.stride() + bytes;
    return end <= buffer.length() && buffer.mapPtr() != nullptr;
  }

  bool readableGeometry(const RasterGeometry& geometry) {
    const auto& p = geometry.positionBuffer;
    const auto& n = geometry.normalBuffer;
    const auto& t = geometry.texcoordBuffer;
    const auto& c = geometry.color0Buffer;
    const auto& i = geometry.indexBuffer;
    if (!geometry.vertexCount || !geometry.indexCount || geometry.indexCount % 3 ||
        (p.vertexFormat() != VK_FORMAT_R32G32B32_SFLOAT && p.vertexFormat() != VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !readable(p, geometry.vertexCount, formatBytes(p.vertexFormat()))) {
      return false;
    }
    if (n.defined() && ((n.vertexFormat() != VK_FORMAT_R32G32B32_SFLOAT &&
         n.vertexFormat() != VK_FORMAT_R32G32B32A32_SFLOAT && n.vertexFormat() != VK_FORMAT_R32_UINT) ||
        !readable(n, geometry.vertexCount, formatBytes(n.vertexFormat())))) {
      return false;
    }
    if (t.defined() && ((t.vertexFormat() != VK_FORMAT_R32G32_SFLOAT &&
         t.vertexFormat() != VK_FORMAT_R32G32B32_SFLOAT && t.vertexFormat() != VK_FORMAT_R32G32B32A32_SFLOAT) ||
        !readable(t, geometry.vertexCount, formatBytes(t.vertexFormat())))) {
      return false;
    }
    if (c.defined() && (c.vertexFormat() != VK_FORMAT_B8G8R8A8_UNORM || !readable(c, geometry.vertexCount, 4))) {
      return false;
    }
    const uint32_t indexBytes = i.defined() && i.indexType() == VK_INDEX_TYPE_UINT32 ? 4 : 2;
    return i.defined() && (i.indexType() == VK_INDEX_TYPE_UINT32 || i.indexType() == VK_INDEX_TYPE_UINT16) &&
      i.stride() == indexBytes && i.offsetFromSlice() % indexBytes == 0 && readable(i, geometry.indexCount, indexBytes);
  }

  const uint8_t* element(const RasterBuffer& buffer, uint32_t index) {
    return static_cast<const uint8_t*>(buffer.mapPtr()) + buffer.offsetFromSlice() + size_t(index) * buffer.stride();
  }

  bool readVertex(const RasterGeometry& geometry, uint32_t index, Cpu::Vertex& vertex) {
    std::memcpy(vertex.position, element(geometry.positionBuffer, index), sizeof(vertex.position));
    if (geometry.normalBuffer.defined()) {
      if (geometry.normalBuffer.vertexFormat() == VK_FORMAT_R32_UINT) {
        uint32_t packed;
        std::memcpy(&packed, element(geometry.normalBuffer, index), sizeof(packed));
        Cpu::decodeOctNormal(packed, vertex.normal);
      } else {
        std::memcpy(vertex.normal, element(geometry.normalBuffer, index), sizeof(vertex.normal));
      }
    }
    if (geometry.texcoordBuffer.defined()) {
      std::memcpy(vertex.texcoord, element(geometry.texcoordBuffer, index), sizeof(vertex.texcoord));
    }
    if (geometry.color0Buffer.defined()) {
      std::memcpy(&vertex.color, element(geometry.color0Buffer, index), sizeof(vertex.color));
    }
    return true;
  }

  bool readIndex(const RasterGeometry& geometry, uint32_t index, uint32_t& output) {
    if (geometry.indexBuffer.indexType() == VK_INDEX_TYPE_UINT32) {
      std::memcpy(&output, element(geometry.indexBuffer, index), sizeof(output));
    } else {
      uint16_t value;
      std::memcpy(&value, element(geometry.indexBuffer, index), sizeof(value));
      output = value;
    }
    return true;
  }

  Rc<DxvkBuffer> createBuffer(const Rc<DxvkContext>& ctx, size_t bytes) {
    DxvkBufferCreateInfo info {};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    info.access = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    return ctx->getDevice()->createBuffer(info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         DxvkMemoryStats::Category::RTXBuffer, "Retained world batch");
  }

  struct Cell {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
    float size = 0;
    Vector3 origin;
    bool operator==(const Cell& other) const {
      return x == other.x && y == other.y && z == other.z && size == other.size && origin == other.origin;
    }
  };

  bool findCell(const ExternalDrawState& state, float size, Cell& output) {
    Cell cell;
    cell.size = size;
    const auto& translation = state.drawCall.getTransformData().objectToWorld[3];
    int64_t* indices[] = {&cell.x, &cell.y, &cell.z};
    for (size_t i = 0; i < 3; ++i) {
      const double index = std::floor(double(translation[i]) / size);
      if (!std::isfinite(index) || std::abs(index) > 9e15) {
        return false;
      }
      *indices[i] = int64_t(index);
      cell.origin[i] = float(index * size);
      if (!std::isfinite(cell.origin[i])) {
        return false;
      }
    }
    output = cell;
    return true;
  }

  struct Candidate {
    Cpu::PartRef ref;
    const Input* input = nullptr;
    const ExternalDrawState* state = nullptr;
    Cell cell;
  };
  struct Bucket {
    std::vector<Candidate> sources;
    uint64_t vertices = 0;
    uint64_t indices = 0;
  };

  uint64_t bucketHash(const Candidate& candidate) {
    const auto& state = *candidate.state;
    const auto& geometry = state.inlineSubmeshes->front();
    const uint64_t fields[] = {
      uint64_t(candidate.cell.x), uint64_t(candidate.cell.y), uint64_t(candidate.cell.z),
      candidate.input->passDomain, candidate.input->wetnessRule, state.resolvedMaterial->getHash(),
      state.drawCall.getMaterialData().getColorTexture().getImageHash(),
      state.categories.raw(), state.drawCall.getCategoryFlags().raw(), uint64_t(state.drawCall.getCullMode()),
      uint64_t(geometry.frontFace), uint64_t(geometry.forceCullBit), uint64_t(geometry.normalBuffer.defined()),
      uint64_t(geometry.texcoordBuffer.defined()), uint64_t(geometry.color0Buffer.defined())
    };
    return XXH3_64bits(fields, sizeof(fields));
  }

  bool compatible(const Candidate& a, const Candidate& b) {
    return a.cell == b.cell && a.input->passDomain == b.input->passDomain &&
      a.input->wetnessRule == b.input->wetnessRule && sameState(*a.state, *b.state);
  }

  struct SavedSource {
    uint64_t id = 0;
    uint32_t part = 0;
    uint64_t passDomain = 0;
    uint64_t wetnessRule = 0;
    ExternalDrawState state;
  };
  struct BatchStorage {
    Cell cell;
    std::vector<SavedSource> sources;
    std::shared_ptr<const std::vector<uint32_t>> allIndices;
    std::vector<Cpu::Range> ranges;
    std::vector<Member> physicalMembers;
    RasterGeometry geometry;
    size_t bytes = 0;
  };
  struct CachedBatch {
    // Payloads own their source states and resources. Speculative views share them without modifying them.
    std::shared_ptr<const BatchStorage> storage;
    // Cached plans carry rendering state only. Full output constructs active ownership separately.
    std::shared_ptr<Plan> plan;
    std::vector<bool> visible;
    std::vector<bool> lastNonemptyMask;
  };

  bool sameSource(const SavedSource& source, const Input& input) {
    if (source.id != input.id || source.part >= input.parts.size() ||
        source.passDomain != input.passDomain || source.wetnessRule != input.wetnessRule) {
      return false;
    }
    const auto& state = input.parts[source.part];
    return eligible(state) && sameState(source.state, state) &&
      source.state.drawCall.getTransformData().objectToWorld == state.drawCall.getTransformData().objectToWorld &&
      sameGeometry(source.state.inlineSubmeshes->front(), state.inlineSubmeshes->front());
  }

  void setRenderingState(const CachedBatch& batch, RasterGeometry geometry, Plan& plan) {
    auto geometries = std::make_shared<const std::vector<RasterGeometry>>(1, std::move(geometry));
    ExternalDrawState state = batch.storage->sources.front().state;
    state.inlineSubmeshes = geometries;
    state.inlineSubmeshParts.reset();
    state.assetKeys.reset();
    state.retainedHandle = 0;
    state.forceFullUpdate = true;
    state.deforming = true;
    state.mesh = reinterpret_cast<remixapi_MeshHandle>((plan.id << 1) | 1ull);
    state.drawCall.overrideGeometryData(std::shared_ptr<const RasterGeometry>(geometries, &geometries->front()));
    state.drawCall.drawCallID = 0;
    auto& transforms = state.drawCall.modifyTransformData();
    transforms.objectToWorld = Matrix4(batch.storage->cell.origin);
    transforms.objectToView = transforms.worldToView * transforms.objectToWorld;
    plan.renderingState = std::move(state);
  }

  BuildResult buildVertices(const Rc<DxvkContext>& ctx, const std::vector<Candidate>& candidates, CachedBatch& batch,
                            RetainedWorldBatcher::Stats& stats) {
    std::vector<Cpu::ReadSource> sources;
    sources.reserve(candidates.size());
    auto storage = std::make_shared<BatchStorage>();
    storage->sources.reserve(candidates.size());
    storage->cell = candidates.front().cell;
    for (size_t i = 0; i < candidates.size(); ++i) {
      const auto& candidate = candidates[i];
      const auto& geometry = candidate.state->inlineSubmeshes->front();
      // The source descriptions strongly retain every buffer until the snapshot is replaced.
      storage->sources.push_back({candidate.input->id, candidate.ref.part, candidate.input->passDomain,
                               candidate.input->wetnessRule, *candidate.state});
      auto transform = matrixArray(candidate.state->drawCall.getTransformData().objectToWorld);
      for (size_t axis = 0; axis < 3; ++axis) {
        transform[12 + axis] -= storage->cell.origin[axis];
      }
      sources.push_back({geometry.vertexCount, geometry.indexCount, transform, geometry.normalBuffer.defined()});
    }
    Cpu::Data packed;
    if (!Cpu::packRead(sources, {},
        [&](size_t source, uint32_t index, Cpu::Vertex& vertex) {
          return readVertex(candidates[source].state->inlineSubmeshes->front(), index, vertex);
        },
        [&](size_t source, uint32_t index, uint32_t& value) {
          return readIndex(candidates[source].state->inlineSubmeshes->front(), index, value);
        }, packed)) {
      return BuildResult::Pack;
    }
    const size_t vertexBytes = packed.vertices.size() * sizeof(Cpu::Vertex);
    ++stats.vertexAllocationAttempts;
    const Rc<DxvkBuffer> buffer = createBuffer(ctx, vertexBytes);
    if (!buffer.ptr()) {
      return BuildResult::Allocation;
    }
    ++stats.vertexAllocations;
    const DxvkBufferSlice slice(buffer);
    if (!slice.mapPtr(0)) {
      return BuildResult::Map;
    }
    std::memcpy(slice.mapPtr(0), packed.vertices.data(), vertexBytes);
    stats.vertexBytesUploaded += vertexBytes;
    RasterGeometry geometry = candidates.front().state->inlineSubmeshes->front();
    geometry.vertexCount = uint32_t(packed.vertices.size());
    geometry.positionBuffer = RasterBuffer(slice, offsetof(Cpu::Vertex, position), sizeof(Cpu::Vertex), VK_FORMAT_R32G32B32_SFLOAT);
    if (geometry.normalBuffer.defined()) {
      geometry.normalBuffer = RasterBuffer(slice, offsetof(Cpu::Vertex, normal), sizeof(Cpu::Vertex), VK_FORMAT_R32G32B32_SFLOAT);
    }
    if (geometry.texcoordBuffer.defined()) {
      geometry.texcoordBuffer = RasterBuffer(slice, offsetof(Cpu::Vertex, texcoord), sizeof(Cpu::Vertex), VK_FORMAT_R32G32_SFLOAT);
    }
    if (geometry.color0Buffer.defined()) {
      geometry.color0Buffer = RasterBuffer(slice, offsetof(Cpu::Vertex, color), sizeof(Cpu::Vertex), VK_FORMAT_B8G8R8A8_UNORM);
    }
    geometry.indexBuffer = {};
    geometry.indexCount = 0;
    geometry.externalMesh = nullptr;
    geometry.externalMaterial = nullptr;
    geometry.legacyDrawHash = geometry.legacyGeometryHash = geometry.legacyMaterialHash = geometry.legacyRuleFingerprint = 0;
    geometry.boundingBox = {};
    for (const Cpu::Vertex& vertex : packed.vertices) {
      AxisAlignedBoundingBox point;
      point.minPos = point.maxPos = Vector3(vertex.position[0], vertex.position[1], vertex.position[2]);
      geometry.boundingBox.unionWith(point);
    }
    const uint64_t vertexHash = XXH3_64bits_withSeed(packed.vertices.data(), vertexBytes, 0x574f524c44564231ull) | 1ull;
    geometry.immutableVertexDataId = vertexHash;
    geometry.hashes = {};
    geometry.hashes[HashComponents::VertexPosition] = vertexHash;
    geometry.hashes[HashComponents::VertexTexcoord] = vertexHash;
    geometry.hashes[HashComponents::VertexLayout] = hashVertexLayout(geometry) | 1ull;
    storage->geometry = std::move(geometry);
    storage->ranges = std::move(packed.ranges);
    storage->allIndices = std::make_shared<const std::vector<uint32_t>>(std::move(packed.indices));
    storage->bytes = vertexBytes + storage->allIndices->size() * sizeof(uint32_t);
    storage->physicalMembers.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      const auto& range = storage->ranges[i];
      storage->physicalMembers.push_back({candidates[i].input->id, candidates[i].ref.part, range.firstVertex,
                                   range.vertexCount, range.firstIndex, range.indexCount});
    }
    // All payload construction completes before it becomes immutable. No borrowed input escapes this call.
    batch.storage = std::move(storage);
    return BuildResult::Success;
  }

  BuildResult buildIndices(const Rc<DxvkContext>& ctx, const std::vector<bool>& visible, const CachedBatch& batch,
                           RetainedWorldBatcher::Stats& stats, std::shared_ptr<Plan>& output,
                           Cpu::VisibilityGeometryAction& action) {
    if (!batch.storage || visible.size() != batch.storage->ranges.size()) {
      return BuildResult::Plan;
    }
    const auto& previous = batch.plan->renderingState.inlineSubmeshes;
    const bool hasNonemptyRenderingState = previous && previous->size() == 1 &&
      previous->front().indexBuffer.defined() && previous->front().indexCount;
    action = Cpu::visibilityGeometryAction(visible, batch.lastNonemptyMask, hasNonemptyRenderingState);
    auto plan = std::make_shared<Plan>(*batch.plan);
    plan->visible = action != Cpu::VisibilityGeometryAction::Hide;
    plan->revision = Cpu::visibilityGeometryRevision(plan->revision, action);
    plan->changed = true;
    if (action != Cpu::VisibilityGeometryAction::Build) {
      // Hide retains the last nonempty state. Revealing its exact mask needs only instance visibility.
      output.swap(plan);
      return BuildResult::Success;
    }
    std::vector<uint32_t> indices;
    if (!Cpu::selectIndices(*batch.storage->allIndices, batch.storage->ranges, visible, indices) || indices.empty()) {
      return BuildResult::Plan;
    }
    const size_t bytes = indices.size() * sizeof(uint32_t);
    ++stats.indexAllocationAttempts;
    const Rc<DxvkBuffer> buffer = createBuffer(ctx, bytes);
    if (!buffer.ptr()) {
      return BuildResult::Allocation;
    }
    ++stats.indexAllocations;
    const DxvkBufferSlice slice(buffer);
    if (!slice.mapPtr(0)) {
      return BuildResult::Map;
    }
    std::memcpy(slice.mapPtr(0), indices.data(), bytes);
    stats.indexBytesUploaded += bytes;
    stats.indexTrianglesUploaded += indices.size() / 3;
    RasterGeometry geometry = batch.storage->geometry;
    geometry.indexBuffer = RasterBuffer(slice, 0, sizeof(uint32_t), VK_INDEX_TYPE_UINT32);
    geometry.indexCount = uint32_t(indices.size());
    geometry.hashes[HashComponents::Indices] = XXH3_64bits_withSeed(indices.data(), bytes, geometry.immutableVertexDataId) | 1ull;
    geometry.hashes[HashComponents::GeometryDescriptor] = hashGeometryDescriptor(geometry.indexCount, geometry.vertexCount,
      uint32_t(VK_INDEX_TYPE_UINT32), uint32_t(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)) | 1ull;
    geometry.hashes.precombine();
    setRenderingState(batch, std::move(geometry), *plan);
    output.swap(plan);
    return BuildResult::Success;
  }
}

struct RetainedWorldBatcher::Impl {
  std::vector<CachedBatch> batches;
  Cpu::VisibilityIndex visibilityIndex;
  Stats stats;
  uint64_t nextId = 0;
  Cpu::MembershipRevision membership;
};

RetainedWorldBatcher::RetainedWorldBatcher() : m_impl(std::make_unique<Impl>()) { }
RetainedWorldBatcher::~RetainedWorldBatcher() = default;

const RetainedWorldBatcher::Stats& RetainedWorldBatcher::stats() const {
  return m_impl->stats;
}

void RetainedWorldBatcher::clear() {
  m_impl->batches.clear();
  m_impl->visibilityIndex.clear();
  m_impl->membership.clear();
  m_impl->stats.batches = 0;
  m_impl->stats.bytes = 0;
}

uint64_t RetainedWorldBatcher::membershipRevision() const {
  return m_impl->membership.value();
}

bool RetainedWorldBatcher::detachChildren(const Rc<DxvkContext>& ctx, uint64_t expectedRevision,
                                         const std::vector<uint64_t>& childIds, std::vector<Plan>& output) {
  return detachChildren(ctx, expectedRevision, childIds, {}, false, output);
}

bool RetainedWorldBatcher::detachChildren(const Rc<DxvkContext>& ctx, uint64_t expectedRevision,
                                         const std::vector<uint64_t>& childIds, const std::vector<VisibilityInput>& visibility,
                                         bool republishAll, std::vector<Plan>& output) {
  auto& stats = m_impl->stats;
  ++stats.detachUpdates;
  try {
    std::vector<VisibilityInput> changes;
    if (!Cpu::detachmentChanges(childIds, visibility, changes)) {
      ++stats.detachFailures;
      ++stats.failures;
      ++stats.failedInput;
      return false;
    }
    if (!updateVisibility(ctx, expectedRevision, changes, republishAll, output)) {
      ++stats.detachFailures;
      return false;
    }
    // No throwing work remains after the visibility transaction commits. Forgetting the locations
    // permanently quarantines old packed ranges until the next successful structural preparation.
    stats.detachedChildren += Cpu::forgetDetachedChildren(m_impl->visibilityIndex, childIds);
    ++stats.detachCommits;
    return true;
  } catch (const std::bad_alloc&) {
    ++stats.detachFailures;
    ++stats.failures;
    ++stats.failedAllocation;
    return false;
  } catch (const std::exception&) {
    ++stats.detachFailures;
    ++stats.failures;
    ++stats.failedException;
    return false;
  }
}

bool RetainedWorldBatcher::updateVisibility(const Rc<DxvkContext>& ctx, uint64_t expectedRevision,
                                           const std::vector<VisibilityInput>& changes, bool republishAll,
                                           std::vector<Plan>& output) {
  ScopedCpuProfileZone();
  auto& stats = m_impl->stats;
  ++stats.visibilityUpdates;
  stats.lastVisibilityChanges = changes.size();
  stats.lastVisibilityAffected = stats.lastVisibilityPlans = 0;
  const auto fail = [&](BuildResult result) {
    ++stats.visibilityFailures;
    ++stats.failures;
    switch (result) {
    case BuildResult::Input: ++stats.failedInput; break;
    case BuildResult::Plan: ++stats.failedPlan; break;
    case BuildResult::Pack: ++stats.failedPack; break;
    case BuildResult::Allocation: ++stats.failedAllocation; break;
    case BuildResult::Map: ++stats.failedMap; break;
    default: ++stats.failedException; break;
    }
    return false;
  };
  if (!ctx.ptr() || !expectedRevision || expectedRevision != membershipRevision()) {
    return fail(BuildResult::Input);
  }
  try {
    std::vector<Cpu::VisibilityPatch> patches;
    if (!Cpu::visibilityPatches(m_impl->visibilityIndex, uint32_t(m_impl->batches.size()), changes,
        [&](uint32_t batch) -> const std::vector<bool>& { return m_impl->batches[batch].visible; }, patches)) {
      return fail(BuildResult::Plan);
    }
    struct Pending {
      uint32_t batch = 0;
      std::vector<bool> visible;
      std::vector<bool> lastNonemptyMask;
      std::shared_ptr<Plan> plan;
      Cpu::VisibilityGeometryAction action = Cpu::VisibilityGeometryAction::Hide;
    };
    std::vector<Pending> pending;
    BuildResult stageResult = BuildResult::Success;
    if (!Cpu::stageVisibilityPatches<Pending>(patches, [&](const Cpu::VisibilityPatch& patch, Pending& stage) {
        ++stats.visibilityIndexBuildAttempts;
        ++stats.indexBuildAttempts;
        stageResult = buildIndices(ctx, patch.visible, m_impl->batches[patch.batch], stats, stage.plan, stage.action);
        if (stageResult != BuildResult::Success) {
          return false;
        }
        stage.batch = patch.batch;
        stage.visible = patch.visible;
        Cpu::stageLastNonemptyMask(stage.action, patch.visible, stage.lastNonemptyMask);
        return true;
      }, pending)) {
      return fail(stageResult);
    }
    // Source states, VB, ranges and all-member indices are borrowed in place; only new masks/plans are staged.
    std::vector<Plan> plans;
    plans.reserve(republishAll ? m_impl->batches.size() : pending.size());
    Cpu::visitVisibilityPublications(uint32_t(m_impl->batches.size()), patches, republishAll,
      [&](uint32_t batch, uint32_t patch) {
        const bool changed = patch != UINT32_MAX;
        plans.push_back(changed ? *pending[patch].plan : *m_impl->batches[batch].plan);
        plans.back().changed = changed;
      });
    // Output construction and every index allocation complete before these no-throw publication swaps.
    for (auto& stage : pending) {
      auto& batch = m_impl->batches[stage.batch];
      batch.visible.swap(stage.visible);
      if (stage.action == Cpu::VisibilityGeometryAction::Build) {
        batch.lastNonemptyMask.swap(stage.lastNonemptyMask);
      }
      batch.plan.swap(stage.plan);
      stats.revealReuse += stage.action == Cpu::VisibilityGeometryAction::Reuse;
      stats.allHiddenNoAllocation += stage.action == Cpu::VisibilityGeometryAction::Hide;
    }
    output.swap(plans);
    ++stats.visibilityCommits;
    stats.visibilityAffectedBatches += patches.size();
    stats.visibilityIndexBuilds += patches.size();
    stats.indexBuilds += patches.size();
    stats.visibilityNoops += patches.empty();
    stats.visibilityRepublishes += republishAll;
    stats.lastVisibilityAffected = patches.size();
    stats.lastVisibilityPlans = output.size();
    return true;
  } catch (const std::bad_alloc&) {
    return fail(BuildResult::Allocation);
  } catch (const std::exception&) {
    return fail(BuildResult::Exception);
  }
}

bool RetainedWorldBatcher::prepare(const Rc<DxvkContext>& ctx, const std::vector<ParentInput>& inputs,
                                  float cellSize, const std::vector<uint64_t>& pendingDetachIds,
                                  std::vector<Plan>& output) {
  ScopedCpuProfileZone();
  ++m_impl->stats.prepares;
  const auto fail = [&](BuildResult reason) {
    ++m_impl->stats.failures;
    switch (reason) {
    case BuildResult::Input: ++m_impl->stats.failedInput; break;
    case BuildResult::Plan: ++m_impl->stats.failedPlan; break;
    case BuildResult::Pack: ++m_impl->stats.failedPack; break;
    case BuildResult::Allocation: ++m_impl->stats.failedAllocation; break;
    case BuildResult::Map: ++m_impl->stats.failedMap; break;
    default: ++m_impl->stats.failedException; break;
    }
    return false;
  };
  if (!ctx.ptr() || !std::isfinite(cellSize) || cellSize <= 0 || inputs.size() > UINT32_MAX) {
    return fail(BuildResult::Input);
  }
  FullPrepareTrace trace(ctx->getDevice()->getCurrentFrameId(), m_impl->stats.prepares);
  try {
    std::unordered_map<uint64_t, size_t> currentById;
    std::vector<Cpu::FixedInput> fixedInputs;
    fixedInputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].id || inputs[i].parts.size() > UINT32_MAX ||
          !currentById.emplace(inputs[i].id, i).second) {
        return fail(BuildResult::Input);
      }
      fixedInputs.push_back({inputs[i].id, uint32_t(inputs[i].parts.size())});
    }
    std::vector<std::vector<Cpu::FixedSource>> fixedSources;
    fixedSources.reserve(m_impl->batches.size());
    for (const auto& batch : m_impl->batches) {
      if (!batch.storage) {
        return fail(BuildResult::Plan);
      }
      fixedSources.emplace_back();
      fixedSources.back().reserve(batch.storage->sources.size());
      for (const auto& source : batch.storage->sources) {
        fixedSources.back().push_back({source.id, source.part});
      }
    }
    Cpu::FixedSelection fixed;
    if (!Cpu::retainFixedChunks(fixedInputs, fixedSources, m_impl->visibilityIndex, pendingDetachIds,
        [&](uint32_t input, uint32_t batch, uint32_t slot) {
          const auto& old = m_impl->batches[batch];
          Cell cell;
          return sameSource(old.storage->sources[slot], inputs[input]) &&
            findCell(inputs[input].parts[old.storage->sources[slot].part], cellSize, cell) && old.storage->cell == cell;
        }, fixed)) {
      return fail(BuildResult::Plan);
    }
    uint64_t fixedBytes = 0;
    const uint64_t fixedParts = fixed.residentParts;
    const uint64_t activeFixedParts = fixed.activeParts;
    std::vector<Cpu::VisibilityPatch> fixedChunks;
    for (size_t i = 0; i < m_impl->batches.size(); ++i) {
      const auto& slots = fixed.slots[i];
      const size_t active = size_t(std::count(slots.begin(), slots.end(), true));
      if (!active) {
        continue;
      }
      const auto& batch = m_impl->batches[i];
      const auto& storage = *batch.storage;
      if (!storage.allIndices || storage.sources.size() != storage.ranges.size() ||
          storage.sources.size() != storage.physicalMembers.size()) {
        return fail(BuildResult::Plan);
      }
      fixedBytes += storage.bytes + storage.allIndices->size() * sizeof(uint32_t);
      fixedChunks.push_back({uint32_t(i), slots});
    }
    if (fixedBytes > kMaxBytes || fixedChunks.size() > kMaxBatches || fixedParts > kMaxParts) {
      return fail(BuildResult::Plan);
    }
    trace.mark(1);
    std::vector<Candidate> candidates;
    std::vector<uint32_t> partCounts(inputs.size(), 0);
    std::vector<size_t> inputOrder;
    inputOrder.reserve(inputs.size());
    uint64_t deltaCandidateChildren = 0, deltaCandidateParts = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!fixed.inputs[i] && !inputs[i].parts.empty()) {
        inputOrder.push_back(i);
        ++deltaCandidateChildren;
        deltaCandidateParts += inputs[i].parts.size();
      }
    }
    std::sort(inputOrder.begin(), inputOrder.end(), [&](size_t a, size_t b) {
      return inputs[a].id < inputs[b].id;
    });
    size_t totalParts = 0;
    const Cpu::Limits limits;
    for (size_t i : inputOrder) {
      const Input& input = inputs[i];
      if (input.parts.size() > kMaxParts - totalParts) {
        ++m_impl->stats.preflightRejectedInputs;
        continue;
      }
      totalParts += input.parts.size();
      partCounts[i] = uint32_t(input.parts.size());
      for (size_t part = 0; part < input.parts.size(); ++part) {
        const auto& state = input.parts[part];
        Candidate candidate {{uint32_t(i), uint32_t(part)}, &input, &state, {}};
        if (!eligible(state) || !readableGeometry(state.inlineSubmeshes->front()) || !findCell(state, cellSize, candidate.cell)) {
          ++m_impl->stats.rejectedParts;
          continue;
        }
        const auto& geometry = state.inlineSubmeshes->front();
        if (geometry.vertexCount > limits.maxVertices || geometry.indexCount > limits.maxIndices ||
            uint64_t(geometry.vertexCount) * sizeof(Cpu::Vertex) + uint64_t(geometry.indexCount) * sizeof(uint32_t) > limits.maxBytes) {
          ++m_impl->stats.rejectedParts;
          continue;
        }
        candidates.push_back(candidate);
      }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      return a.input->id < b.input->id || (a.input->id == b.input->id && a.ref.part < b.ref.part);
    });
    std::vector<Bucket> buckets;
    std::unordered_multimap<uint64_t, size_t> bucketLookup;
    for (const Candidate& candidate : candidates) {
      const uint64_t hash = bucketHash(candidate);
      const auto& geometry = candidate.state->inlineSubmeshes->front();
      const auto found = bucketLookup.equal_range(hash);
      size_t index = buckets.size();
      for (auto it = found.first; it != found.second; ++it) {
        const Bucket& bucket = buckets[it->second];
        const uint64_t vertices = bucket.vertices + geometry.vertexCount;
        const uint64_t indices = bucket.indices + geometry.indexCount;
        if (compatible(bucket.sources.front(), candidate) && vertices <= limits.maxVertices && indices <= limits.maxIndices &&
            vertices * sizeof(Cpu::Vertex) + indices * sizeof(uint32_t) <= limits.maxBytes) {
          index = it->second;
          break;
        }
      }
      if (index == buckets.size()) {
        bucketLookup.emplace(hash, index);
        buckets.emplace_back();
      }
      buckets[index].sources.push_back(candidate);
      buckets[index].vertices += geometry.vertexCount;
      buckets[index].indices += geometry.indexCount;
    }
    trace.mark(2);
    std::vector<std::vector<Cpu::PartRef>> refs;
    refs.reserve(buckets.size());
    for (const auto& bucket : buckets) {
      refs.emplace_back();
      for (const auto& source : bucket.sources) {
        refs.back().push_back(source.ref);
      }
    }
    const auto active = Cpu::completeInputs(partCounts, refs);
    std::vector<Cpu::BudgetBucket> projected(buckets.size());
    // Establish complete resident components and their maximum storage before packing or creating buffers.
    for (size_t i = 0; i < buckets.size(); ++i) {
      Bucket& bucket = buckets[i];
      bucket.sources.erase(std::remove_if(bucket.sources.begin(), bucket.sources.end(), [&](const Candidate& source) {
        return !active[source.ref.input];
      }), bucket.sources.end());
      if (bucket.sources.empty()) {
        continue;
      }
      auto& budget = projected[i];
      budget.stableKey = bucket.sources.front().input->id;
      for (const Candidate& source : bucket.sources) {
        const auto& geometry = source.state->inlineSubmeshes->front();
        budget.parts.push_back(source.ref);
        budget.bytes += uint64_t(geometry.vertexCount) * sizeof(Cpu::Vertex) + uint64_t(geometry.indexCount) * sizeof(uint32_t) * 2;
      }
    }
    Cpu::BudgetSelection selection;
    if (!Cpu::selectBudget(partCounts, projected,
        {kMaxBytes - fixedBytes, uint32_t(kMaxBatches - fixedChunks.size()), uint32_t(kMaxParts - fixedParts)}, selection)) {
      return fail(BuildResult::Plan);
    }
    m_impl->stats.preflightRejectedInputs += selection.rejectedInputs;
    m_impl->stats.preflightRejectedBatches += selection.rejectedBatches;
    m_impl->stats.preflightRejectedComponents += selection.rejectedComponents;
    for (size_t i = 0; i < selection.inputs.size(); ++i) {
      m_impl->stats.rejectedInputs += !fixed.inputs[i] && !selection.inputs[i];
    }
    trace.mark(3);
    std::vector<CachedBatch> next;
    std::vector<Plan> plans;
    size_t bytes = 0;
    uint64_t vertexBuilds = 0;
    uint64_t indexBuilds = 0;
    uint64_t vertexReuses = 0;
    uint64_t revealReuse = 0;
    uint64_t allHiddenNoAllocation = 0;
    uint64_t nextId = m_impl->nextId;
    const size_t selectedBatches = fixedChunks.size() + size_t(selection.selectedBatches);
    const uint64_t selectedBytes = fixedBytes + selection.selectedBytes;
    std::vector<std::vector<bool>> activeSlots;
    plans.reserve(selectedBatches);
    activeSlots.reserve(selectedBatches);
    BuildResult fixedResult = BuildResult::Success;
    // Staged views share immutable physical slots. All old cache entries remain untouched until publication.
    if (!Cpu::stageVisibilityPatches<CachedBatch>(fixedChunks,
        [&](const Cpu::VisibilityPatch& fixedChunk, CachedBatch& batch) {
          const auto& old = m_impl->batches[fixedChunk.batch];
          batch = old;
          const auto& storage = *old.storage;
          std::vector<bool> visible(storage.sources.size(), false);
          for (size_t slot = 0; slot < storage.sources.size(); ++slot) {
            if (fixedChunk.visible[slot]) {
              visible[slot] = inputs[currentById.at(storage.sources[slot].id)].visible;
            }
          }
          const bool changed = visible != old.visible;
          if (changed) {
            ++m_impl->stats.indexBuildAttempts;
            std::shared_ptr<Plan> nextPlan;
            auto action = Cpu::VisibilityGeometryAction::Hide;
            fixedResult = buildIndices(ctx, visible, batch, m_impl->stats, nextPlan, action);
            if (fixedResult != BuildResult::Success) {
              return false;
            }
            Cpu::stageLastNonemptyMask(action, visible, batch.lastNonemptyMask);
            batch.plan.swap(nextPlan);
            batch.visible.swap(visible);
            revealReuse += action == Cpu::VisibilityGeometryAction::Reuse;
            allHiddenNoAllocation += action == Cpu::VisibilityGeometryAction::Hide;
            ++indexBuilds;
          }
          plans.push_back(*batch.plan);
          plans.back().changed = changed;
          if (!Cpu::filterPhysicalMembers(storage.physicalMembers, fixedChunk.visible, plans.back().members)) {
            fixedResult = BuildResult::Plan;
            return false;
          }
          activeSlots.push_back(fixedChunk.visible);
          bytes += storage.bytes + storage.allIndices->size() * sizeof(uint32_t);
          ++vertexReuses;
          return true;
        }, next, selectedBatches)) {
      return fail(fixedResult);
    }
    trace.mark(4);
    for (size_t i = 0; i < buckets.size(); ++i) {
      if (!selection.buckets[i]) {
        continue;
      }
      const Bucket& bucket = buckets[i];
      CachedBatch batch;
      batch.plan = std::make_shared<Plan>();
      ++m_impl->stats.vertexBuildAttempts;
      if (const BuildResult result = buildVertices(ctx, bucket.sources, batch, m_impl->stats); result != BuildResult::Success) {
        return fail(result);
      }
      batch.plan->id = ++nextId;
      ++vertexBuilds;
      std::vector<bool> visible;
      visible.reserve(bucket.sources.size());
      for (const auto& source : bucket.sources) {
        visible.push_back(source.input->visible);
      }
      ++m_impl->stats.indexBuildAttempts;
      std::shared_ptr<Plan> nextPlan;
      auto action = Cpu::VisibilityGeometryAction::Hide;
      if (const BuildResult result = buildIndices(ctx, visible, batch, m_impl->stats, nextPlan, action);
          result != BuildResult::Success) {
        return fail(result);
      }
      Cpu::stageLastNonemptyMask(action, visible, batch.lastNonemptyMask);
      batch.plan.swap(nextPlan);
      batch.visible = std::move(visible);
      revealReuse += action == Cpu::VisibilityGeometryAction::Reuse;
      allHiddenNoAllocation += action == Cpu::VisibilityGeometryAction::Hide;
      ++indexBuilds;
      // Charge the packed CPU topology and both current GPU buffers before publishing anything.
      bytes += batch.storage->bytes + batch.storage->allIndices->size() * sizeof(uint32_t);
      if (bytes > selectedBytes || next.size() >= selectedBatches) {
        return fail(BuildResult::Plan);
      }
      plans.push_back(*batch.plan);
      plans.back().changed = true;
      plans.back().members = batch.storage->physicalMembers;
      activeSlots.emplace_back(batch.storage->sources.size(), true);
      next.push_back(std::move(batch));
    }
    if (next.size() != selectedBatches || bytes > selectedBytes || activeSlots.size() != next.size()) {
      return fail(BuildResult::Plan);
    }
    trace.mark(5);
    std::vector<uint32_t> physicalCounts;
    std::vector<std::vector<Cpu::IndexedMember>> indexedMembers;
    physicalCounts.reserve(next.size());
    indexedMembers.reserve(next.size());
    for (size_t i = 0; i < next.size(); ++i) {
      const auto& storage = *next[i].storage;
      physicalCounts.push_back(uint32_t(storage.sources.size()));
      indexedMembers.emplace_back();
      for (size_t slot = 0; slot < storage.sources.size(); ++slot) {
        if (activeSlots[i][slot]) {
          indexedMembers.back().push_back({storage.sources[slot].id, uint32_t(slot)});
        }
      }
    }
    Cpu::VisibilityIndex nextIndex;
    if (!Cpu::buildPhysicalVisibilityIndex(physicalCounts, indexedMembers, nextIndex)) {
      return fail(BuildResult::Plan);
    }
    // Every allocating operation precedes the no-throw publication of this membership generation.
    output.swap(plans);
    m_impl->batches.swap(next);
    m_impl->visibilityIndex.swap(nextIndex);
    m_impl->membership.publish();
    m_impl->nextId = nextId;
    m_impl->stats.vertexBuilds += vertexBuilds;
    m_impl->stats.indexBuilds += indexBuilds;
    m_impl->stats.vertexReuses += vertexReuses;
    m_impl->stats.revealReuse += revealReuse;
    m_impl->stats.allHiddenNoAllocation += allHiddenNoAllocation;
    m_impl->stats.batches = m_impl->batches.size();
    m_impl->stats.bytes = bytes;
    m_impl->stats.lastProjectedBatches = fixedChunks.size() + selection.projectedBatches;
    m_impl->stats.lastProjectedBytes = fixedBytes + selection.projectedBytes;
    m_impl->stats.lastSelectedBatches = selectedBatches;
    m_impl->stats.lastSelectedBytes = selectedBytes;
    m_impl->stats.lastFixedChunks = fixedChunks.size();
    m_impl->stats.lastFixedBytes = fixedBytes;
    m_impl->stats.lastResidentParts = fixedParts + selection.selectedParts;
    m_impl->stats.lastActiveParts = activeFixedParts + selection.selectedParts;
    m_impl->stats.lastDeltaCandidateChildren = deltaCandidateChildren;
    m_impl->stats.lastDeltaCandidateParts = deltaCandidateParts;
    m_impl->stats.lastDeltaAdmittedChildren = std::count(selection.inputs.begin(), selection.inputs.end(), true);
    m_impl->stats.lastDeltaAdmittedParts = selection.selectedParts;
    m_impl->stats.lastDeadChunks = fixed.slots.size() - fixedChunks.size();
    trace.complete();
    return true;
  } catch (const std::bad_alloc&) {
    return fail(BuildResult::Allocation);
  } catch (const std::exception&) {
    return fail(BuildResult::Exception);
  }
}

}
