/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#include <cmath>
#include <cstring>
#include <exception>

#include <remix/remix_c.h>

#include "rtx_retained_static_batch.h"
#include "rtx_retained_static_batch_cpu.h"
#include "rtx_retained_batch_policy.h"
#include "rtx_fork_material_derivation.h"
#include "rtx_scene_manager.h"

namespace dxvk {
namespace {
  using Reason = RetainedBatchReason;

  template<typename T>
  bool sameRepresentation(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
  }

  bool sameMaterial(const MaterialData& a, const MaterialData& b) {
    if (a.getType() != MaterialDataType::Opaque || b.getType() != MaterialDataType::Opaque ||
        a.getIgnored() || b.getIgnored() || a.getParticleSystemDesc() || b.getParticleSystemDesc() ||
        a.usesLegacyDefaults() != b.usesLegacyDefaults()) {
      return false;
    }
    // This includes every authored field and resource reference. Different padding only rejects a candidate.
    return sameRepresentation(a.getOpaqueMaterialData(), b.getOpaqueMaterialData());
  }

  bool sameTransforms(const DrawCallTransforms& a, const DrawCallTransforms& b) {
    return !a.instancesToObject && !b.instancesToObject && !a.instancesFromHost && !b.instancesFromHost &&
      sameRepresentation(a.objectToWorld, b.objectToWorld) &&
      sameRepresentation(a.objectToView, b.objectToView) &&
      sameRepresentation(a.worldToView, b.worldToView) &&
      sameRepresentation(a.viewToProjection, b.viewToProjection) &&
      sameRepresentation(a.textureTransform, b.textureTransform) &&
      a.enableClipPlane == b.enableClipPlane && sameRepresentation(a.clipPlane, b.clipPlane) &&
      a.texgenMode == b.texgenMode;
  }

  Reason eligible(const ExternalDrawState& state) {
    if (!state.frontendResolved || !state.assetKeys.has_value() ||
        !state.inlineSubmeshes || state.inlineSubmeshes->size() != 1 || !state.resolvedMaterial) {
      return Reason::Source;
    }
    if (state.inlineSubmeshParts || state.deforming || state.optionalParticleDesc || state.materialOverride ||
        !state.gpuInstancingTransforms.empty() || state.sortOrder || state.drawCall.usesVertexShader ||
        state.drawCall.newGeometryEveryFrame || state.drawCall.isUsingRaytracedRenderTarget ||
        state.drawCall.materialIndependentGeometry || state.drawCall.getSkinningState().numBones ||
        state.drawCall.getSkinningState().numBonesPerVertex || !state.drawCall.getSkinningState().pBoneMatrices.empty()) {
      return Reason::Special;
    }
    CategoryFlags allowed;
    allowed.set(InstanceCategories::IgnoreLights);
    allowed.set(InstanceCategories::IgnoreAntiCulling);
    allowed.set(InstanceCategories::IgnoreMotionBlur);
    allowed.set(InstanceCategories::IgnoreOpacityMicromap);
    allowed.set(InstanceCategories::IgnoreAlphaChannel);
    allowed.set(InstanceCategories::IgnoreBakedLighting);
    if (state.drawCall.getCategoryFlags().raw() & ~allowed.raw()) {
      return Reason::Categories;
    }
    const MaterialData& material = *state.resolvedMaterial;
    if (material.getType() != MaterialDataType::Opaque || material.getIgnored() || material.getParticleSystemDesc()) {
      return Reason::Material;
    }
    const auto& opaque = material.getOpaqueMaterialData();
    if (!retained_batch_policy::hasNoEffectiveAuthoredEmission(opaque.getEnableEmission(), opaque.getEmissiveIntensity()) ||
        opaque.getSpriteSheetFPS() ||
        opaque.getSpriteSheetRows() > 1 || opaque.getSpriteSheetCols() > 1 || opaque.getHeightTexture().isValid()) {
      return Reason::MaterialService;
    }
    const RtSurface::AlphaState alpha = state.alphaState.has_value() ? *state.alphaState
      : deriveAlphaState(FixedFunctionAlphaInput::fromLegacyMaterial(state.drawCall.getMaterialData()),
                         material, state.drawCall.getCategoryFlags());
    if (!alpha.isBlendingDisabled || !alpha.isFullyOpaque || alpha.alphaTestType != AlphaTestType::kAlways ||
        alpha.isParticle || alpha.isDecal || alpha.emissiveBlend) {
      return Reason::Alpha;
    }
    const auto& transforms = state.drawCall.getTransformData();
    const RasterGeometry& geometry = state.inlineSubmeshes->front();
    if (state.cameraType != CameraType::Main || state.drawCall.cameraType != CameraType::Main ||
        transforms.instancesToObject || transforms.instancesFromHost ||
        transforms.texgenMode != TexGenMode::None || transforms.enableClipPlane) {
      return Reason::CameraTransform;
    }
    if (!geometry.legacyDrawHash || !geometry.legacyGeometryHash || !geometry.legacyRuleFingerprint) {
      return Reason::Identity;
    }
    return geometry.numBonesPerVertex || geometry.skinningRetarget ||
      geometry.blendWeightBuffer.defined() || geometry.blendIndicesBuffer.defined() ? Reason::Skinning : Reason::None;
  }

  bool matchesLayout(const RasterBuffer& attribute, const RasterBuffer& position,
                     uint32_t offset, VkFormat format) {
    return attribute.defined() && attribute.matches(position) &&
      attribute.offsetFromSlice() == offset &&
      attribute.stride() == sizeof(remixapi_HardcodedVertex) && attribute.vertexFormat() == format;
  }

  Reason readSource(const RasterGeometry& geometry, RetainedStaticBatchCpu::Source& source) {
    const RasterBuffer& position = geometry.positionBuffer;
    const RasterBuffer& indices = geometry.indexBuffer;
    if (geometry.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || !position.defined() || !indices.defined() ||
        position.offsetFromSlice() != offsetof(remixapi_HardcodedVertex, position) ||
        position.stride() != sizeof(remixapi_HardcodedVertex) || position.vertexFormat() != VK_FORMAT_R32G32B32_SFLOAT ||
        indices.indexType() != VK_INDEX_TYPE_UINT32 || indices.stride() != sizeof(uint32_t)) {
      return Reason::BufferLayout;
    }
    if (position.isPendingGpuWrite() || indices.isPendingGpuWrite()) {
      return Reason::PendingWrite;
    }
    if (!(position.buffer()->memFlags() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
        !(indices.buffer()->memFlags() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      return Reason::HostVisible;
    }
    if (!matchesLayout(geometry.texcoordBuffer, position, offsetof(remixapi_HardcodedVertex, texcoord), VK_FORMAT_R32G32_SFLOAT) ||
        !matchesLayout(geometry.color0Buffer, position, offsetof(remixapi_HardcodedVertex, color), VK_FORMAT_B8G8R8A8_UNORM) ||
        (geometry.normalBuffer.defined() &&
         !matchesLayout(geometry.normalBuffer, position, offsetof(remixapi_HardcodedVertex, normal), VK_FORMAT_R32G32B32_SFLOAT)) ||
        indices.offsetFromSlice() > indices.length()) {
      return Reason::BufferLayout;
    }
    const auto* pVertices = static_cast<const uint8_t*>(position.mapPtr());
    const auto* pIndices = static_cast<const uint8_t*>(indices.mapPtr());
    if (!pVertices || !pIndices) {
      return Reason::Map;
    }
    source = {pVertices, position.length(), geometry.vertexCount,
              pIndices + indices.offsetFromSlice(), indices.length() - indices.offsetFromSlice(), geometry.indexCount};
    return Reason::None;
  }

bool sameBuffer(const RasterBuffer& a, const RasterBuffer& b, bool index = false) {
    if (a.defined() != b.defined()) {
      return false;
    }
    if (!a.defined()) {
      return true;
    }
    return a.buffer().ptr() == b.buffer().ptr() && a.offset() == b.offset() &&
      a.length() == b.length() && a.offsetFromSlice() == b.offsetFromSlice() &&
      a.stride() == b.stride() &&
      (index ? a.indexType() == b.indexType() : a.vertexFormat() == b.vertexFormat());
  }

  bool sameSource(const RasterGeometry& a, const RasterGeometry& b) {
    return a.vertexCount == b.vertexCount && a.indexCount == b.indexCount &&
      a.topology == b.topology && a.cullMode == b.cullMode && a.frontFace == b.frontFace &&
      a.forceCullBit == b.forceCullBit && a.numBonesPerVertex == b.numBonesPerVertex &&
      a.legacyDrawHash == b.legacyDrawHash && a.legacyGeometryHash == b.legacyGeometryHash &&
      a.legacyMaterialHash == b.legacyMaterialHash && a.legacyRuleFingerprint == b.legacyRuleFingerprint &&
      a.getHashForRule<rules::FullGeometryHash>() == b.getHashForRule<rules::FullGeometryHash>() &&
      sameBuffer(a.positionBuffer, b.positionBuffer) && sameBuffer(a.normalBuffer, b.normalBuffer) &&
      sameBuffer(a.texcoordBuffer, b.texcoordBuffer) && sameBuffer(a.color0Buffer, b.color0Buffer) &&
      sameBuffer(a.indexBuffer, b.indexBuffer, true) &&
      sameBuffer(a.blendWeightBuffer, b.blendWeightBuffer) && sameBuffer(a.blendIndicesBuffer, b.blendIndicesBuffer);
  }

  uint64_t hashSources(const std::vector<RasterGeometry>& sources) {
    uint64_t hash = 0x5245545354415443ull;
    const auto feed = [&](uint64_t value) { hash = XXH3_64bits_withSeed(&value, sizeof(value), hash); };
    feed(sources.size());
    const auto buffer = [&](const RasterBuffer& value, bool index = false) {
      feed(value.defined());
      if (value.defined()) {
        feed(reinterpret_cast<uintptr_t>(value.buffer().ptr()));
        feed(value.offset());
        feed(value.length());
        feed(value.offsetFromSlice());
        feed(value.stride());
        feed(index ? uint32_t(value.indexType()) : uint32_t(value.vertexFormat()));
      }
    };
    for (const RasterGeometry& source : sources) {
      feed(source.vertexCount); feed(source.indexCount); feed(source.topology);
      feed(source.cullMode); feed(source.frontFace); feed(source.forceCullBit); feed(source.numBonesPerVertex);
      feed(source.legacyDrawHash); feed(source.legacyGeometryHash); feed(source.legacyMaterialHash); feed(source.legacyRuleFingerprint);
      feed(source.getHashForRule<rules::FullGeometryHash>());
      buffer(source.positionBuffer); buffer(source.normalBuffer); buffer(source.texcoordBuffer);
      buffer(source.color0Buffer); buffer(source.indexBuffer, true);
      buffer(source.blendWeightBuffer); buffer(source.blendIndicesBuffer);
    }
    return hash;
  }

  ExternalDrawState mergedState(const ExternalDrawState& source,
                               const std::shared_ptr<const std::vector<RasterGeometry>>& geometry) {
    ExternalDrawState merged = source;
    merged.inlineSubmeshes = geometry;
    merged.drawCall.overrideGeometryData(std::shared_ptr<const RasterGeometry>(geometry, &geometry->front()));
    merged.assetKeys.reset();
    merged.mesh = reinterpret_cast<remixapi_MeshHandle>(geometry->front().hashes[HashComponents::VertexPosition]);
    return merged;
  }

  Rc<DxvkBuffer> createBuffer(const Rc<DxvkContext>& ctx, size_t size) {
    DxvkBufferCreateInfo info {};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    info.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    return ctx->getDevice()->createBuffer(info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                         DxvkMemoryStats::Category::RTXBuffer, "Retained static batch");
  }
}

bool RetainedStaticBatchCache::Key::operator==(const Key& other) const {
  if (sources.size() != other.sources.size()) {
    return false;
  }
  for (size_t i = 0; i < sources.size(); ++i) {
    if (!sameSource(sources[i], other.sources[i])) {
      return false;
    }
  }
  return true;
}

const char* RetainedStaticBatch::reasonName(RetainedBatchReason reason) {
  static constexpr const char* names[] = {
    "none", "source", "special", "categories", "material", "material_service", "alpha", "camera_transform",
    "identity", "skinning", "pair_state", "pair_transform", "pair_material", "pair_legacy_material",
    "pair_alpha", "pair_legacy_surface", "pair_layout", "replacement", "wetness", "buffer_layout",
    "pending_write", "host_visible", "map", "proof", "pack", "non_finite", "allocation"
  };
  static_assert(sizeof(names) / sizeof(names[0]) == size_t(RetainedBatchReason::Count));
  return size_t(reason) < size_t(RetainedBatchReason::Count) ? names[size_t(reason)] : "unknown";
}

RetainedBatchReason RetainedStaticBatch::rejectionReason(const ExternalDrawState& a, const ExternalDrawState& b) {
  if (const Reason why = eligible(a); why != Reason::None) {
    return why;
  }
  if (const Reason why = eligible(b); why != Reason::None) {
    return why;
  }
  if (a.cameraType != b.cameraType || a.doubleSided != b.doubleSided || a.categories.raw() != b.categories.raw() ||
      a.drawCall.getCategoryFlags().raw() != b.drawCall.getCategoryFlags().raw() ||
      a.drawCall.drawCallID != b.drawCall.drawCallID || a.drawCall.wetnessTag != b.drawCall.wetnessTag ||
      a.drawCall.cameraType != b.drawCall.cameraType || a.retainedHandle != b.retainedHandle ||
      a.alphaState.has_value() != b.alphaState.has_value() || a.legacySurface.has_value() != b.legacySurface.has_value()) {
    return Reason::PairState;
  }
  if (!sameTransforms(a.drawCall.getTransformData(), b.drawCall.getTransformData())) {
    return Reason::PairTransform;
  }
  if (!sameMaterial(*a.resolvedMaterial, *b.resolvedMaterial)) {
    return Reason::PairMaterial;
  }
  if (!sameRepresentation(a.drawCall.getMaterialData(), b.drawCall.getMaterialData())) {
    return Reason::PairLegacyMaterial;
  }
  if (a.alphaState && std::strcmp(firstAlphaStateDifference(*a.alphaState, *b.alphaState), "none")) {
    return Reason::PairAlpha;
  }
  if (a.legacySurface && std::strcmp(firstLegacySurfaceDifference(*a.legacySurface, *b.legacySurface), "none")) {
    return Reason::PairLegacySurface;
  }
  const RasterGeometry& ga = a.inlineSubmeshes->front();
  const RasterGeometry& gb = b.inlineSubmeshes->front();
  return ga.topology == gb.topology && ga.cullMode == gb.cullMode && ga.frontFace == gb.frontFace &&
    ga.forceCullBit == gb.forceCullBit && a.drawCall.getCullMode() == b.drawCall.getCullMode() &&
    ga.normalBuffer.defined() == gb.normalBuffer.defined() && ga.legacyRuleFingerprint == gb.legacyRuleFingerprint
      ? Reason::None : Reason::PairLayout;
}

bool RetainedStaticBatch::compatible(const ExternalDrawState& a, const ExternalDrawState& b) {
  return rejectionReason(a, b) == Reason::None;
}

bool RetainedStaticBatch::build(const Rc<DxvkContext>& ctx, const std::vector<ExternalDrawState>& sources,
                               const RetainedStaticBatchProof& proof, RetainedStaticBatchCache& cache, ExternalDrawState& output) {
  ScopedCpuProfileZone();
  static_assert(sizeof(remixapi_HardcodedVertex) == RetainedStaticBatchCpu::kVertexStride);
  ++cache.m_buildAttempts;
  const auto reject = [&](Reason reason) {
    ++cache.m_buildRejects[size_t(reason)];
    return false;
  };
  if (!ctx.ptr() || !proof.parent || !proof.apiOwnedOriginals || !proof.noMeshReplacements ||
      !proof.sameWetnessRule || !proof.identityToolsInactive || sources.size() < 2 || sources.size() > 64) {
    return reject(Reason::Proof);
  }
  try {
    std::vector<RetainedStaticBatchCpu::Source> inputs;
    inputs.reserve(sources.size());
    RetainedStaticBatchCache::Key key;
    key.sources.reserve(sources.size());
    size_t sourceBytes = 0;
    for (const ExternalDrawState& source : sources) {
      if (const Reason reason = rejectionReason(sources.front(), source); reason != Reason::None) {
        return reject(reason);
      }
      if (source.retainedHandle && source.retainedHandle != proof.parent) {
        return reject(Reason::Proof);
      }
      RetainedStaticBatchCpu::Source input;
      if (const Reason reason = readSource(source.inlineSubmeshes->front(), input); reason != Reason::None) {
        return reject(reason);
      }
      inputs.push_back(input);
      key.sources.push_back(source.inlineSubmeshes->front());
      const auto& geometry = source.inlineSubmeshes->front();
      sourceBytes += geometry.positionBuffer.buffer()->info().size + geometry.indexBuffer.buffer()->info().size;
    }
    const uint64_t cacheHash = hashSources(key.sources);
    if (const auto* found = cache.m_lru.find(cacheHash, key)) {
      ExternalDrawState merged = mergedState(sources.front(), *found);
      output = std::move(merged);
      return true;
    }
    RetainedStaticBatchCpu::Data packed;
    if (RetainedStaticBatchCpu::pack(inputs, {}, packed) != RetainedStaticBatchCpu::Result::Success) {
      return reject(Reason::Pack);
    }
    AxisAlignedBoundingBox bounds;
    for (size_t i = 0; i < packed.vertices.size(); i += sizeof(remixapi_HardcodedVertex)) {
      float xyz[3];
      std::memcpy(xyz, packed.vertices.data() + i + offsetof(remixapi_HardcodedVertex, position), sizeof(xyz));
      if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])) {
        return reject(Reason::NonFinite);
      }
      AxisAlignedBoundingBox point;
      point.minPos = point.maxPos = Vector3(xyz[0], xyz[1], xyz[2]);
      bounds.unionWith(point);
    }
    const Rc<DxvkBuffer> vertexBuffer = createBuffer(ctx, packed.vertices.size());
    const Rc<DxvkBuffer> indexBuffer = createBuffer(ctx, packed.indices.size() * sizeof(uint32_t));
    DxvkBufferSlice vertices(vertexBuffer);
    DxvkBufferSlice indices(indexBuffer);
    if (!vertices.mapPtr(0) || !indices.mapPtr(0)) {
      return reject(Reason::Map);
    }
    std::memcpy(vertices.mapPtr(0), packed.vertices.data(), packed.vertices.size());
    std::memcpy(indices.mapPtr(0), packed.indices.data(), packed.indices.size() * sizeof(uint32_t));

    RasterGeometry geometry = sources.front().inlineSubmeshes->front();
    geometry.vertexCount = uint32_t(packed.vertices.size() / sizeof(remixapi_HardcodedVertex));
    geometry.indexCount = uint32_t(packed.indices.size());
    geometry.positionBuffer = RasterBuffer(vertices, offsetof(remixapi_HardcodedVertex, position),
                                           sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32B32_SFLOAT);
    if (geometry.normalBuffer.defined()) {
      geometry.normalBuffer = RasterBuffer(vertices, offsetof(remixapi_HardcodedVertex, normal),
                                           sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32B32_SFLOAT);
    }
    geometry.texcoordBuffer = RasterBuffer(vertices, offsetof(remixapi_HardcodedVertex, texcoord),
                                           sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32_SFLOAT);
    geometry.color0Buffer = RasterBuffer(vertices, offsetof(remixapi_HardcodedVertex, color),
                                         sizeof(remixapi_HardcodedVertex), VK_FORMAT_B8G8R8A8_UNORM);
    geometry.indexBuffer = RasterBuffer(indices, 0, sizeof(uint32_t), VK_INDEX_TYPE_UINT32);
    geometry.boundingBox = bounds;
    geometry.legacyDrawHash = geometry.legacyGeometryHash = geometry.legacyMaterialHash = geometry.legacyRuleFingerprint = 0;
    geometry.externalMesh = nullptr;
    geometry.externalMaterial = nullptr;
    const uint64_t vertexHash = XXH3_64bits_withSeed(packed.vertices.data(), packed.vertices.size(), 0x5245545354415443ull);
    const uint64_t indexHash = XXH3_64bits_withSeed(packed.indices.data(), packed.indices.size() * sizeof(uint32_t), vertexHash);
    const uint64_t layout[] = {uint64_t(geometry.normalBuffer.defined()), uint64_t(geometry.cullMode),
                               uint64_t(geometry.frontFace), uint64_t(geometry.forceCullBit)};
    const uint64_t identity = XXH3_64bits_withSeed(layout, sizeof(layout), indexHash) | 1ull;
    geometry.hashes = {};
    geometry.hashes[HashComponents::VertexPosition] = identity;
    geometry.hashes[HashComponents::Indices] = indexHash;
    geometry.hashes[HashComponents::VertexTexcoord] = vertexHash;
    geometry.hashes[HashComponents::GeometryDescriptor] = identity;
    geometry.hashes[HashComponents::VertexLayout] = identity;
    geometry.hashes.precombine();

    auto fused = std::make_shared<const std::vector<RasterGeometry>>(1, std::move(geometry));
    ExternalDrawState merged = mergedState(sources.front(), fused);
    ++cache.m_creates;
    try {
      cache.m_lru.insert(cacheHash, std::move(key), fused,
                         sourceBytes + packed.vertices.size() + packed.indices.size() * sizeof(uint32_t));
    } catch (const std::bad_alloc&) {
      // The child still owns the complete geometry when the optional cache cannot allocate.
    }
    output = std::move(merged);
    return true;
  } catch (const std::exception&) {
    return reject(Reason::Allocation);
  }
}

}
