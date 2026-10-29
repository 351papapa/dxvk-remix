#include "rtx_water_foam_geometry.h"
#include "rtx_context.h"
#include "rtx_shader_manager.h"
#include "rtx_option.h"
#include "rtx_imgui.h"
#include "dxvk_scoped_annotation.h"
#include <rtx_shaders/water_foam_surface.h>
#include <rtx_shaders/water_foam_surface_culled.h>
#include <rtx_shaders/water_foam_surface_live.h>
#include <rtx_shaders/water_foam_surface_live_culled.h>
#include <rtx_shaders/water_foam_surface_live_pair_reuse.h>
#include <rtx_shaders/water_foam_surface_live_culled_pair_reuse.h>
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <exception>
#include "../../util/util_error.h"

namespace dxvk {
namespace {
class WaterFoamGeometryOptions {
  RTX_OPTION("rtx.water", bool, foamTrianglePairReuse, false,
    "Experimental: reuse identical live-water vertex evaluations between paired foam triangles. Preserves subdivisions, vertex attributes, triangle order and independent transparent-triangle collapse.");
  RTX_OPTION("rtx.water", bool, foamValidateTrianglePairReuse, false,
    "Diagnostic: compare all 9 words per foam vertex between both live-water variants on identical GPU inputs every 600 frames. Uses asynchronous readback; disable for performance measurements.");
};
// The sheet generator uses its own descriptors before the frame constants exist.
//
// No BINDLESS_ENABLED() and no COMMON_RAYTRACING_BINDINGS. This pass is
// self-contained: it reads saved density and current generated water, then writes vertices
// through explicit descriptors. It does not need this frame's common constants
// or acceleration structure, which are not ready during scene preparation.
class WaterFoamSurfaceShader : public ManagedShader {
  SHADER_SOURCE(WaterFoamSurfaceShader, VK_SHADER_STAGE_COMPUTE_BIT, water_foam_surface)
  PUSH_CONSTANTS(WaterFoamSurfaceArgs)
  BEGIN_PARAMETER()
    // Explicit standalone output binding at 222. This is NOT supplied by
    // bindCommonRayTracingResources; the CPU side binds it directly.
    RW_STRUCTURED_BUFFER(BINDING_WATER_FOAM_SHEET_VERTICES)
    // The foam density/height field, read-only, bound directly like the output above.
    TEXTURE2D(BINDING_WATER_FOAM_SURFACE_FIELD)
    TEXTURE2D(BINDING_WATER_FOAM_SURFACE_HEIGHT)
    STRUCTURED_BUFFER(225)
    STRUCTURED_BUFFER(226)
    STRUCTURED_BUFFER(227)
    STRUCTURED_BUFFER(228)
  END_PARAMETER()
};
}

void WaterFoamGpuGeometry::showImguiSettings() {
  RemixGui::Checkbox("Reuse paired foam vertex evaluations", &WaterFoamGeometryOptions::foamTrianglePairReuseObject());
  RemixGui::Checkbox("Validate paired foam vertices on GPU", &WaterFoamGeometryOptions::foamValidateTrianglePairReuseObject());
}

AxisAlignedBoundingBox WaterFoamGpuGeometry::placedSheetBox(const FrameParams& params) {
  const WaterFoamBox bounds = waterFoamPlacedSheetBox(params.zUp, params.gridOriginX,
    params.gridOriginY, params.waterHeightCentre, params.waterHeightHalfRange);
  AxisAlignedBoundingBox box {};
  box.minPos = Vector3(bounds.minX, bounds.minY, bounds.minZ);
  box.maxPos = Vector3(bounds.maxX, bounds.maxY, bounds.maxZ);
  return box;
}

void WaterFoamGpuGeometry::ensureResources() {
  m_vertexCount = WATER_FOAM_TOTAL_VERTICES;
  m_indexCount = WATER_FOAM_TOTAL_VERTICES;

  if (m_vertexBuffer != nullptr && m_previousVertexBuffer != nullptr && m_indexBuffer != nullptr) {
    return;
  }

  // DEVICE_LOCAL, and deliberately SOURCE-capable. Two separate reasons:
  //
  //  * TRANSFER_SRC must be present because the internal geometry path copies FROM
  //    these buffers: rtx_geometry_utils.cpp cacheIndexDataOnGPU and
  //    cacheVertexDataOnGPU take the input index/position buffers as the copyBuffer
  //    source when building the internal caches that the AS and shading read.
  //  * AS-build-input and device-address are needed because the acceleration
  //    structure reads these buffers directly as well.
  //
  // HOST_VISIBLE would not itself prevent GPU writes; the reason this module does not
  // use remixapi_CreateMesh is that the public path requires a CPU pointer and memcpys
  // it, and offers no update expression suited to a per-frame GPU-written surface.
  DxvkBufferCreateInfo vertexInfo {};
  vertexInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    | VK_BUFFER_USAGE_TRANSFER_DST_BIT
    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  // Stages/access cover the whole consumption chain, not just the compute write:
  // compute write -> transfer read (internal cache copy) -> AS read -> shading read.
  vertexInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT
    | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
    | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
  vertexInfo.access = VK_ACCESS_SHADER_WRITE_BIT
    | VK_ACCESS_TRANSFER_WRITE_BIT
    | VK_ACCESS_TRANSFER_READ_BIT
    | VK_ACCESS_SHADER_READ_BIT
    | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vertexInfo.size = align(size_t(m_vertexCount) * sizeof(WaterFoamGpuVertex), CACHE_LINE_SIZE);
  m_vertexBuffer = m_device->createBuffer(vertexInfo,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Water foam sheet vertices A");
  m_previousVertexBuffer = m_device->createBuffer(vertexInfo,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Water foam sheet vertices B");

  // The index buffer does not inherit the vertex buffer's access intent: it is never
  // shader-written, only uploaded once and then read by transfer and the AS.
  DxvkBufferCreateInfo indexInfo {};
  indexInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    | VK_BUFFER_USAGE_TRANSFER_DST_BIT
    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  indexInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT
    | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
    | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
  indexInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
  indexInfo.size = align(size_t(m_indexCount) * sizeof(uint32_t), CACHE_LINE_SIZE);
  m_indexBuffer = m_device->createBuffer(indexInfo,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Water foam sheet indices");

  // Small host-visible staging, used exactly once to seed the fixed topology.
  DxvkBufferCreateInfo stagingInfo {};
  stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  stagingInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
  stagingInfo.access = VK_ACCESS_TRANSFER_READ_BIT;
  stagingInfo.size = align(size_t(m_indexCount) * sizeof(uint32_t), CACHE_LINE_SIZE);
  m_indexStaging = m_device->createBuffer(stagingInfo,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    DxvkMemoryStats::Category::RTXBuffer, "Water foam sheet index staging");
}

bool WaterFoamGpuGeometry::ensureTopology(const Rc<DxvkContext>& ctx) {
  ensureResources();
  if (m_topologyReady) {
    return true;
  }
  if (m_indexBuffer == nullptr || m_indexStaging == nullptr) {
    return false;
  }

  // Fixed topology, built once. Vertices move every frame; the index buffer never
  // does, so the acceleration structure can take its update path as long as the
  // vertex count stays constant and the mesh is never re-registered.
  auto* dst = reinterpret_cast<uint32_t*>(m_indexStaging->mapPtr(0));
  if (dst == nullptr) {
    return false;
  }
  // Static index buffer over standalone triangles. Quad k owns vertices [6k, 6k+6), laid
  // out as the generator emits them: triangle 0 = corners (00,10,11), triangle 1 =
  // (00,11,01). Keeping the indices static is what lets an invalid triangle be collapsed
  // by coincident positions instead of by rewriting indices, so no index cache, hash or
  // topology resynchronisation is involved.
  for (uint32_t i = 0; i < m_indexCount; ++i) {
    dst[i] = i;
  }

  ctx->uploadBuffer(m_indexBuffer, dst, uint32_t(size_t(m_indexCount) * sizeof(uint32_t)));
  m_topologyReady = true;
  m_stats.indicesWritten = m_indexCount;
  return true;
}

void WaterFoamGpuGeometry::dispatchSheetUpdate(const Rc<DxvkContext>& ctx, const FrameParams& params,
                                                   const Resources::Resource& densityField,
                                                   const Resources::Resource& heightField,
                                                   const Rc<DxvkBuffer>& waterInput, const Rc<DxvkBuffer>& waterPositions,
                                                   const Rc<DxvkBuffer>& waterDepth, const Rc<DxvkBuffer>& waterVertices) {
  ensureResources();
  if (m_vertexBuffer == nullptr || m_previousVertexBuffer == nullptr) {
    return;
  }
  ScopedGpuProfileZone(ctx, "Water Foam Sheet Update");
  std::swap(m_vertexBuffer, m_previousVertexBuffer);
  // Reusing this buffer must wait for both current and previous-position readers.
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
      | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
      | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
      | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  WaterFoamSurfaceArgs args {};
  // Culling uses the current camera, not the previous density snapshot's view.
  std::memcpy(args.worldToClip, &params.worldToClip, sizeof(args.worldToClip));
  args.vertexCount = m_vertexCount;
  args.sheetSide = kWaterFoamSheetSide;
  args.sheetExtent = kWaterFoamSheetExtent;
  // Supplied here rather than read from the frame's raytrace constants, which are not
  // yet uploaded at this point in the frame.
  args.upAxisIsZ = params.zUp ? 1u : 0u;
  // The display grid is laid over the density domain the update pass owns, so the
  // foam line is generated on top of the density that produced it.
  args.gridOrigin[0] = params.gridOriginX;
  args.gridOrigin[1] = params.gridOriginY;
  args.gridSpacing[0] = params.gridSpacing;
  args.gridSpacing[1] = params.gridSpacing;
  args.coverageScale = params.coverageScale;
  args.surfaceOffset = params.surfaceOffset;
  // Time and edge fade: erosion and UV drift need the same clock the density update used,
  // passed explicitly because this pass cannot read cb.timeSinceStartSeconds.
  args.time = params.time;
  args.edgeFadeWidth = kWaterFoamEdgeFadeCells;
  args.cameraUp = params.cameraUp;
  args.textureScale = std::clamp(params.textureScale, 0.1f, 100.0f);

  ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
  ctx->pushConstants(0, sizeof(args), &args);
  ctx->bindResourceBuffer(BINDING_WATER_FOAM_SHEET_VERTICES, DxvkBufferSlice(m_vertexBuffer));
  // The density field itself, so coverage and height come from the real foam line.
  // Bound as a sampled texture: the generator only reads it, and it is the buffer the
  // update pass finished writing earlier this frame (the field is double buffered, so
  // this is not the one being written).
  ctx->bindResourceView(BINDING_WATER_FOAM_SURFACE_FIELD, densityField.view, nullptr);
  // Recorded authored water and static terrain; live ocean variants replace
  // the ocean carrier height before testing whether it still covers the terrain.
  ctx->bindResourceView(BINDING_WATER_FOAM_SURFACE_HEIGHT, heightField.view, nullptr);
  const bool liveWater = params.zUp && waterInput != nullptr && waterPositions != nullptr && waterDepth != nullptr && waterVertices != nullptr;
  ctx->bindResourceBuffer(225, liveWater ? DxvkBufferSlice(waterInput) : DxvkBufferSlice());
  ctx->bindResourceBuffer(226, liveWater ? DxvkBufferSlice(waterPositions) : DxvkBufferSlice());
  ctx->bindResourceBuffer(227, liveWater ? DxvkBufferSlice(waterDepth) : DxvkBufferSlice());
  ctx->bindResourceBuffer(228, liveWater ? DxvkBufferSlice(waterVertices) : DxvkBufferSlice());
  // Both variants keep the same inputs/output ABI, so toggling this option does not
  // reset the density history, geometry generation or previous-position buffer.
  const bool pairReuse = liveWater && WaterFoamGeometryOptions::foamTrianglePairReuse();
  if (pairReuse) {
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, params.frustumCulling
      ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live_culled_pair_reuse)
      : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live_pair_reuse));
  } else {
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, liveWater
      ? (params.frustumCulling
        ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live_culled)
        : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live))
      : (params.frustumCulling
        ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_culled)
        : WaterFoamSurfaceShader::getShader()));
  }
  // One thread per QUAD, not per vertex: each thread emits its quad's two standalone
  // triangles, so a triangle can be collapsed when its corners lack water.
  // Group count and the shader's [numthreads] must agree, or the dispatch silently covers
  // only part of the domain. Both come from the shared constants now.
  const uint32_t groups = kWaterFoamDispatchGroups;
  ctx->dispatch(groups, groups, 2u);
  if (WaterFoamGeometryOptions::foamValidateTrianglePairReuse()
      && m_device->getCurrentFrameId() % 600 == 0) {
    validateTrianglePairReuse(ctx, params, args, liveWater, pairReuse, groups);
  }
  m_stats.verticesWritten = m_vertexCount;
  m_stats.dispatchedThisFrame = true;
  // The vertex contents changed, so advance the content generation. The CPU owns this
  // counter and never reads the vertices back to derive it.
  ++m_contentGeneration;
}

void WaterFoamGpuGeometry::validateTrianglePairReuse(const Rc<DxvkContext>& ctx,
    const FrameParams& params, const WaterFoamSurfaceArgs& args,
    bool liveWater, bool selectedReuse, uint32_t groups) {
  // Called only for enabled diagnostics at a 600-frame boundary. Allocation and
  // the alternate dispatch must never enter the normal (disabled) path.
  const uint64_t frame = m_device->getCurrentFrameId();
  if (m_pairReuseValidationLastFrame == frame) {
    Logger::info(str::format("[Water foam pair comparison] status=skipped reason=duplicate-frame frame=", frame));
    return;
  }
  m_pairReuseValidationLastFrame = frame;
  if (!liveWater) {
    Logger::info(str::format("[Water foam pair comparison] status=skipped reason=live-water-unavailable frame=", frame,
      " live=", liveWater, " culling=", params.frustumCulling, " extent=", args.sheetSide, "x", args.sheetSide, "x2"));
    return;
  }
  if (m_pairReuseValidationPending != nullptr
      && m_pairReuseValidationPending->load(std::memory_order_acquire)) {
    Logger::warn(str::format("[Water foam pair comparison] status=skipped reason=previous-comparison-pending frame=", frame));
    return;
  }

  static_assert(sizeof(WaterFoamGpuVertex) == 9 * sizeof(uint32_t));
  constexpr VkDeviceSize kMaximumOutputBytes = 64ull * 1024ull * 1024ull;
  const VkDeviceSize vertices = m_vertexCount;
  // Bound before multiplying; each comparison has one device-local output and
  // three host-visible snapshots, at most 256 MiB total with only one in flight.
  if (vertices == 0 || vertices > kMaximumOutputBytes / sizeof(WaterFoamGpuVertex)
      || args.vertexCount != vertices || m_vertexBuffer == nullptr
      || m_vertexBuffer->info().size < vertices * sizeof(WaterFoamGpuVertex)) {
    Logger::err(str::format("[Water foam pair comparison] status=failed reason=invalid-size frame=", frame,
      " vertices=", vertices, " argsVertices=", args.vertexCount, " maxOutputBytes=", kMaximumOutputBytes));
    return;
  }
  const VkDeviceSize words = vertices * 9;
  const VkDeviceSize bytes = words * sizeof(uint32_t);
  const bool culling = params.frustumCulling;
  const uint32_t sheetSide = args.sheetSide;
  Rc<DxvkBuffer> alternate;
  Rc<DxvkBuffer> readback;
  Rc<sync::CallbackFence> completed;
  std::shared_ptr<std::atomic<bool>> pending;
  // Prepare every resource/callback before changing the selected output binding.
  try {
    DxvkBufferCreateInfo alternateInfo {};
    alternateInfo.size = bytes;
    alternateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    alternateInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    alternateInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    alternate = m_device->createBuffer(alternateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer, "Water foam pair comparison output");
    DxvkBufferCreateInfo readbackInfo {};
    readbackInfo.size = 3 * bytes;
    readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    readbackInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    readbackInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    readback = m_device->createBuffer(readbackInfo,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      DxvkMemoryStats::Category::RTXBuffer, "Water foam pair comparison readback");
    if (alternate == nullptr || readback == nullptr) {
      Logger::err(str::format("[Water foam pair comparison] status=failed reason=allocation-unavailable frame=", frame));
      return;
    }
    pending = std::make_shared<std::atomic<bool>>(true);
    completed = new sync::CallbackFence();
    // No this capture: geometry may be destroyed before this command list retires.
    // The callback owns the readback; command-list tracking owns both GPU sources.
    completed->setCallback(1, [readback, pending, vertices, words, bytes, frame,
        selectedReuse, liveWater, culling, sheetSide, groups] {
      const auto* values = static_cast<const uint32_t*>(readback->mapPtr(0));
      if (values == nullptr) {
        Logger::err(str::format("[Water foam pair comparison] status=failed reason=readback-unavailable frame=", frame,
          " selectedReuse=", selectedReuse, " vertices=", vertices, " words=", words,
          " live=", liveWater, " culling=", culling, " extent=", sheetSide, "x", sheetSide, "x2"));
        pending->store(false, std::memory_order_release);
        return;
      }
      VkDeviceSize nonFiniteWords = 0;
      VkDeviceSize repeatedMismatches = 0;
      VkDeviceSize variantMismatches = 0;
      VkDeviceSize mismatches = 0;
      VkDeviceSize mismatchVertices = 0;
      VkDeviceSize firstMismatch = words;
      VkDeviceSize lastMismatchVertex = vertices;
      for (VkDeviceSize i = 0; i < words; ++i) {
        // Complete interleaved vertex: position[0..2], normal[3..5], UV[6..7], packed color[8].
        // Compare integers so signed zero and NaN payload changes also count.
        if (i % 9 != 8) {
          for (VkDeviceSize snapshot = 0; snapshot < 3; ++snapshot) {
            if ((values[snapshot * words + i] & 0x7f800000u) == 0x7f800000u) { ++nonFiniteWords; }
          }
        }
        if (values[i] != values[words + i]) { ++repeatedMismatches; }
        if (values[words + i] != values[2 * words + i]) { ++variantMismatches; }
        if (values[i] != values[2 * words + i]) {
          ++mismatches;
          firstMismatch = std::min(firstMismatch, i);
          const VkDeviceSize vertex = i / 9;
          if (vertex != lastMismatchVertex) {
            ++mismatchVertices;
            lastMismatchVertex = vertex;
          }
        }
      }
      const auto message = str::format("[Water foam pair comparison] status=", (mismatches == 0 && repeatedMismatches == 0 && nonFiniteWords == 0) ? "passed" : "mismatch",
        " frame=", frame, " selectedReuse=", selectedReuse,
        " vertices=", vertices, " words=", words, " wordsPerVertex=9 bytesPerOutput=", bytes,
        " nonFiniteWords=", nonFiniteWords, " repeatedMismatches=", repeatedMismatches, " variantMismatches=", variantMismatches, " mismatches=", mismatches, " mismatchVertices=", mismatchVertices, " firstMismatch=", firstMismatch,
        " firstVertex=", firstMismatch < words ? firstMismatch / 9 : vertices,
        " firstWordInVertex=", firstMismatch < words ? firstMismatch % 9 : 9,
        " selectedWord=", firstMismatch < words ? values[firstMismatch] : 0u,
        " alternateWord=", firstMismatch < words ? values[2 * words + firstMismatch] : 0u,
        " live=", liveWater, " culling=", culling, " extent=", sheetSide, "x", sheetSide, "x2",
        " dispatch=", groups, "x", groups, "x2");
      if (mismatches == 0 && repeatedMismatches == 0 && nonFiniteWords == 0) {
        Logger::info(message);
      } else {
        Logger::err(message);
      }
      pending->store(false, std::memory_order_release);
    });
  } catch (const DxvkError& error) {
    Logger::err(str::format("[Water foam pair comparison] status=failed reason=prepare frame=", frame, " error=", error.message()));
    return;
  } catch (const std::exception& error) {
    Logger::err(str::format("[Water foam pair comparison] status=failed reason=prepare frame=", frame, " error=", error.what()));
    return;
  } catch (...) {
    Logger::err(str::format("[Water foam pair comparison] status=failed reason=prepare-unknown frame=", frame));
    return;
  }

  ScopedGpuProfileZone(ctx, "Water Foam Pair Validation");
  try {
    // Both dispatches are adjacent and use the same bound input resources, constant
    // values and grid. Only the output binding and paired/repeated variant change.
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx->pushConstants(0, sizeof(args), &args);
    ctx->bindResourceBuffer(BINDING_WATER_FOAM_SHEET_VERTICES, DxvkBufferSlice(alternate));
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(alternate);
    const auto bindVariant = [&](bool reuse) {
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, reuse
        ? (culling
          ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live_culled_pair_reuse)
          : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live_pair_reuse))
        : (culling
          ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live_culled)
          : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live)));
    };
    // Repeat the selected shader first to distinguish unstable inputs/readback from a variant difference.
    bindVariant(selectedReuse);
    ctx->dispatch(groups, groups, 2u);
    ctx->emitMemoryBarrier(0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    ctx->copyBuffer(readback, 0, m_vertexBuffer, 0, bytes);
    ctx->copyBuffer(readback, bytes, alternate, 0, bytes);
    bindVariant(!selectedReuse);
    ctx->dispatch(groups, groups, 2u);
    ctx->emitMemoryBarrier(0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    ctx->copyBuffer(readback, 2 * bytes, alternate, 0, bytes);
    ctx->emitMemoryBarrier(0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    ctx->signal(completed, 1);
    m_pairReuseValidationPending = pending;
    Logger::info(str::format("[Water foam pair comparison] status=submitted frame=", frame,
      " selectedReuse=", selectedReuse, " vertices=", vertices, " words=", words,
      " readbackBytes=", 3 * bytes, " live=", liveWater, " culling=", culling,
      " extent=", sheetSide, "x", sheetSide, "x2", " dispatch=", groups, "x", groups, "x2"));
  } catch (const DxvkError& error) {
    Logger::err(str::format("[Water foam pair comparison] status=failed reason=record frame=", frame, " error=", error.message()));
  } catch (const std::exception& error) {
    Logger::err(str::format("[Water foam pair comparison] status=failed reason=record frame=", frame, " error=", error.what()));
  } catch (...) {
    Logger::err(str::format("[Water foam pair comparison] status=failed reason=record-unknown frame=", frame));
  }
  // The actual scene always consumes the originally selected output, including
  // after a diagnostic failure. Do not swap buffers or advance geometry/history.
  ctx->bindResourceBuffer(BINDING_WATER_FOAM_SHEET_VERTICES, DxvkBufferSlice(m_vertexBuffer));
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, selectedReuse
    ? (culling
      ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live_culled_pair_reuse)
      : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live_pair_reuse))
    : (culling
      ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live_culled)
      : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamSurfaceShader, water_foam_surface_live)));
}


bool WaterFoamGpuGeometry::ownsVertexBuffer(const Rc<DxvkBuffer>& buffer) const {
  return buffer != nullptr && (buffer == m_vertexBuffer || buffer == m_previousVertexBuffer);
}

bool WaterFoamGpuGeometry::ownsGeometry(const RasterGeometry& geometry) const {
  if (!ready() || m_contentGeneration == 0 || geometry.vertexCount != m_vertexCount
      || geometry.indexCount != m_indexCount || geometry.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
      || geometry.hashes[HashComponents::VertexPosition]
        != (kWaterFoamSheetHashPosition ^ (m_contentGeneration * 0x9E3779B97F4A7C15ull))) {
    return false;
  }
  const DxvkBufferSlice vertices(m_vertexBuffer);
  const auto matchesAttribute = [&](const RasterBuffer& attribute, uint32_t offset, VkFormat format) {
    return attribute.matches(vertices) && attribute.offsetFromSlice() == offset
      && attribute.stride() == sizeof(WaterFoamGpuVertex) && attribute.vertexFormat() == format;
  };
  return matchesAttribute(geometry.positionBuffer, offsetof(WaterFoamGpuVertex, position), VK_FORMAT_R32G32B32_SFLOAT)
    && matchesAttribute(geometry.normalBuffer, offsetof(WaterFoamGpuVertex, normal), VK_FORMAT_R32G32B32_SFLOAT)
    && matchesAttribute(geometry.texcoordBuffer, offsetof(WaterFoamGpuVertex, texcoord), VK_FORMAT_R32G32_SFLOAT)
    && matchesAttribute(geometry.color0Buffer, offsetof(WaterFoamGpuVertex, color), VK_FORMAT_B8G8R8A8_UNORM)
    && geometry.indexBuffer.matches(DxvkBufferSlice(m_indexBuffer))
    && geometry.indexBuffer.offsetFromSlice() == 0 && geometry.indexBuffer.stride() == sizeof(uint32_t)
    && geometry.indexBuffer.indexType() == VK_INDEX_TYPE_UINT32;
}

std::shared_ptr<const RasterGeometry> WaterFoamGpuGeometry::buildGeometrySnapshot(remixapi_MaterialHandle material, const FrameParams& params, uint64_t waterTopologyHash) {
  ensureResources();
  const DxvkBufferSlice vertexSlice { m_vertexBuffer };
  const DxvkBufferSlice indexSlice { m_indexBuffer };

  auto geometry = std::make_shared<RasterGeometry>();
  geometry->externalMaterial = material;
  geometry->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  geometry->cullMode = VK_CULL_MODE_NONE;
  geometry->frontFace = VK_FRONT_FACE_CLOCKWISE;
  geometry->vertexCount = m_vertexCount;
  // Four views over the one interleaved vertex buffer, matching the contract header's
  // layout exactly.
  geometry->positionBuffer = RasterBuffer { vertexSlice, offsetof(WaterFoamGpuVertex, position), sizeof(WaterFoamGpuVertex), VK_FORMAT_R32G32B32_SFLOAT };
  geometry->normalBuffer   = RasterBuffer { vertexSlice, offsetof(WaterFoamGpuVertex, normal),   sizeof(WaterFoamGpuVertex), VK_FORMAT_R32G32B32_SFLOAT };
  geometry->texcoordBuffer = RasterBuffer { vertexSlice, offsetof(WaterFoamGpuVertex, texcoord), sizeof(WaterFoamGpuVertex), VK_FORMAT_R32G32_SFLOAT };
  geometry->color0Buffer   = RasterBuffer { vertexSlice, offsetof(WaterFoamGpuVertex, color),    sizeof(WaterFoamGpuVertex), VK_FORMAT_B8G8R8A8_UNORM };
  geometry->indexCount = m_indexCount;
  geometry->indexBuffer = RasterBuffer { indexSlice, 0, sizeof(uint32_t), VK_INDEX_TYPE_UINT32 };

  // Topological identity: constant, so this mesh keeps exactly one topological bucket
  // and neither the geometry cache nor the BLAS grows per frame.
  geometry->hashes[HashComponents::Indices] = kWaterFoamSheetHashIndices;
  geometry->hashes[HashComponents::GeometryDescriptor] = kWaterFoamSheetHashDescriptor;
  geometry->hashes[HashComponents::VertexLayout] = kWaterFoamSheetHashLayout;
  // World-space UV correspondence changes when the live patch wraps or changes LOD.
  geometry->hashes[HashComponents::VertexTexcoord] = kWaterFoamSheetHashTexcoord ^ waterTopologyHash;
  // Content identity: mixed with the per-submission generation, so a changed vertex
  // buffer is reported as changed content (VertexDataHash) without disturbing the
  // topological bucket above. This is what moves the object onto the BVH-update path
  // instead of letting it be reused with stale geometry.
  geometry->hashes[HashComponents::VertexPosition] = kWaterFoamSheetHashPosition ^ (m_contentGeneration * 0x9E3779B97F4A7C15ull);
  geometry->hashes.precombine();

  // The bounding box MUST be set here. Omitting it leaves the acceleration structure
  // with a default box, which is why fixing the box helper alone changed nothing that
  // the renderer could see. Derived from the same FrameParams the dispatch used, so the
  // box and the generated vertices cannot disagree about the up axis or the translation.
  const AxisAlignedBoundingBox box = placedSheetBox(params);
  geometry->boundingBox.minPos = box.minPos;
  geometry->boundingBox.maxPos = box.maxPos;
  return geometry;
}

bool WaterFoamGpuGeometry::tryIssueReadback(const Rc<DxvkContext>& ctx, const FrameParams& params, uint64_t frameIndex) {
  if (m_vertexBuffer == nullptr) {
    return false;
  }

  // Find a free slot. Slots are only freed by tryConsumeReadback once the fence confirms
  // completion, so a full set simply means skip this sample rather than stall.
  ReadbackSlot* slot = nullptr;
  for (uint32_t i = 0; i < kReadbackSlotCount; ++i) {
    if (!m_readbackSlots[i].pending) {
      slot = &m_readbackSlots[i];
      slot->sample.slot = i;
      break;
    }
  }
  if (slot == nullptr) {
    return false;
  }

  if (slot->buffer == nullptr) {
    DxvkBufferCreateInfo info {};
    // Destination of the copy, read by the host once the fence says the copy retired.
    // HOST_COHERENT is requested deliberately so no explicit invalidate is required; if
    // that ever changes, an invalidate must be added with it.
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    info.size = align(size_t(3) * sizeof(WaterFoamGpuVertex), CACHE_LINE_SIZE);
    slot->buffer = m_device->createBuffer(info,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      DxvkMemoryStats::Category::RTXBuffer, "Water foam readback");
    if (slot->buffer == nullptr) {
      return false;
    }
  }

  // Three vertices, first / middle / last, each 36 bytes, into slot offsets 0/36/72.
  for (uint32_t i = 0; i < 3; ++i) {
    const VkDeviceSize srcOffset = VkDeviceSize(kReadbackVertexIndices[i]) * sizeof(WaterFoamGpuVertex);
    ctx->copyBuffer(slot->buffer, VkDeviceSize(i) * sizeof(WaterFoamGpuVertex),
                    m_vertexBuffer, srcOffset, sizeof(WaterFoamGpuVertex));
  }

  // Make the transfer writes visible to the host before the fence can be observed as
  // signalled.
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

  const uint64_t target = ++m_readbackSignalValue;
  ctx->signal(m_readbackFence, target);

  slot->pending = true;
  slot->targetValue = target;
  // Record the parameters in effect for THIS dispatch, so the expected values can be
  // derived from the sample rather than from whatever the frame that happens to read it
  // is using.
  slot->sample.generation = m_contentGeneration;
  slot->sample.frame = frameIndex;
  slot->sample.zUp = params.zUp;
  slot->sample.translation[0] = params.gridOriginX;
  slot->sample.translation[1] = params.gridOriginY;
  slot->sample.translation[2] = params.gridSpacing;
  slot->sample.valid = false;
  return true;
}

bool WaterFoamGpuGeometry::tryConsumeReadback(ReadbackSample& out) {
  for (uint32_t i = 0; i < kReadbackSlotCount; ++i) {
    ReadbackSlot& slot = m_readbackSlots[i];
    if (!slot.pending) {
      continue;
    }
    // Non-blocking completion check. No wait(), no sleep, no waitIdle.
    if (m_readbackFence->value() < slot.targetValue) {
      continue;
    }

    const void* mapped = slot.buffer->mapPtr(0);
    if (mapped == nullptr) {
      // A failed map is a real diagnostic, not something to drop silently, but it is
      // reported at low frequency rather than every frame.
      if ((m_readbackMapFailures++ % 60u) == 0u) {
        Logger::warn(str::format("[WaterFoam] readback slot ", i,
          " map failed after fence value ", slot.targetValue,
          " (fence=", m_readbackFence->value(), ")"));
      }
      slot.pending = false;
      continue;
    }
    const auto* verts = reinterpret_cast<const WaterFoamGpuVertex*>(mapped);
    for (uint32_t v = 0; v < 3; ++v) {
      slot.sample.position[v][0] = verts[v].position[0];
      slot.sample.position[v][1] = verts[v].position[1];
      slot.sample.position[v][2] = verts[v].position[2];
      slot.sample.normal[v][0] = verts[v].normal[0];
      slot.sample.normal[v][1] = verts[v].normal[1];
      slot.sample.normal[v][2] = verts[v].normal[2];
      slot.sample.texcoord[v][0] = verts[v].texcoord[0];
      slot.sample.texcoord[v][1] = verts[v].texcoord[1];
      slot.sample.color[v] = verts[v].color;
    }
    slot.sample.valid = true;
    out = slot.sample;
    slot.pending = false;
    return true;
  }
  return false;
}

uint32_t WaterFoamGpuGeometry::pendingReadbackCount() const {
  uint32_t count = 0;
  for (uint32_t i = 0; i < kReadbackSlotCount; ++i) {
    if (m_readbackSlots[i].pending) {
      ++count;
    }
  }
  return count;
}

} // namespace dxvk
