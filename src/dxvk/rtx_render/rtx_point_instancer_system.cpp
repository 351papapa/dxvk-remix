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
#include "rtx_point_instancer_system.h"

#include "dxvk_device.h"
#include "rtx_render/rtx_shader_manager.h"
#include "dxvk_scoped_annotation.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_imgui.h"

#include <rtx_shaders/point_instancer_culling.h>
#include <rtx_shaders/point_instancer_culling_frustum.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dxvk {

  namespace {
    class PointInstancerCullingShader : public ManagedShader {
      SHADER_SOURCE(PointInstancerCullingShader, VK_SHADER_STAGE_COMPUTE_BIT, point_instancer_culling)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(POINT_INSTANCER_CULLING_BINDING_CONSTANTS)
        STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_TRANSFORMS_INPUT)
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_INSTANCE_BUFFER)
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_SURFACE_BUFFER)
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_MATERIAL_BUFFER)
        STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_BATCHES)
        STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_THREAD_BATCH)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PointInstancerCullingShader);
  }

  RtxPointInstancerSystem::RtxPointInstancerSystem(DxvkDevice* device)
    : CommonDeviceObject(device) { }

  void RtxPointInstancerSystem::showImguiSettings() {
    if (RemixGui::CollapsingHeader("Point Instancer Culling")) {
      ImGui::PushID("rtx_point_instancer");
      ImGui::Dummy({ 0, 2 });
      ImGui::Indent();

      RemixGui::Checkbox("Enable Culling", &enableObject());
      ImGui::BeginDisabled(!enable());
      RemixGui::Checkbox("Cull Outside Camera Frustum", &frustumCullObject());
      RemixGui::DragFloat("Frustum Margin", &frustumSpherePadObject(), 1.f, 0.f, 100000.f);
      RemixGui::DragFloat("Keep Nearby Off-screen Instances", &frustumNearSkipObject(), 1.f, 0.f, 100000.f);

      RemixGui::DragFloat("Culling Radius", &cullingRadiusObject(), 10.f, fadeStartRadius(), 100000.f, "%.0f");
      RemixGui::DragFloat("Fade Start Radius", &fadeStartRadiusObject(), 10.f, 0.f, cullingRadius(), "%.0f");

      ImGui::EndDisabled();

      RemixGui::Checkbox("Cull Host (API) Instancers", &cullHostInstancersObject());
      RemixGui::Checkbox("Single Dispatch For All Batches", &singleDispatchObject());
      RemixGui::Checkbox("Skip Unchanged Uploads", &skipUnchangedUploadsObject());
      ImGui::Unindent();
      ImGui::PopID();
    }
  }

  void RtxPointInstancerSystem::dispatchCulling(
      Rc<DxvkContext> ctx,
      const Rc<DxvkBuffer>& instanceBuffer,
      const Rc<DxvkBuffer>& surfaceBuffer,
      const Rc<DxvkBuffer>& surfaceMaterialBuffer,
      const std::vector<PointInstancerBatch>& batches,
      const Vector3& cameraPosition) {
    ScopedGpuProfileZone(ctx, "PointInstancerCulling");

    if (batches.empty()) {
      return;
    }

    const Rc<DxvkDevice>& dev = ctx->getDevice();

    // Allocate constant buffer (once)
    if (m_cb.ptr() == nullptr) {
      DxvkBufferCreateInfo info;
      info.usage  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
      info.size   = sizeof(PointInstancerCullingConstants);
      m_cb = dev->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               DxvkMemoryStats::Category::RTXBuffer,
                               "RTX PointInstancer - Constant Buffer");
    }

    // --- Build the frame's batch table, transform list and thread list (CPU) ---
    // With rtx.pointInstancer.singleDispatch all batches share one transform upload and one dispatch. Per-batch
    // uploads and dispatches need a barrier pair between each, and GPU-instanced content can arrive as thousands
    // of small batches.
    const size_t batchCount = batches.size();
    m_batchScratch.clear();
    m_batchScratch.reserve(batchCount);
    m_transformFirstScratch.resize(batchCount);

    const bool cullHost = cullHostInstancers();
    uint32_t threadCount = 0;
    uint32_t transformCount = 0;
    for (size_t i = 0; i < batchCount; ++i) {
      const PointInstancerBatch& batch = batches[i];
      // The SSS twin of a batch (recorded right after it by addPointInstancerBlas) shares its
      // transforms and surface slots: reuse the primary's transforms, and let only the
      // primary write the surface/material slots.
      const bool isTwin = i > 0 &&
        batches[i - 1].transforms == batch.transforms &&
        batches[i - 1].baseSurfaceIndex == batch.baseSurfaceIndex &&
        batches[i - 1].instanceCount == batch.instanceCount;
      if (isTwin) {
        m_transformFirstScratch[i] = m_transformFirstScratch[i - 1];
      } else {
        m_transformFirstScratch[i] = transformCount;
        transformCount += batch.instanceCount;
      }

      PointInstancerCullingBatch gpu {};
      gpu.objectToWorld        = batch.objectToWorld;
      gpu.prevObjectToWorld    = batch.prevObjectToWorld;
      gpu.baseSurfaceIndex     = batch.baseSurfaceIndex;
      gpu.instanceCount        = batch.instanceCount;
      gpu.customIndexFlags     = batch.customIndexFlags;
      gpu.instanceMask         = batch.instanceMask;
      gpu.sbtOffsetAndFlags    = batch.sbtOffsetAndFlags;
      gpu.blasRefLo            = static_cast<uint32_t>(batch.blasReference & 0xFFFFFFFFull);
      gpu.blasRefHi            = static_cast<uint32_t>(batch.blasReference >> 32);
      gpu.instanceBufferOffset = batch.instanceBufferByteOffset;
      gpu.firstThread          = threadCount;
      gpu.firstTransform       = m_transformFirstScratch[i];
      gpu.writeSurfaceData     = isTwin ? 0u : 1u;
      gpu.cullable             = (!batch.fromHost || cullHost) ? 1u : 0u;
      m_batchScratch.push_back(gpu);

      threadCount += batch.instanceCount;
    }

    if (threadCount == 0) {
      return;
    }

    // --- Upload what changed ---
    const bool skipUnchanged = skipUnchangedUploads();

    m_transformScratch.resize(transformCount);
    for (size_t i = 0; i < batchCount; ++i) {
      if (m_batchScratch[i].writeSurfaceData == 0) {
        continue; // twin: shares the primary's transforms
      }
      memcpy(&m_transformScratch[m_transformFirstScratch[i]], batches[i].transforms->data(),
             size_t(batches[i].instanceCount) * sizeof(Matrix4));
    }

    bool transformsUnchanged = skipUnchanged && m_transformsGpu.ptr() != nullptr &&
      m_uploadedTransforms.size() == transformCount &&
      memcmp(m_uploadedTransforms.data(), m_transformScratch.data(), size_t(transformCount) * sizeof(Matrix4)) == 0;

    const bool batchesUnchanged = skipUnchanged && m_batchesGpu.ptr() != nullptr &&
      m_uploadedThreadCount == threadCount &&
      m_uploadedBatches.size() == batchCount &&
      memcmp(m_uploadedBatches.data(), m_batchScratch.data(), batchCount * sizeof(PointInstancerCullingBatch)) == 0;

    const auto ensureBuffer = [&](Rc<DxvkBuffer>& buffer, size_t size, const char* name) {
      if (buffer.ptr() == nullptr || buffer->info().size < size) {
        DxvkBufferCreateInfo info;
        info.usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        info.size   = align(size + size / 4, 256); // headroom: the scene grows while streaming
        buffer = dev->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                   DxvkMemoryStats::Category::RTXBuffer, name);
        return true;
      }
      return false;
    };

    const size_t transformsSize = size_t(transformCount) * sizeof(Matrix4);
    if (ensureBuffer(m_transformsGpu, transformsSize, "RTX PointInstancer - Transforms Input")) {
      transformsUnchanged = false;
    }
    if (!transformsUnchanged) {
      ctx->writeToBuffer(m_transformsGpu, 0, transformsSize, m_transformScratch.data());
      m_uploadedTransforms.swap(m_transformScratch);
    }

    const size_t batchesSize = batchCount * sizeof(PointInstancerCullingBatch);
    const size_t threadBatchSize = size_t(threadCount) * sizeof(uint32_t);
    const bool batchRealloc = ensureBuffer(m_batchesGpu, batchesSize, "RTX PointInstancer - Batches");
    const bool threadRealloc = ensureBuffer(m_threadBatchGpu, threadBatchSize, "RTX PointInstancer - Thread Batch");
    if (!batchesUnchanged || batchRealloc || threadRealloc) {
      ctx->writeToBuffer(m_batchesGpu, 0, batchesSize, m_batchScratch.data());

      m_threadBatchScratch.resize(threadCount);
      for (size_t i = 0; i < batchCount; ++i) {
        std::fill_n(m_threadBatchScratch.begin() + m_batchScratch[i].firstThread,
                    m_batchScratch[i].instanceCount, static_cast<uint32_t>(i));
      }
      ctx->writeToBuffer(m_threadBatchGpu, 0, threadBatchSize, m_threadBatchScratch.data());

      m_uploadedBatches = m_batchScratch;
      m_uploadedThreadCount = threadCount;
    }

    // --- Dispatch ---
    // When culling is disabled, use FLT_MAX so every instance passes the distance test.
    const bool cullingEnabled = enable();
    PointInstancerCullingConstants constants {};
    constants.cameraPosition  = { cameraPosition.x, cameraPosition.y, cameraPosition.z };
    constants.cullingRadius   = cullingEnabled ? cullingRadius() : FLT_MAX;
    constants.fadeStartRadius = cullingEnabled ? fadeStartRadius() : 0.f;
    static_assert(sizeof(PointInstancerCullingConstants) == 112);
    constants.frustumCull = cullingEnabled && frustumCull();
    constants.frustumSpherePad = std::isfinite(frustumSpherePad()) ? std::max(0.f, frustumSpherePad()) : 0.f;
    constants.frustumNearSkip = std::isfinite(frustumNearSkip()) ? std::max(0.f, frustumNearSkip()) : 0.f;
    if (constants.frustumCull) {
      // Camera metadata only; every instance's transform and visibility are evaluated in the compute shader.
      const auto& camera = ctx->getCommonObjects()->getSceneManager().getCamera();
      const Matrix4 worldToClip = camera.getViewToProjectionf() * camera.getWorldToViewf();
      const Vector4 rowW(worldToClip[0][3], worldToClip[1][3], worldToClip[2][3], worldToClip[3][3]);
      for (uint32_t axis = 0; axis < 2; ++axis) {
        const Vector4 row(worldToClip[0][axis], worldToClip[1][axis], worldToClip[2][axis], worldToClip[3][axis]);
        for (uint32_t side = 0; side < 2; ++side) {
          Vector4 plane = side == 0 ? rowW + row : rowW - row;
          const float length = std::sqrt(dot(plane.xyz(), plane.xyz()));
          if (std::isfinite(length) && length > 1e-6f && std::isfinite(plane.w)) { plane /= length; }
          else { plane = Vector4(0.f, 0.f, 0.f, 1.f); }
          constants.viewPlanes[axis * 2 + side] = { plane.x, plane.y, plane.z, plane.w };
        }
      }
    }

    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_TRANSFORMS_INPUT,
                            DxvkBufferSlice(m_transformsGpu, 0, std::max<size_t>(transformsSize, sizeof(Matrix4))));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_BATCHES, DxvkBufferSlice(m_batchesGpu, 0, batchesSize));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_THREAD_BATCH, DxvkBufferSlice(m_threadBatchGpu, 0, threadBatchSize));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_INSTANCE_BUFFER, DxvkBufferSlice(instanceBuffer));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_SURFACE_BUFFER, DxvkBufferSlice(surfaceBuffer));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_MATERIAL_BUFFER, DxvkBufferSlice(surfaceMaterialBuffer));
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, constants.frustumCull
      ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, PointInstancerCullingShader, point_instancer_culling_frustum)
      : PointInstancerCullingShader::getShader());

    const auto dispatchRange = [&](uint32_t firstThread, uint32_t count) {
      constants.firstThread = firstThread;
      constants.threadCount = count;

      const DxvkBufferSliceHandle cSlice = m_cb->allocSlice();
      ctx->invalidateBuffer(m_cb, cSlice);
      ctx->writeToBuffer(m_cb, 0, sizeof(PointInstancerCullingConstants), &constants);
      ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_CONSTANTS, DxvkBufferSlice(m_cb));

      const VkExtent3D workgroups = util::computeBlockCount(
        VkExtent3D { count, 1, 1 },
        VkExtent3D { 64, 1, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    };

    if (singleDispatch()) {
      dispatchRange(0, threadCount);
    } else {
      // One dispatch per batch (same data, same output).
      for (const PointInstancerCullingBatch& batch : m_batchScratch) {
        if (batch.instanceCount != 0) {
          dispatchRange(batch.firstThread, batch.instanceCount);
        }
      }
    }
  }
}
