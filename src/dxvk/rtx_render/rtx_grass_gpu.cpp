#include "rtx_grass_gpu.h"
#include "rtx_point_instancer_system.h"
#include "rtx_shader_manager.h"
#include "dxvk_context.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include <rtx_shaders/grass_init.h>
#include <rtx_shaders/grass_cull.h>
#include <rtx_shaders/grass_histogram.h>
#include <rtx_shaders/grass_radix.h>
#include <rtx_shaders/grass_count.h>
#include <rtx_shaders/grass_prefix.h>
#include <rtx_shaders/grass_scatter.h>
#include <rtx_shaders/grass_emit.h>

namespace dxvk {
namespace {
  class GrassInitShader : public ManagedShader {
    SHADER_SOURCE(GrassInitShader, VK_SHADER_STAGE_COMPUTE_BIT, grass_init)
    BEGIN_PARAMETER()
      CONSTANT_BUFFER(GRASS_CONSTANTS)
      RW_STRUCTURED_BUFFER(GRASS_STATE)
      RW_STRUCTURED_BUFFER(GRASS_SELECTED)
    END_PARAMETER()
  };
  class GrassCullShader : public ManagedShader {
    SHADER_SOURCE(GrassCullShader, VK_SHADER_STAGE_COMPUTE_BIT, grass_cull)
    BEGIN_PARAMETER()
      CONSTANT_BUFFER(GRASS_CONSTANTS)
      STRUCTURED_BUFFER(GRASS_ROOTS)
      RW_STRUCTURED_BUFFER(GRASS_CANDIDATES)
      RW_STRUCTURED_BUFFER(GRASS_STATE)
    END_PARAMETER()
  };
  class GrassHistogramShader : public ManagedShader {
    SHADER_SOURCE(GrassHistogramShader, VK_SHADER_STAGE_COMPUTE_BIT, grass_histogram)
    BEGIN_PARAMETER()
      CONSTANT_BUFFER(GRASS_CONSTANTS)
      RW_STRUCTURED_BUFFER(GRASS_CANDIDATES)
      RW_STRUCTURED_BUFFER(GRASS_STATE)
    END_PARAMETER()
  };
  class GrassRadixShader : public ManagedShader {
    SHADER_SOURCE(GrassRadixShader, VK_SHADER_STAGE_COMPUTE_BIT, grass_radix)
    BEGIN_PARAMETER()
      CONSTANT_BUFFER(GRASS_CONSTANTS)
      RW_STRUCTURED_BUFFER(GRASS_STATE)
    END_PARAMETER()
  };
  class GrassCountShader : public ManagedShader {
    SHADER_SOURCE(GrassCountShader, VK_SHADER_STAGE_COMPUTE_BIT, grass_count)
    BEGIN_PARAMETER()
      CONSTANT_BUFFER(GRASS_CONSTANTS)
      RW_STRUCTURED_BUFFER(GRASS_CANDIDATES)
      RW_STRUCTURED_BUFFER(GRASS_STATE)
      RW_STRUCTURED_BUFFER(GRASS_GROUPS)
    END_PARAMETER()
  };
  class GrassPrefixShader : public ManagedShader {
    SHADER_SOURCE(GrassPrefixShader, VK_SHADER_STAGE_COMPUTE_BIT, grass_prefix)
    BEGIN_PARAMETER()
      CONSTANT_BUFFER(GRASS_CONSTANTS)
      RW_STRUCTURED_BUFFER(GRASS_STATE)
      RW_STRUCTURED_BUFFER(GRASS_GROUPS)
    END_PARAMETER()
  };
  class GrassScatterShader : public ManagedShader {
    SHADER_SOURCE(GrassScatterShader, VK_SHADER_STAGE_COMPUTE_BIT, grass_scatter)
    BEGIN_PARAMETER()
      CONSTANT_BUFFER(GRASS_CONSTANTS)
      STRUCTURED_BUFFER(GRASS_ROOTS)
      RW_STRUCTURED_BUFFER(GRASS_CANDIDATES)
      RW_STRUCTURED_BUFFER(GRASS_STATE)
      RW_STRUCTURED_BUFFER(GRASS_GROUPS)
      RW_STRUCTURED_BUFFER(GRASS_SELECTED)
      STRUCTURED_BUFFER(GRASS_PREVIOUS)
    END_PARAMETER()
  };
  class GrassEmitShader : public ManagedShader {
    SHADER_SOURCE(GrassEmitShader, VK_SHADER_STAGE_COMPUTE_BIT, grass_emit)
    BEGIN_PARAMETER()
      CONSTANT_BUFFER(GRASS_CONSTANTS)
      STRUCTURED_BUFFER(GRASS_SELECTED)
      RW_STRUCTURED_BUFFER(GRASS_INSTANCES)
      RW_STRUCTURED_BUFFER(GRASS_SURFACES)
      RW_STRUCTURED_BUFFER(GRASS_MATERIALS)
      RW_STRUCTURED_BUFFER(GRASS_MAPPING)
    END_PARAMETER()
  };
}

static_assert(sizeof(GrassRoot) == 48);
static_assert(sizeof(GrassSelection) == 144);

void GrassGpu::setRoots(std::vector<GrassRoot>&& roots) {
  m_roots = std::move(roots);
  m_rootsDirty = true;
}

void GrassGpu::clear() {
  m_roots.clear();
  m_rootsDirty = false;
  m_hasHistory = false;
  m_previousBatches.clear();
}

void GrassGpu::dispatch(const Rc<DxvkContext>& ctx, const Rc<DxvkBuffer>& instances,
    const Rc<DxvkBuffer>& surfaces, const Rc<DxvkBuffer>& materials,
    const Rc<DxvkBuffer>& mapping, const std::vector<PointInstancerBatch>& batches) {
  if (m_roots.empty() || !m_constants.budget ||
      std::none_of(batches.begin(), batches.end(), [](const auto& batch) { return batch.proceduralGrass; })) { return; }
  ScopedGpuProfileZone(ctx, "GrassGpu");
  const uint32_t frame = ctx->getDevice()->getCurrentFrameId();
  const bool history = m_hasHistory && frame == m_lastFrame + 1;
  auto ensure = [&](Rc<DxvkBuffer>& buffer, size_t size, const char* name, bool uniform = false) {
    if (buffer != nullptr && buffer->info().size >= size) { return false; }
    DxvkBufferCreateInfo info {};
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | (uniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    // Uniform reads have a separate access scope from storage-buffer shader reads.
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT | (uniform ? VK_ACCESS_UNIFORM_READ_BIT
      : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    info.size = align(size, 256);
    buffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, name);
    return true;
  };
  GrassConstants constants = m_constants;
  constants.rootCount = static_cast<uint32_t>(m_roots.size());
  constants.groupCount = (constants.rootCount + GRASS_GROUP_SIZE - 1) / GRASS_GROUP_SIZE;
  constants.previousBudget = history ? m_previousBudget : 0;
  ensure(m_constantsBuffer, sizeof(constants), "Grass constants", true);
  ensure(m_rootBuffer, m_roots.size() * sizeof(GrassRoot), "Grass roots");
  if (m_rootsDirty) {
    ctx->writeToBuffer(m_rootBuffer, 0, m_roots.size() * sizeof(GrassRoot), m_roots.data());
    m_rootsDirty = false;
  }
  ensure(m_candidates, m_roots.size() * 8, "Grass candidates");
  ensure(m_state, 260 * sizeof(uint32_t), "Grass selection state");
  ensure(m_groups, constants.groupCount * 8, "Grass group counts");
  std::swap(m_selected, m_previous);
  ensure(m_selected, constants.budget * sizeof(GrassSelection), "Grass selection");
  if (ensure(m_previous, std::max(1u, constants.previousBudget) * sizeof(GrassSelection), "Grass previous selection")) {
    constants.previousBudget = 0;
  }
  auto bindConstants = [&] {
    ctx->invalidateBuffer(m_constantsBuffer, m_constantsBuffer->allocSlice());
    ctx->writeToBuffer(m_constantsBuffer, 0, sizeof(constants), &constants);
    ctx->bindResourceBuffer(GRASS_CONSTANTS, DxvkBufferSlice(m_constantsBuffer));
  };
  auto barrier = [&] {
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  };
  ctx->bindResourceBuffer(GRASS_ROOTS, DxvkBufferSlice(m_rootBuffer));
  ctx->bindResourceBuffer(GRASS_CANDIDATES, DxvkBufferSlice(m_candidates));
  ctx->bindResourceBuffer(GRASS_STATE, DxvkBufferSlice(m_state));
  ctx->bindResourceBuffer(GRASS_GROUPS, DxvkBufferSlice(m_groups));
  ctx->bindResourceBuffer(GRASS_SELECTED, DxvkBufferSlice(m_selected));
  ctx->bindResourceBuffer(GRASS_PREVIOUS, DxvkBufferSlice(m_previous));
  constants.radixShift = 24;
  bindConstants();
  barrier();
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GrassInitShader::getShader());
  ctx->dispatch((std::max(256u, constants.budget) + 255) / 256, 1, 1);
  barrier();
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GrassCullShader::getShader());
  ctx->dispatch(constants.groupCount, 1, 1);
  for (int shift = 24; shift >= 0; shift -= 8) {
    constants.radixShift = static_cast<uint32_t>(shift);
    bindConstants();
    if (shift != 24) {
      barrier();
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GrassHistogramShader::getShader());
      ctx->dispatch(constants.groupCount, 1, 1);
    }
    barrier();
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GrassRadixShader::getShader());
    ctx->dispatch(1, 1, 1);
  }
  barrier();
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GrassCountShader::getShader());
  ctx->dispatch(constants.groupCount, 1, 1);
  barrier();
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GrassPrefixShader::getShader());
  ctx->dispatch(1, 1, 1);
  barrier();
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GrassScatterShader::getShader());
  ctx->dispatch(constants.groupCount, 1, 1);
  barrier();

  ctx->bindResourceBuffer(GRASS_INSTANCES, DxvkBufferSlice(instances));
  ctx->bindResourceBuffer(GRASS_SURFACES, DxvkBufferSlice(surfaces));
  ctx->bindResourceBuffer(GRASS_MATERIALS, DxvkBufferSlice(materials));
  ctx->bindResourceBuffer(GRASS_MAPPING, DxvkBufferSlice(mapping));
  for (const auto& batch : batches) {
    if (!batch.proceduralGrass) { continue; }
    constants.budget = batch.instanceCount;
    memcpy(&constants.prototypeTransform, &batch.objectToWorld, sizeof(mat4));
    constants.templateSurface = batch.baseSurfaceIndex;
    constants.outputSurface = batch.baseSurfaceIndex + 1;
    const auto old = m_previousBatches.find(batch.grassBatchId);
    constants.previousSurface = history && old != m_previousBatches.end() && old->second.frame == frame - 1
      ? old->second.surface : GRASS_INVALID;
    constants.previousInstanceCount = constants.previousSurface != GRASS_INVALID ? old->second.count : 0;
    constants.instanceOffset = batch.instanceBufferByteOffset;
    constants.customIndexFlags = batch.customIndexFlags;
    constants.instanceMask = batch.instanceMask;
    constants.sbtOffsetAndFlags = batch.sbtOffsetAndFlags;
    constants.blasLo = static_cast<uint32_t>(batch.blasReference);
    constants.blasHi = static_cast<uint32_t>(batch.blasReference >> 32);
    bindConstants();
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GrassEmitShader::getShader());
    ctx->dispatch((batch.instanceCount + 63) / 64, 1, 1);
    barrier();
  }
  for (const auto& batch : batches) {
    if (batch.proceduralGrass) { m_previousBatches[batch.grassBatchId] = { frame, batch.baseSurfaceIndex + 1, batch.instanceCount }; }
  }
  for (auto it = m_previousBatches.begin(); it != m_previousBatches.end();) {
    if (it->second.frame != frame) { it = m_previousBatches.erase(it); } else { ++it; }
  }
  m_previousBudget = m_constants.budget;
  m_lastFrame = frame;
  m_hasHistory = true;
}
}
