#include "rtx_water_simulation.h"
#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_asset_replacer.h"
#include "rtx_shader_manager.h"
#include "dxvk_scoped_annotation.h"
#include "rtx/pass/water_wave_args.h"
#include "rtx/pass/water_foam_follow_args.h"
#include "rtx/pass/water_flow_args.h"
#include "rtx_water.h"
#include <rtx_shaders/water_flow.h>
#include "rtx/pass/water_hull_args.h"
#include <rtx_shaders/water_hull.h>
#include <rtx_shaders/water_wave.h>
#include <rtx_shaders/water_foam_follow.h>
#include <Windows.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace dxvk {
namespace {
namespace sim = remixapi::water::simulation;
constexpr uint32_t kWaterLevelSide = 128;
constexpr uint32_t kWaterLevelHeader = 4;
constexpr uint32_t kWaterLevelCount = kWaterLevelSide * kWaterLevelSide;

bool sampleWaterLevel(const std::vector<float>& map, const Vector3& position, float& height) {
  if (map.size() != kWaterLevelHeader + kWaterLevelCount || !std::isfinite(map[0])
      || !std::isfinite(map[1]) || !std::isfinite(map[2]) || map[2] <= 0.0f) {
    return false;
  }
  const float x = (position.x - map[0]) / map[2];
  const float y = (position.y - map[1]) / map[2];
  if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0f || y < 0.0f
      || x >= float(kWaterLevelSide) || y >= float(kWaterLevelSide)) {
    return false;
  }
  const uint32_t index = uint32_t(y) * kWaterLevelSide + uint32_t(x);
  const float level = map[kWaterLevelHeader + index];
  if (!std::isfinite(level) || level <= -10000.0f) {
    return false;
  }
  height = level;
  return true;
}

constexpr uint64_t kFoamMesh = 0x574C475055464F41ull;
constexpr uint64_t kFoamMaterial = 0x574C4649454C4432ull;
static_assert(sizeof(WaterFlowArgs) == 68 && sizeof(WaterHullArgs) == 60);
static_assert(sizeof(sim::Impulse) == sizeof(WaterWaveImpulse));
static_assert(sizeof(sim::FlowForce) == sizeof(WaterHullForce));
class WaterHullShader : public ManagedShader {
  SHADER_SOURCE(WaterHullShader, VK_SHADER_STAGE_COMPUTE_BIT, water_hull)
  PUSH_CONSTANTS(WaterHullArgs)
  BEGIN_PARAMETER()
    STRUCTURED_BUFFER(222)
    RW_STRUCTURED_BUFFER(223)
    STRUCTURED_BUFFER(224)
    STRUCTURED_BUFFER(225)
    RW_STRUCTURED_BUFFER(226)
  END_PARAMETER()
};
static_assert(sizeof(sim::FlowSource) == sizeof(WaterFlowSource) && sizeof(sim::FlowForce) == sizeof(WaterFlowForce));
class WaterFlowShader : public ManagedShader {
  SHADER_SOURCE(WaterFlowShader, VK_SHADER_STAGE_COMPUTE_BIT, water_flow)
  PUSH_CONSTANTS(WaterFlowArgs)
  BEGIN_PARAMETER()
    STRUCTURED_BUFFER(FLOW_STATE_SOURCE)
    RW_STRUCTURED_BUFFER(FLOW_STATE_TARGET)
    STRUCTURED_BUFFER(FLOW_PRESSURE_SOURCE)
    RW_STRUCTURED_BUFFER(FLOW_PRESSURE_TARGET)
    RW_STRUCTURED_BUFFER(FLOW_DIVERGENCE)
    STRUCTURED_BUFFER(FLOW_SOURCES)
    RW_STRUCTURED_BUFFER(FLOW_VERTICES)
    STRUCTURED_BUFFER(FLOW_WATER_MAP)
    STRUCTURED_BUFFER(FLOW_FORCES)
    RW_STRUCTURED_BUFFER(FLOW_INDICES)
  END_PARAMETER()
};
class WaterWaveShader : public ManagedShader {
  SHADER_SOURCE(WaterWaveShader, VK_SHADER_STAGE_COMPUTE_BIT, water_wave)
  PUSH_CONSTANTS(WaterWaveArgs)
  BEGIN_PARAMETER()
    STRUCTURED_BUFFER(WATER_WAVE_SOURCE)
    RW_STRUCTURED_BUFFER(WATER_WAVE_TARGET)
    STRUCTURED_BUFFER(WATER_WAVE_MASK)
    STRUCTURED_BUFFER(WATER_WAVE_IMPULSES)
  END_PARAMETER()
};
class WaterFoamFollowShader : public ManagedShader {
  SHADER_SOURCE(WaterFoamFollowShader, VK_SHADER_STAGE_COMPUTE_BIT, water_foam_follow)
  PUSH_CONSTANTS(WaterFoamFollowArgs)
  BEGIN_PARAMETER()
    STRUCTURED_BUFFER(WATER_FOAM_FOLLOW_INPUT)
    STRUCTURED_BUFFER(WATER_FOAM_FOLLOW_POSITIONS)
    STRUCTURED_BUFFER(WATER_FOAM_FOLLOW_SOURCE)
    RW_STRUCTURED_BUFFER(WATER_FOAM_FOLLOW_TARGET)
  END_PARAMETER()
};
}
WaterGpuSimulation::~WaterGpuSimulation() {
  if (m_shared) { UnmapViewOfFile(m_shared); }
  if (m_mapping) { CloseHandle(m_mapping); }
  if (m_mutex) { CloseHandle(m_mutex); }
}
void WaterGpuSimulation::ensureResources() {
  if (m_state[0] != nullptr) { return; }
  auto create = [&](size_t bytes, const char* name) {
    DxvkBufferCreateInfo info {};
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
      | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
      | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    info.size = align(bytes, CACHE_LINE_SIZE);
    return m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, name);
  };
  m_state[0] = create(WATER_WAVE_COUNT * 8, "Water interaction state A");
  m_state[1] = create(WATER_WAVE_COUNT * 8, "Water interaction state B");
  m_mask = create(WATER_WAVE_COUNT * 8, "Water interaction mask");
  m_impulses = create(sim::kMaxImpulses * sizeof(sim::Impulse), "Water interaction events");
  m_foamSource = create(sim::kFoamVertices * sizeof(sim::Vertex), "Water foam flat vertices");
  m_foamTarget = create(sim::kFoamVertices * sizeof(sim::Vertex), "Water foam displaced vertices");
  m_foamIndices = create(sim::kFoamIndices * sizeof(uint32_t), "Water foam indices");
}
bool WaterGpuSimulation::waterLevelAt(const Vector3& position, float& height) const {
  return waterLevelBuffer() != nullptr && sampleWaterLevel(m_waterLevelMap, position, height);
}

Rc<DxvkBuffer> WaterGpuSimulation::waterLevelBuffer() const {
  if (!m_waterLevelReady || GetTickCount() - m_waterLevelTick >= 500u) {
    return nullptr;
  }
  return m_waterLevelBuffer;
}

bool WaterGpuSimulation::prepare(const Rc<DxvkContext>& ctx, bool enabled, remixapi::water::gpu::Frame& frame) {
  ScopedCpuProfileZone();
  auto* assets = ctx->getCommonObjects()->getSceneManager().getAssetReplacer().get();
  enabled = enabled && assets->accessExternalMaterial(reinterpret_cast<remixapi_MaterialHandle>(kFoamMaterial)) != nullptr;
  if (!m_mutex) { m_mutex = OpenMutexW(SYNCHRONIZE | MUTEX_MODIFY_STATE, FALSE, sim::kMutexName); }
  if (!m_mapping) { m_mapping = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, sim::kMappingName); }
  if (m_mapping && !m_shared) {
    m_shared = static_cast<sim::Shared*>(MapViewOfFile(m_mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sim::Shared)));
  }
  if (enabled && m_shared) { ensureResources(); }
  std::vector<sim::Packet> packets;
  std::vector<sim::Vertex> vertices;
  std::vector<uint32_t> indices;
  std::vector<float> waterMap;
  bool copied = false, active = false, foamChanged = false;
  uint32_t epoch = m_epoch;
  if (m_shared && m_mutex) {
    const DWORD lock = WaitForSingleObject(m_mutex, 0);
    if (lock == WAIT_OBJECT_0 || lock == WAIT_ABANDONED) {
      if (m_shared->version == sim::kVersion && m_shared->bytes == sizeof(sim::Shared)) {
        m_shared->ready = enabled ? 1u : 0u;
        m_shared->readyTick = GetTickCount();
        m_shared->flowReady = enabled ? (WaterSystem::localFlow()?sim::kFoamFlow:0) | (WaterSystem::hullWaves()?sim::kHullWater:0) : 0;
        // Optical water levels remain available when the interaction solvers are disabled.
        const bool mapReady = m_shared->waterMapReady && GetTickCount() - m_shared->tick < 500u;
        if (mapReady && (!m_waterLevelReady || m_waterLevelEpoch != m_shared->epoch)) {
          waterMap.assign(m_shared->waterMap, m_shared->waterMap + kWaterLevelCount);
        }
        m_waterLevelReady = mapReady;
        m_waterLevelTick = m_shared->tick;
        m_waterLevelEpoch = m_shared->epoch;
        copied = true;
        active = enabled && m_shared->active && GetTickCount()-m_shared->tick < 500;
        epoch = m_shared->epoch;
        const uint32_t count = m_shared->writeSequence - m_shared->readSequence;
        if (active && count <= sim::kQueueSize) {
          packets.reserve(count);
          for (uint32_t i=0;i<count;++i) { packets.push_back(m_shared->packets[(m_shared->readSequence+i)%sim::kQueueSize]); }
          m_shared->readSequence = m_shared->writeSequence;
          const bool flowPacket = WaterSystem::localFlow() && ((!packets.empty() && (packets.back().flow.enabled&sim::kFoamFlow)) || (packets.empty() && m_flowActive));
          foamChanged = !flowPacket && (m_foamSequence != m_shared->foamSequence || !m_valid || m_flowMesh);
          if (foamChanged && m_shared->foamVertexCount <= sim::kFoamVertices && m_shared->foamIndexCount <= sim::kFoamIndices) {
            vertices.assign(m_shared->foamVertices, m_shared->foamVertices + m_shared->foamVertexCount);
            indices.assign(m_shared->foamIndices, m_shared->foamIndices + m_shared->foamIndexCount);
            m_foamSequence = m_shared->foamSequence;
          }
        } else if (count > sim::kQueueSize) { active = false; }
      } else {
        m_waterLevelReady = false;
      }
      ReleaseMutex(m_mutex);
    }
  }
  if (!waterMap.empty()) {
    DxvkBufferCreateInfo info {};
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    info.size = waterMap.size() * sizeof(float);
    m_waterMap = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Local flow native water map");
    ctx->uploadBuffer(m_waterMap, waterMap.data(), uint32_t(info.size));
    m_waterLevelMap.resize(kWaterLevelHeader + kWaterLevelCount);
    std::copy(waterMap.begin(), waterMap.end(), m_waterLevelMap.begin() + kWaterLevelHeader);
    m_waterMin = 0.0f;
    m_waterMax = 0.0f;
    for (float z : waterMap) { if (std::isfinite(z) && z > -10000) { m_waterMin = (std::min)(m_waterMin,z); m_waterMax = (std::max)(m_waterMax,z); } }
    m_mapUploaded = true;
  }
  const auto mapOrigin = WaterSystem::waterMapOrigin();
  const float mapSpacing = WaterSystem::waterMapCellSize();
  const float edgeFade = (std::min)(frame.data[3] * 0.5f, mapSpacing * 0.5f);
  if (!std::isfinite(mapOrigin.x) || !std::isfinite(mapOrigin.y) || !std::isfinite(mapSpacing)
      || mapSpacing <= 0.0f || !std::isfinite(edgeFade) || edgeFade <= 0.0f) {
    m_waterLevelBuffer = nullptr;
  } else if (m_waterLevelReady && m_waterLevelMap.size() == kWaterLevelHeader + kWaterLevelCount) {
    const bool changed = !waterMap.empty() || m_waterLevelBuffer == nullptr || m_waterLevelMap[0] != mapOrigin.x
      || m_waterLevelMap[1] != mapOrigin.y || m_waterLevelMap[2] != mapSpacing || m_waterLevelMap[3] != edgeFade;
    if (changed) {
      m_waterLevelMap[0] = mapOrigin.x;
      m_waterLevelMap[1] = mapOrigin.y;
      m_waterLevelMap[2] = mapSpacing;
      m_waterLevelMap[3] = edgeFade;
      DxvkBufferCreateInfo info {};
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
      info.size = m_waterLevelMap.size() * sizeof(float);
      m_waterLevelBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "Water optical level map");
      ctx->uploadBuffer(m_waterLevelBuffer, m_waterLevelMap.data(), uint32_t(info.size));
    }
  }
  if (!enabled || (copied && !active) || (GetTickCount()-m_surface.tick >= 500 && packets.empty())) {
    m_valid = false;
    m_flowActive = false;
    m_hullActive = false;
    return false;
  }
  if (!WaterSystem::localFlow()) { m_flowActive = false; }
  if (!WaterSystem::hullWaves()) { m_hullActive = false; }
  if (!packets.empty()) {
    ScopedGpuProfileZone(ctx, "Water Interaction Solver");
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    // Packets reuse GPU inputs, so each upload must precede its own dispatch in the execution queue.
    for (const auto& packet : packets) {
      if (!std::isfinite(packet.spacing) || packet.spacing < .5f || packet.spacing > 4.f
          || packet.impulseCount > sim::kMaxImpulses || packet.steps > 12) { m_valid = false; return false; }
      WaterWaveArgs args {};
      args.spacing = packet.spacing; args.stepSeconds = packet.stepSeconds;
      args.speed = packet.speed; args.damping = packet.damping; args.maxHeight = packet.maxHeight; args.circular = packet.circular;
      auto dispatch = [&](uint32_t operation) {
        args.operation = operation;
        ctx->bindResourceBuffer(WATER_WAVE_SOURCE, DxvkBufferSlice(m_state[m_parity]));
        ctx->bindResourceBuffer(WATER_WAVE_TARGET, DxvkBufferSlice(m_state[m_parity^1]));
        ctx->bindResourceBuffer(WATER_WAVE_MASK, DxvkBufferSlice(m_mask));
        ctx->bindResourceBuffer(WATER_WAVE_IMPULSES, DxvkBufferSlice(m_impulses));
        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, WaterWaveShader::getShader());
        ctx->pushConstants(0, sizeof(args), &args);
        ctx->dispatch((WATER_WAVE_COUNT+63)/64, 1, 1);
        m_parity ^= 1;
      };
      if (!m_valid || epoch != m_epoch || packet.resetSerial != m_resetSerial || packet.spacing != m_spacing) {
        m_parity = 0;
        ctx->writeToBuffer(m_state[0], 0, sizeof(packet.seed), packet.seed);
      } else {
        args.shiftX = int32_t(std::round((packet.originX-m_originX)/packet.spacing));
        args.shiftY = int32_t(std::round((packet.originY-m_originY)/packet.spacing));
        if (args.shiftX || args.shiftY) { dispatch(WATER_WAVE_SHIFT); }
      }
      ctx->writeToBuffer(m_mask, 0, sizeof(packet.mask), packet.mask);
      dispatch(WATER_WAVE_APPLY_MASK);
      if (packet.impulseCount) {
        ctx->writeToBuffer(m_impulses, 0, packet.impulseCount * sizeof(sim::Impulse), packet.impulses);
        args.impulseCount = packet.impulseCount;
        dispatch(WATER_WAVE_IMPULSE);
      }
      for (uint32_t step=0;step<packet.steps;++step) { dispatch(WATER_WAVE_STEP); }
      if (!m_valid || epoch != m_epoch) {
        Logger::info(str::format("[Water GPU Simulation] solver active: epoch=", epoch,
          " cells=", WATER_WAVE_COUNT, "; CPU event recording, GPU persistent height/velocity, no production readback"));
      }
      // The producer keeps publishing a packet even when there are no boats or
      // local wake sources.  Do not run the fixed-size hull/flow solver in
      // that idle state; keep dispatching while an already active wake is
      // decaying so existing foam is preserved.
      const bool hasFlowEvents = packet.flow.sourceCount != 0 || packet.flow.forceCount != 0;
      const bool flowNeedsWork = hasFlowEvents || m_flowActive;
      const bool hullNeedsWork = packet.flow.forceCount != 0 || m_hullActive;
      if (WaterSystem::hullWaves() && (packet.flow.enabled&sim::kHullWater) && hullNeedsWork && m_mapUploaded) {
        prepareHull(ctx,packet.flow,!m_valid || epoch!=m_epoch || packet.resetSerial!=m_resetSerial);
      } else { m_hullActive=false; }
      if (WaterSystem::localFlow() && (packet.flow.enabled&sim::kFoamFlow) && flowNeedsWork && m_mapUploaded) {
        prepareFlow(ctx, packet.flow, !m_valid || epoch != m_epoch);
      } else { m_flowActive = false; }
      m_valid = true; m_epoch = epoch; m_resetSerial = packet.resetSerial;
      m_originX = packet.originX; m_originY = packet.originY; m_spacing = packet.spacing;
      m_surface = packet.surface;
      if (std::isfinite(packet.flow.dt) && packet.flow.dt > 0.0f) {
        m_simulationTime += std::clamp(packet.flow.dt, 0.0f, 0.1f);
      }
      m_surface.sequence = packet.sequence ^ 0x80000000u;
    }
  }
  if (m_flowActive) {
    m_flowMesh = true;
    m_vertexCount = FLOW_COUNT; m_indexCount = 128*128*6;
    ++m_foamSequence;
    m_topologyHash = kFoamMesh ^ (uint64_t(m_foamSequence) * 0xD6E8FEB86659FD93ull);
    m_min = Vector3(m_flowX,m_flowY,m_waterMin-12);
    m_max = Vector3(m_flowX+128*.6f,m_flowY+128*.6f,m_waterMax+12);
  } else if (foamChanged) {
    m_flowMesh = false;
    m_vertexCount = uint32_t(vertices.size()); m_indexCount = uint32_t(indices.size());
    if (m_vertexCount && m_indexCount) {
      if (std::any_of(indices.begin(), indices.end(), [&](uint32_t i) { return i >= m_vertexCount; })) { m_indexCount = 0; }
      ctx->writeToBuffer(m_foamSource, 0, uint32_t(vertices.size() * sizeof(sim::Vertex)), vertices.data());
      ctx->writeToBuffer(m_foamIndices, 0, uint32_t(indices.size() * sizeof(uint32_t)), indices.data());
      m_topologyHash = XXH3_64bits(indices.data(), indices.size() * sizeof(uint32_t));
      m_min = Vector3(vertices[0].position[0],vertices[0].position[1],vertices[0].position[2]-12);
      m_max = Vector3(vertices[0].position[0],vertices[0].position[1],vertices[0].position[2]+12);
      for (const auto& v : vertices) {
        m_min.x = (std::min)(m_min.x,v.position[0]);m_min.y = (std::min)(m_min.y,v.position[1]);m_min.z = (std::min)(m_min.z,v.position[2]-12);
        m_max.x = (std::max)(m_max.x,v.position[0]);m_max.y = (std::max)(m_max.y,v.position[1]);m_max.z = (std::max)(m_max.z,v.position[2]+12);
      }
    }
  }
  if (m_valid) { frame = m_surface; }
  return m_valid;
}
void WaterGpuSimulation::prepareHull(const Rc<DxvkContext>& ctx, const sim::FlowFrame& frame, bool reset) {
  if (frame.forceCount>sim::kMaxFlowForces || !std::isfinite(frame.originX+frame.originY+frame.dt)
      || frame.dt<0 || frame.dt>.11f) { m_hullActive=false; return; }
  ScopedGpuProfileZone(ctx,"Water Hull Pressure Waves");
  if (m_hullState[0]==nullptr) {
    auto create=[&](size_t bytes,const char* name) {
      DxvkBufferCreateInfo info {};
      info.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      info.stages=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT;
      info.size=align(bytes,CACHE_LINE_SIZE);
      return m_device->createBuffer(info,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,DxvkMemoryStats::Category::RTXBuffer,name);
    };
    m_hullState[0]=create(HULL_COUNT*8,"Hull water height and velocity A");
    m_hullState[1]=create(HULL_COUNT*8,"Hull water height and velocity B");
    m_hullPressure=create(HULL_COUNT*4,"Hull water pressure");
    m_hullForces=create(sim::kMaxFlowForces*sizeof(sim::FlowForce),"Hull boat sources");
    m_hullDomain=create(16,"Hull water domain");
  }
  WaterHullArgs args {};
  const auto mapOrigin = WaterSystem::waterMapOrigin();
  const float mapCellSize = WaterSystem::waterMapCellSize();
  args.waterMapOriginX = std::isfinite(mapOrigin.x) ? mapOrigin.x : 0.0f;
  args.waterMapOriginY = std::isfinite(mapOrigin.y) ? mapOrigin.y : 0.0f;
  args.waterMapCellSize = std::isfinite(mapCellSize) ? std::clamp(mapCellSize, 0.01f, 1024.0f) : 1.0f;
  args.originX=frame.originX;args.originY=frame.originY;args.spacing=.6f;args.dt=1.f/120;
  args.speed=std::clamp(WaterSystem::hullWaveSpeed(),2.f,8.f);
  args.damping=std::clamp(WaterSystem::hullWaveDamping(),.05f,2.f);
  args.strength=std::clamp(WaterSystem::hullWaveStrength(),0.f,3.f);args.maxHeight=1.2f;
  args.forceCount=frame.forceCount;
  auto dispatch=[&](uint32_t op) {
    args.operation=op;
    ctx->bindResourceBuffer(222,DxvkBufferSlice(m_hullState[m_hullParity]));
    ctx->bindResourceBuffer(223,DxvkBufferSlice(m_hullState[m_hullParity^1]));
    ctx->bindResourceBuffer(224,DxvkBufferSlice(m_waterMap));
    ctx->bindResourceBuffer(225,DxvkBufferSlice(m_hullForces));
    ctx->bindResourceBuffer(226,DxvkBufferSlice(m_hullPressure));
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT,WaterHullShader::getShader());
    ctx->pushConstants(0,sizeof(args),&args);ctx->dispatch((HULL_COUNT+63)/64,1,1);
    if(op!=2) { m_hullParity^=1; }
  };
  if(reset||!m_hullActive) {
    dispatch(0);m_hullBudget=0;
    Logger::info("[Water Hull Waves] GPU pressure-height solver active; actual surface positions and geometric normals; independent of foam");
  } else {
    args.shiftX=int32_t(std::round((frame.originX-m_hullX)/.6f));
    args.shiftY=int32_t(std::round((frame.originY-m_hullY)/.6f));
    if(args.shiftX||args.shiftY) { dispatch(1); }
  }
  if(frame.forceCount) { ctx->writeToBuffer(m_hullForces, 0, frame.forceCount*sizeof(sim::FlowForce), frame.forces); }
  dispatch(2);
  m_hullBudget=(std::min)(m_hullBudget+frame.dt,.1f);
  while(m_hullBudget>=args.dt) { dispatch(3);m_hullBudget-=args.dt; }
  float domain[4]={frame.originX,frame.originY,.6f,0};ctx->writeToBuffer(m_hullDomain, 0, sizeof(domain), domain);
  m_hullX=frame.originX;m_hullY=frame.originY;m_hullActive=true;
}
void WaterGpuSimulation::prepareFlow(const Rc<DxvkContext>& ctx, const sim::FlowFrame& frame, bool reset) {
  if (frame.sourceCount > sim::kMaxFlowSources || frame.forceCount > sim::kMaxFlowForces
      || !std::isfinite(frame.originX+frame.originY+frame.dt+frame.time+frame.lifetime)
      || frame.dt < 0 || frame.dt > .11f || frame.lifetime <= 0) { m_flowActive = false; return; }
  ScopedGpuProfileZone(ctx, "Water Local Flow");
  if (m_flowState[0] == nullptr) {
    auto create = [&](size_t bytes, const char* name) {
      DxvkBufferCreateInfo info {};
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
      info.size = align(bytes, CACHE_LINE_SIZE);
      return m_device->createBuffer(info,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,DxvkMemoryStats::Category::RTXBuffer,name);
    };
    for (int i=0;i<2;++i) {
      m_flowState[i] = create(FLOW_COUNT*16,"Local flow velocity and foam");
      m_flowPressure[i] = create(FLOW_COUNT*4,"Local flow pressure");
    }
    m_flowDivergence = create(FLOW_COUNT*4,"Local flow divergence");
    m_flowSources = create(sim::kMaxFlowSources*sizeof(sim::FlowSource),"Local flow foam sources");
    m_flowForces = create(sim::kMaxFlowForces*sizeof(sim::FlowForce),"Local flow boat forces");
  }
  WaterFlowArgs args {};
  const auto mapOrigin = WaterSystem::waterMapOrigin();
  const float mapCellSize = WaterSystem::waterMapCellSize();
  args.waterMapOriginX = std::isfinite(mapOrigin.x) ? mapOrigin.x : 0.0f;
  args.waterMapOriginY = std::isfinite(mapOrigin.y) ? mapOrigin.y : 0.0f;
  args.waterMapCellSize = std::isfinite(mapCellSize) ? std::clamp(mapCellSize, 0.01f, 1024.0f) : 1.0f;
  args.originX=frame.originX; args.originY=frame.originY; args.spacing=.6f;
  args.drag=std::clamp(WaterSystem::localFlowDrag(),.05f,2.f);
  args.strength=std::clamp(WaterSystem::localFlowStrength(),0.f,4.f);
  args.swirl=std::clamp(WaterSystem::localFlowSwirl(),0.f,4.f);
  args.time=frame.time; args.lifetime=frame.lifetime;
  auto dispatch = [&](uint32_t operation) {
    args.operation=operation;
    ctx->bindResourceBuffer(FLOW_STATE_SOURCE,DxvkBufferSlice(m_flowState[m_flowParity]));
    ctx->bindResourceBuffer(FLOW_STATE_TARGET,DxvkBufferSlice(m_flowState[m_flowParity^1]));
    ctx->bindResourceBuffer(FLOW_PRESSURE_SOURCE,DxvkBufferSlice(m_flowPressure[m_pressureParity]));
    ctx->bindResourceBuffer(FLOW_PRESSURE_TARGET,DxvkBufferSlice(m_flowPressure[m_pressureParity^1]));
    ctx->bindResourceBuffer(FLOW_DIVERGENCE,DxvkBufferSlice(m_flowDivergence));
    ctx->bindResourceBuffer(FLOW_SOURCES,DxvkBufferSlice(m_flowSources));
    ctx->bindResourceBuffer(FLOW_FORCES,DxvkBufferSlice(m_flowForces));
    ctx->bindResourceBuffer(FLOW_VERTICES,DxvkBufferSlice(m_foamSource));
    ctx->bindResourceBuffer(FLOW_INDICES,DxvkBufferSlice(m_foamIndices));
    ctx->bindResourceBuffer(FLOW_WATER_MAP,DxvkBufferSlice(m_waterMap));
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT,WaterFlowShader::getShader());
    ctx->pushConstants(0,sizeof(args),&args);
    ctx->dispatch((FLOW_COUNT+63)/64,1,1);
    if (operation<=3 || operation==6 || operation==7) { m_flowParity^=1; }
    if (operation==0 || operation==4 || operation==5) { m_pressureParity^=1; }
  };
  if (reset || !m_flowActive || frame.resetSerial!=m_flowReset) {
    dispatch(0); m_flowBudget=0;
    Logger::info("[Water Local Flow] GPU MAC velocity, pressure projection and foam transport active; no production readback");
  } else {
    args.shiftX=int32_t(std::round((frame.originX-m_flowX)/.6f));
    args.shiftY=int32_t(std::round((frame.originY-m_flowY)/.6f));
    if (args.shiftX || args.shiftY) { dispatch(1); }
  }
  if (frame.sourceCount) {
    ctx->writeToBuffer(m_flowSources, 0, frame.sourceCount*sizeof(sim::FlowSource), frame.sources);
    args.sourceCount=frame.sourceCount; dispatch(2); args.sourceCount=0;
  }
  if (frame.forceCount) { ctx->writeToBuffer(m_flowForces, 0, frame.forceCount*sizeof(sim::FlowForce), frame.forces); }
  args.forceCount=frame.forceCount;
  m_flowBudget=(std::min)(m_flowBudget+frame.dt,.15f);
  args.dt=1.f/30;
  while (m_flowBudget>=args.dt) {
    m_flowBudget-=args.dt;
    if (args.forceCount) { dispatch(2); }
    dispatch(3); dispatch(4);
    for (int i=0;i<64;++i) { dispatch(5); }
    dispatch(6); dispatch(7);
  }
  dispatch(8);
  m_flowSourceEvents += frame.sourceCount;
  m_flowForceEvents += frame.forceCount;
  m_flowActive=true; m_flowReset=frame.resetSerial; m_flowX=frame.originX; m_flowY=frame.originY;
}
void WaterGpuSimulation::submitFoam(const Rc<DxvkContext>& ctx, const Rc<DxvkBuffer>& input, const Rc<DxvkBuffer>& positions, bool enabled,
                                    float verticalDisplacementPadding) {
  auto& scene = ctx->getCommonObjects()->getSceneManager();
  auto* assets = scene.getAssetReplacer().get();
  const auto mesh = reinterpret_cast<remixapi_MeshHandle>(kFoamMesh);
  if (!enabled || !m_valid || !m_vertexCount || !m_indexCount) {
    if (m_registered) { scene.destroyExternalMesh(ctx, mesh); m_registered = false; }
    return;
  }
  ScopedGpuProfileZone(ctx, "Water Foam Surface Follow");
  ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
  ctx->bindResourceBuffer(WATER_FOAM_FOLLOW_INPUT, DxvkBufferSlice(input));
  ctx->bindResourceBuffer(WATER_FOAM_FOLLOW_POSITIONS, DxvkBufferSlice(positions));
  ctx->bindResourceBuffer(WATER_FOAM_FOLLOW_SOURCE, DxvkBufferSlice(m_foamSource));
  ctx->bindResourceBuffer(WATER_FOAM_FOLLOW_TARGET, DxvkBufferSlice(m_foamTarget));
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, WaterFoamFollowShader::getShader());
  WaterFoamFollowArgs args { m_vertexCount };
  ctx->pushConstants(0, sizeof(args), &args);
  ctx->dispatch((m_vertexCount+63)/64, 1, 1);
  if (m_flowActive && WaterSystem::localFlowDiagnostics()) { diagnoseFlow(ctx); }
  RasterGeometry geometry;
  geometry.externalMaterial = reinterpret_cast<remixapi_MaterialHandle>(kFoamMaterial);
  geometry.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  geometry.cullMode = VK_CULL_MODE_NONE;
  geometry.frontFace = VK_FRONT_FACE_CLOCKWISE;
  geometry.vertexCount = m_vertexCount;
  const DxvkBufferSlice vb(m_foamTarget), ib(m_foamIndices);
  geometry.positionBuffer = RasterBuffer {vb,0,36,VK_FORMAT_R32G32B32_SFLOAT};
  geometry.normalBuffer = RasterBuffer {vb,12,36,VK_FORMAT_R32G32B32_SFLOAT};
  geometry.texcoordBuffer = RasterBuffer {vb,24,36,VK_FORMAT_R32G32_SFLOAT};
  geometry.color0Buffer = RasterBuffer {vb,32,36,VK_FORMAT_B8G8R8A8_UNORM};
  geometry.indexCount = m_indexCount; geometry.indexBuffer = RasterBuffer {ib,0,4,VK_INDEX_TYPE_UINT32};
  geometry.hashes[HashComponents::Indices] = m_topologyHash;
  geometry.hashes[HashComponents::GeometryDescriptor] = kFoamMesh;
  geometry.hashes[HashComponents::VertexLayout] = kFoamMesh+1;
  geometry.hashes[HashComponents::VertexTexcoord] = kFoamMesh ^ uint64_t(m_foamSequence);
  geometry.hashes[HashComponents::VertexPosition] = kFoamMesh ^ (++m_foamGeneration * 0x9E3779B97F4A7C15ull);
  geometry.hashes.precombine();geometry.boundingBox.minPos=m_min;geometry.boundingBox.maxPos=m_max;
  // The GPU follow pass samples the amplified ocean surface. Its geometry bounds
  // must include that same displacement even though the source foam stayed flat.
  geometry.boundingBox.minPos.z -= verticalDisplacementPadding;
  geometry.boundingBox.maxPos.z += verticalDisplacementPadding;
  if (!m_registered || !assets->registerGeneratedMesh(mesh,{geometry})) {
    assets->registerExternalMesh(mesh,{geometry});m_registered=true;
    Logger::info("[Water GPU Simulation] wake foam registered; GPU height and normal sampling, original foam material");
  }
  auto state=std::make_unique<ExternalDrawState>();
  state->drawCall.cameraType=CameraType::Main;
  state->drawCall.modifyTransformData().objectToWorld=Matrix4();
  state->drawCall.modifyTransformData().texgenMode=TexGenMode::None;
  state->drawCall.modifyCategoryFlags().set(InstanceCategories::IgnoreAntiCulling);
  auto& material=state->drawCall.modifyMaterialData();
  material.alphaTestEnabled=true;material.alphaTestReferenceValue=1;material.alphaTestCompareOp=VK_COMPARE_OP_GREATER;
  material.blendMode.enableBlending=true;
  material.blendMode.colorSrcFactor=VK_BLEND_FACTOR_SRC_ALPHA;material.blendMode.colorDstFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  material.blendMode.colorBlendOp=VK_BLEND_OP_ADD;
  material.blendMode.alphaSrcFactor=VK_BLEND_FACTOR_ONE;material.blendMode.alphaDstFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  material.blendMode.alphaBlendOp=VK_BLEND_OP_ADD;material.blendMode.writeMask=15;
  material.textureColorArg1Source=RtTextureArgSource::Texture;material.textureColorOperation=DxvkRtTextureOperation::SelectArg1;
  material.textureAlphaArg1Source=RtTextureArgSource::Texture;material.textureAlphaArg2Source=RtTextureArgSource::VertexColor0;
  material.textureAlphaOperation=DxvkRtTextureOperation::Modulate;material.tFactor=0xffffffff;material.isVertexColorBakedLighting=false;
  state->mesh=mesh;state->cameraType=CameraType::Main;
  state->categories.set(InstanceCategories::IgnoreAntiCulling);state->doubleSided=true;
  scene.submitExternalDraw(ctx,std::move(state));
}
void WaterGpuSimulation::diagnoseFlow(const Rc<DxvkContext>& ctx) {
  constexpr size_t stateBytes = FLOW_COUNT * 16;
  constexpr size_t vertexBytes = FLOW_COUNT * sizeof(sim::Vertex);
  constexpr size_t indexBytes = 128 * 128 * 6 * sizeof(uint32_t);
  if (m_flowReadPending && m_flowFence->value() >= m_flowSignal) {
    const auto* data = static_cast<const uint8_t*>(m_flowReadback->mapPtr(0));
    if (data) {
      const auto* flow = reinterpret_cast<const float*>(data);
      const auto* vertices = reinterpret_cast<const sim::Vertex*>(data + stateBytes);
      const auto* indices = reinterpret_cast<const uint32_t*>(data + stateBytes + vertexBytes);
      double mass = 0;
      float maxSpeed = 0, maxDensity = 0, minZ = 1e20f, maxZ = -1e20f;
      uint32_t visible = 0, triangles = 0, alphaTriangles = 0, invalid = 0, peak = 0;
      for (uint32_t i=0;i<FLOW_COUNT;++i) {
        const float u=flow[i*4],v=flow[i*4+1],density=flow[i*4+2];
        if (!std::isfinite(u+v+density)) { ++invalid; continue; }
        mass+=density; maxSpeed=(std::max)(maxSpeed,std::hypot(u,v));
        if (density>maxDensity) { maxDensity=density; peak=i; }
        if ((vertices[i].color>>24)>1) {
          ++visible; minZ=(std::min)(minZ,vertices[i].position[2]); maxZ=(std::max)(maxZ,vertices[i].position[2]);
        }
      }
      for (uint32_t i=0;i<128*128*6;i+=3) {
        const uint32_t a=indices[i],b=indices[i+1],c=indices[i+2];
        if (a>=FLOW_COUNT || b>=FLOW_COUNT || c>=FLOW_COUNT) { ++invalid; continue; }
        if (a!=b && b!=c && a!=c) {
          ++triangles;
          alphaTriangles+=((vertices[a].color|vertices[b].color|vertices[c].color)>>24)>1;
        }
      }
      Logger::info(str::format("[Water Local Flow GPU result] sample=",m_flowSignal,
        " mass=",mass," maxDensity=",maxDensity," maxSpeed=",maxSpeed,
        " visibleVertices=",visible," triangles=",triangles," alphaTriangles=",alphaTriangles,
        " visibleZ=",minZ,"..",maxZ," peakWorld=",vertices[peak].position[0],",",vertices[peak].position[1],",",vertices[peak].position[2],
        " peakAlpha=",vertices[peak].color>>24," invalid=",invalid));
    } else { Logger::warn("[Water Local Flow GPU result] completed readback could not be mapped"); }
    m_flowReadPending=false;
  }
  if (m_flowReadPending || GetTickCount()-m_flowReadTick<2000) { return; }
  if (m_flowReadback == nullptr) {
    DxvkBufferCreateInfo info {};
    info.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages=VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_HOST_BIT;
    info.access=VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_HOST_READ_BIT;
    info.size=align(stateBytes+vertexBytes+indexBytes,CACHE_LINE_SIZE);
    m_flowReadback=m_device->createBuffer(info,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      DxvkMemoryStats::Category::RTXBuffer,"Local flow temporary diagnostic readback");
  }
  ctx->copyBuffer(m_flowReadback,0,m_flowState[m_flowParity],0,stateBytes);
  ctx->copyBuffer(m_flowReadback,stateBytes,m_foamTarget,0,vertexBytes);
  ctx->copyBuffer(m_flowReadback,stateBytes+vertexBytes,m_foamIndices,0,indexBytes);
  ctx->emitMemoryBarrier(0,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_HOST_READ_BIT);
  ctx->signal(m_flowFence,++m_flowSignal);
  Logger::info(str::format("[Water Local Flow GPU request] sample=",m_flowSignal," sourceEvents=",m_flowSourceEvents,
    " boatEvents=",m_flowForceEvents," origin=",m_flowX,",",m_flowY," strength=",WaterSystem::localFlowStrength()," swirl=",WaterSystem::localFlowSwirl()));
  m_flowSourceEvents=m_flowForceEvents=0; m_flowReadTick=GetTickCount(); m_flowReadPending=true;
}
}
