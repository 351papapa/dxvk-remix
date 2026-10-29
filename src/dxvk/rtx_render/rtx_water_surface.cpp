#include "rtx_water_surface.h"
#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_asset_replacer.h"
#include "rtx_shader_manager.h"
#include "rtx_water.h"
#include "rtx_numos_weather.h"
#include "rtx/pass/common_binding_indices.h"
#include "dxvk_scoped_annotation.h"
#include "rtx/pass/water_surface_args.h"
#include <rtx_shaders/water_surface.h>
#include <rtx_shaders/water_surface_shore.h>
#include <rtx_shaders/water_surface_simulation.h>
#include <rtx_shaders/water_surface_hull.h>
#include <Windows.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <filesystem>
#include <fstream>

namespace dxvk {
namespace {
namespace gpu = remixapi::water::gpu;
constexpr uint64_t kMesh = 0x574C475055535246ull;
constexpr uint64_t kMaterial = 0x57415445524C4142ull;
struct WaterVertex { float position[3], normal[3], uv[2]; uint32_t color; };
static_assert(sizeof(WaterVertex) == 36);
static_assert(sizeof(WaterSurfaceArgs) == 28);
static_assert(offsetof(WaterSurfaceArgs, oceanWaveStrength) == 24);
static_assert(gpu::kSections == WATER_SECTIONS && gpu::kSplashes == WATER_SPLASHES);
static_assert(gpu::kVertexCount == WATER_VERTEX_COUNT);
class WaterSurfaceShoreShader : public ManagedShader {
  SHADER_SOURCE(WaterSurfaceShoreShader, VK_SHADER_STAGE_COMPUTE_BIT, water_surface_shore)
  PUSH_CONSTANTS(WaterSurfaceArgs)
  BEGIN_PARAMETER()
    STRUCTURED_BUFFER(WATER_SURFACE_INPUT)
    STRUCTURED_BUFFER(WATER_SURFACE_BEACH_MASK)
    RW_STRUCTURED_BUFFER(WATER_SURFACE_SHORE_DEPTH)
    RW_STRUCTURED_BUFFER(WATER_SURFACE_SHORE_TERRAIN)
    STRUCTURED_BUFFER(WATER_SURFACE_SHORE_SEEDS)
    RW_STRUCTURED_BUFFER(WATER_SURFACE_SHORE_NEXT)
    ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE)
    STRUCTURED_BUFFER(BINDING_SURFACE_DATA_BUFFER)
    STRUCTURED_BUFFER(BINDING_SURFACE_MATERIAL_DATA_BUFFER)
  END_PARAMETER()
};
class WaterSurfaceShader : public ManagedShader {
  SHADER_SOURCE(WaterSurfaceShader, VK_SHADER_STAGE_COMPUTE_BIT, water_surface)
  PUSH_CONSTANTS(WaterSurfaceArgs)
  BEGIN_PARAMETER()
    RW_STRUCTURED_BUFFER(WATER_SURFACE_INPUT)
    STRUCTURED_BUFFER(WATER_SURFACE_BEACH_MASK)
    RW_STRUCTURED_BUFFER(WATER_SURFACE_POSITIONS)
    RW_STRUCTURED_BUFFER(WATER_SURFACE_VERTICES)
    STRUCTURED_BUFFER(WATER_SURFACE_SHORE_DEPTH)
  END_PARAMETER()
};
class WaterSurfaceSimulationShader : public ManagedShader {
  SHADER_SOURCE(WaterSurfaceSimulationShader, VK_SHADER_STAGE_COMPUTE_BIT, water_surface_simulation)
  PUSH_CONSTANTS(WaterSurfaceArgs)
  BEGIN_PARAMETER()
    RW_STRUCTURED_BUFFER(WATER_SURFACE_INPUT)
    STRUCTURED_BUFFER(WATER_SURFACE_BEACH_MASK)
    RW_STRUCTURED_BUFFER(WATER_SURFACE_POSITIONS)
    RW_STRUCTURED_BUFFER(WATER_SURFACE_VERTICES)
    STRUCTURED_BUFFER(WATER_SURFACE_SHORE_DEPTH)
    STRUCTURED_BUFFER(225)
  END_PARAMETER()
};
class WaterSurfaceHullShader : public ManagedShader {
  SHADER_SOURCE(WaterSurfaceHullShader, VK_SHADER_STAGE_COMPUTE_BIT, water_surface_hull)
  PUSH_CONSTANTS(WaterSurfaceArgs)
  BEGIN_PARAMETER()
    RW_STRUCTURED_BUFFER(WATER_SURFACE_INPUT)
    STRUCTURED_BUFFER(WATER_SURFACE_BEACH_MASK)
    RW_STRUCTURED_BUFFER(WATER_SURFACE_POSITIONS)
    RW_STRUCTURED_BUFFER(WATER_SURFACE_VERTICES)
    STRUCTURED_BUFFER(WATER_SURFACE_SHORE_DEPTH)
    STRUCTURED_BUFFER(225)
    STRUCTURED_BUFFER(226)
    STRUCTURED_BUFFER(227)
  END_PARAMETER()
};
}
WaterGpuSurface::~WaterGpuSurface() {
  if (m_shared) { UnmapViewOfFile(m_shared); }
  if (m_mapping) { CloseHandle(m_mapping); }
  if (m_mutex) { CloseHandle(m_mutex); }
}
void WaterGpuSurface::ensureResources(const Rc<DxvkContext>& ctx) {
  if (m_vertices != nullptr) { return; }
  auto create = [&](size_t bytes, const char* name) {
    DxvkBufferCreateInfo info {};
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
      | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
      | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    info.size = align(bytes, CACHE_LINE_SIZE);
    return m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer, name);
  };
  m_input = create(sizeof(m_frame.data), "Water surface inputs");
  m_positions = create(gpu::kVertexCount * 12, "Water surface positions");
  m_vertices = create(WATER_RENDER_VERTEX_COUNT * sizeof(WaterVertex), "Water surface vertices");
  m_indices = create(WATER_RENDER_INDEX_COUNT * sizeof(uint32_t), "Water surface indices");
  std::vector<uint32_t> renderIndices(WATER_RENDER_INDEX_COUNT);
  for(uint32_t i=0;i<WATER_RENDER_INDEX_COUNT;++i) { renderIndices[i]=i; }
  ctx->uploadBuffer(m_indices,renderIndices.data(),uint32_t(renderIndices.size()*sizeof(uint32_t)));
  m_shoreDepth = create(WATER_VERTEX_COUNT * 4 * sizeof(float), "Water shore terrain depth");
  m_shoreTerrain = create(WATER_VERTEX_COUNT * 2 * sizeof(float), "Water shore raw terrain");
  m_shoreSeeds[0] = create(WATER_VERTEX_COUNT * 2 * sizeof(float), "Water shore seeds A");
  m_shoreSeeds[1] = create(WATER_VERTEX_COUNT * 2 * sizeof(float), "Water shore seeds B");
}
void WaterGpuSurface::updateBeachMask(const Rc<DxvkContext>& ctx) {
  const std::string path = WaterSystem::beachMaskPath();
  if (m_beachMask != nullptr && path == m_beachMaskPath) { return; }
  m_beachMaskPath = path;
  // uint header: width, height, world origin.xy, world inverse extent.xy, enabled, reserved.
  std::vector<uint32_t> words(8, 0u);
  if (!path.empty()) {
    std::ifstream file(std::filesystem::u8path(WaterSystem::resolveAssetPath(path)), std::ios::binary);
    uint32_t header[3] {};
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    const bool dimensionsValid = header[0] == 0x58545649u && header[1] >= 16u
      && header[1] <= 4096u && header[2] > 0u && header[2] <= 4096u;
    bool valid = file.good() && dimensionsValid;
    std::vector<uint32_t> pixels;
    float transform[4] {};
    if (valid) {
      pixels.resize(size_t(header[1]) * header[2]);
      file.read(reinterpret_cast<char*>(pixels.data()), std::streamsize(pixels.size() * sizeof(uint32_t)));
      valid = bool(file);
      if (valid) {
        uint8_t metadata[16];
        for (uint32_t i = 0; i < 16; ++i) { metadata[i] = uint8_t(pixels[i] >> 24); }
        std::memcpy(transform, metadata, sizeof(transform));
        valid = std::all_of(transform, transform + 4, [](float x) { return std::isfinite(x); })
          && transform[2] > 0.0f && transform[2] < 1.0f && transform[3] > 0.0f && transform[3] < 1.0f;
      }
    }
    if (valid) {
      words[0] = header[1]; words[1] = header[2];
      std::memcpy(words.data() + 2, transform, sizeof(transform));
      words[6] = 1u;
      words.insert(words.end(), pixels.begin(), pixels.end());
      Logger::info(str::format("[WaterShore] beachMask loaded: ", path, " size=", header[1], "x", header[2],
        " worldOrigin=", transform[0], ",", transform[1], " worldExtent=", 1.0f / transform[2], ",", 1.0f / transform[3]));
    } else {
      Logger::warn(str::format("[WaterShore] beachMask invalid or missing: ", path, "; using terrain eligibility fallback"));
    }
  }
  DxvkBufferCreateInfo info {};
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  info.size = align(words.size() * sizeof(uint32_t), CACHE_LINE_SIZE);
  m_beachMask = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    DxvkMemoryStats::Category::RTXBuffer, "Water beach eligibility mask");
  ctx->uploadBuffer(m_beachMask, words.data(), uint32_t(words.size() * sizeof(uint32_t)));
}

void WaterGpuSurface::submit(const Rc<DxvkContext>& ctx, bool enabled, bool simulationEnabled) {
  ScopedCpuProfileZone();
  const auto mesh = reinterpret_cast<remixapi_MeshHandle>(kMesh);
  auto& scene = ctx->getCommonObjects()->getSceneManager();
  auto* assets = scene.getAssetReplacer().get();
  if (!m_mutex) { m_mutex = OpenMutexW(SYNCHRONIZE | MUTEX_MODIFY_STATE, FALSE, gpu::kMutexName); }
  if (!m_mapping) { m_mapping = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, gpu::kMappingName); }
  if (m_mapping && !m_shared) {
    m_shared = static_cast<gpu::Frame*>(MapViewOfFile(m_mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(gpu::Frame)));
  }
  const auto material = reinterpret_cast<remixapi_MaterialHandle>(kMaterial);
  const bool available = enabled && assets->accessExternalMaterial(material) != nullptr;
  if (available && m_shared) { ensureResources(ctx); updateBeachMask(ctx); }
  bool copied = false;
  if (m_shared && m_mutex) {
    const DWORD result = WaitForSingleObject(m_mutex, 0);
    if (result == WAIT_OBJECT_0 || result == WAIT_ABANDONED) {
      if (m_shared->version == gpu::kVersion && m_shared->bytes == sizeof(gpu::Frame)) {
        m_frame = *m_shared;
        m_shared->ready = available ? 1u : 0u;
        m_shared->readyTick = GetTickCount();
        copied = true;
      }
      ReleaseMutex(m_mutex);
    }
  }
  if (!m_simulation) { m_simulation = std::make_unique<WaterGpuSimulation>(m_device); }
  const bool simulated = m_simulation->prepare(ctx, available && simulationEnabled, m_frame);
  const bool fresh = GetTickCount() - m_frame.tick < 500u;
  if (!enabled || !fresh || !m_frame.active || !available) {
    if (m_registered) { scene.destroyExternalMesh(ctx, mesh); m_registered = false; }
    m_simulation->submitFoam(ctx, m_input, m_positions, false);
    return;
  }
  const float* f = m_frame.data;
  if (!std::all_of(f, f + 32, [](float x) { return std::isfinite(x); })
      || f[3] <= 0 || f[9] <= 0 || f[12] <= 0) {
    if (m_registered) { scene.destroyExternalMesh(ctx, mesh); m_registered = false; }
    m_simulation->submitFoam(ctx, m_input, m_positions, false);
    return;
  }
  // This generated mesh uses only the shared-protocol ocean material (kMaterial),
  // the same identity used by Surface::isOceanWater. Authored pools and fountains
  // never enter this displacement path.
  const auto finiteOption = [](float value, float fallback, float low, float high) {
    return std::clamp(std::isfinite(value) ? value : fallback, low, high);
  };
  const float rainyMultiplier = finiteOption(WaterSystem::rainWaveMultiplier(), 1.25f, 1.0f, 3.0f);
  const float stormyMultiplier = finiteOption(WaterSystem::stormWaveMultiplier(), 1.5f, 1.0f, 3.0f);
  const float weatherMultiplier = ctx->getCommonObjects()->metaNumosWeather().waterState().multiplier(rainyMultiplier, stormyMultiplier);
  WaterSurfaceArgs args {};
  args.oceanWaveStrength = finiteOption(WaterSystem::oceanWaveStrength(), 1.25f, 0.0f, 3.0f) * weatherMultiplier;
  args.beachWaveHeight = WaterSystem::beachWaves()
    ? finiteOption(WaterSystem::beachWaveHeight(), 1.06f, 0.0f, 2.0f) : 0.0f;
  args.beachWaveLength = finiteOption(WaterSystem::beachWaveLength(), 8.9f, 2.0f, 100.0f);
  args.beachWavePeriod = finiteOption(WaterSystem::beachWavePeriod(), 4.5f, 1.0f, 30.0f);
  args.beachWaveDepth = finiteOption(WaterSystem::beachWaveDepth(), 3.1f, 0.25f, 12.0f);
  // The producer clock pauses with the game and already drives ocean phases.
  args.timeSeconds = std::max(f[4], 0.0f);
  const uint64_t settingsHash = XXH3_64bits_withSeed(m_beachMaskPath.data(), m_beachMaskPath.size(), XXH3_64bits(&args, sizeof(args)));
  // Preserve the existing interaction/hull safety margin, then add a spectrum
  // bound (sum of amplitudes) and both terrain-driven swells.
  float swellMagnitude = 0.0f;
  for (uint32_t i = 0; i < 4; ++i) { swellMagnitude += std::abs(f[24+i]) * args.oceanWaveStrength; }
  swellMagnitude *= WATER_OCEAN_AMPLITUDE_BOUND_SCALE;
  // Bound both profiles (1.45 each, second weight <= 0.55) and the foam lift.
  const float shoreAmplitudeBound = std::min(
    args.beachWaveHeight * 1.25f * std::pow(2.0f * args.beachWaveDepth, 0.25f),
    0.03f + 0.39f * std::min(2.2f * args.beachWaveDepth, 16.0f));
  const float shoreDisplacement = args.beachWaveHeight > 0.0f
    ? 2.2475f * shoreAmplitudeBound + 0.155f : 0.0f;
  const float verticalDisplacement = swellMagnitude + shoreDisplacement;
  const float horizontalPadding = 8.0f + std::abs(f[28]) * swellMagnitude;
  const float verticalPadding = 12.0f + verticalDisplacement;
  // A bridge frame can be reused between producer ticks. Still update GPU
  // geometry when the current weather blend or a wave slider changes.
  if ((copied || simulated || settingsHash != m_settingsHash)
      && (!m_registered || m_sequence != m_frame.sequence || settingsHash != m_settingsHash)) {
    ScopedGpuProfileZone(ctx, "Water Surface Deformation");
    // Publish the exact gain paired with these inputs, not a later option or
    // weather value, so analytic crest foam follows this frame's GPU surface.
    m_oceanWaveStrength = args.oceanWaveStrength;
    ctx->writeToBuffer(m_input, 0, sizeof(m_frame.data), f);
    // Domain changes invalidate vertex correspondence, not the immutable index topology.
    m_topologyHash=XXH3_64bits_withSeed(f,4*sizeof(float),kMesh);
    m_topologyHash=XXH3_64bits_withSeed(f+8,5*sizeof(float),m_topologyHash);
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx->bindResourceBuffer(WATER_SURFACE_INPUT, DxvkBufferSlice(m_input));
    ctx->bindResourceBuffer(WATER_SURFACE_POSITIONS, DxvkBufferSlice(m_positions));
    ctx->bindResourceBuffer(WATER_SURFACE_VERTICES, DxvkBufferSlice(m_vertices));
    ctx->bindResourceBuffer(WATER_SURFACE_SHORE_DEPTH, DxvkBufferSlice(m_shoreDepth));
    ctx->bindResourceBuffer(WATER_SURFACE_BEACH_MASK, DxvkBufferSlice(m_beachMask));
    const auto previousSurfaces = scene.getSurfaceBuffer();
    const auto previousMaterials = scene.getSurfaceMaterialBuffer();
    const auto previousTlas = ctx->getCommonObjects()->getResources().getTLAS(Tlas::Opaque).accelStructure;
    const bool shoreReady = args.beachWaveHeight > 0.0f && previousTlas.ptr() != nullptr
      && previousSurfaces != nullptr && previousMaterials != nullptr;
    if ((m_shoreLogFrames++ % 600u) == 0u) {
      Logger::info(str::format("[WaterShore] terrainProbe=", shoreReady,
        " height=", args.beachWaveHeight, " length=", args.beachWaveLength,
        " period=", args.beachWavePeriod, " depth=", args.beachWaveDepth,
        " spacing=", f[3], " coarse/fineSamples=", WATER_VERTEX_COUNT,
        " oceanWaveStrength=", args.oceanWaveStrength, " weatherMultiplier=", weatherMultiplier));
    }
    if (shoreReady) {
      ScopedGpuProfileZone(ctx, "Water Shore Terrain");
      // This submit precedes prepareSceneData: all three resources describe the
      // same completed frame. Never fetch geometry through this frame's tables.
      ctx->bindAccelerationStructure(BINDING_ACCELERATION_STRUCTURE, previousTlas);
      ctx->bindResourceBuffer(BINDING_SURFACE_DATA_BUFFER, DxvkBufferSlice(previousSurfaces));
      ctx->bindResourceBuffer(BINDING_SURFACE_MATERIAL_DATA_BUFFER, DxvkBufferSlice(previousMaterials));
      ctx->bindResourceBuffer(WATER_SURFACE_SHORE_TERRAIN, DxvkBufferSlice(m_shoreTerrain));
      ctx->bindResourceBuffer(WATER_SURFACE_SHORE_SEEDS, DxvkBufferSlice(m_shoreSeeds[1]));
      ctx->bindResourceBuffer(WATER_SURFACE_SHORE_NEXT, DxvkBufferSlice(m_shoreSeeds[0]));
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, WaterSurfaceShoreShader::getShader());
      const uint32_t groups = (WATER_VERTEX_COUNT + WATER_SURFACE_GROUP_SIZE - 1) / WATER_SURFACE_GROUP_SIZE;
      args.phase = 0;
      ctx->pushConstants(0, sizeof(args), &args);
      ctx->dispatch(groups, 1, 1);
      args.phase = 1;
      ctx->pushConstants(0, sizeof(args), &args);
      ctx->dispatch(groups, 1, 1);
      uint32_t current = 0;
      for (uint32_t jump = 64; jump > 0; jump >>= 1) {
        ctx->bindResourceBuffer(WATER_SURFACE_SHORE_SEEDS, DxvkBufferSlice(m_shoreSeeds[current]));
        ctx->bindResourceBuffer(WATER_SURFACE_SHORE_NEXT, DxvkBufferSlice(m_shoreSeeds[1-current]));
        args.phase = jump + 2;
        ctx->pushConstants(0, sizeof(args), &args);
        ctx->dispatch(groups, 1, 1);
        current = 1-current;
      }
      ctx->bindResourceBuffer(WATER_SURFACE_SHORE_SEEDS, DxvkBufferSlice(m_shoreSeeds[current]));
      args.phase = 100;
      ctx->pushConstants(0, sizeof(args), &args);
      ctx->dispatch(groups, 1, 1);
      args.phase = 101;
      ctx->pushConstants(0, sizeof(args), &args);
      ctx->dispatch(groups, 1, 1);
    } else {
      ctx->clearBuffer(m_shoreDepth, 0, m_shoreDepth->info().size, 0);
    }
    if (simulated) {
      ctx->bindResourceBuffer(225, DxvkBufferSlice(m_simulation->heights()));
      if(m_simulation->hullActive()) {
        ctx->bindResourceBuffer(226,DxvkBufferSlice(m_simulation->hullHeights()));
        ctx->bindResourceBuffer(227,DxvkBufferSlice(m_simulation->hullDomain()));
        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT,WaterSurfaceHullShader::getShader());
      } else { ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, WaterSurfaceSimulationShader::getShader()); }
    } else {
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, WaterSurfaceShader::getShader());
    }
    for (uint32_t phase = 0; phase < 3; ++phase) {
      args.phase = phase;
      ctx->pushConstants(0, sizeof(args), &args);
      uint32_t count = phase == 0 ? gpu::kCoarseCount
        : phase == 1 ? gpu::kFineCount : WATER_RENDER_CELL_COUNT;
      ctx->dispatch((count + WATER_SURFACE_GROUP_SIZE - 1) / WATER_SURFACE_GROUP_SIZE, 1, 1);
    }
    m_sequence = m_frame.sequence;
    m_settingsHash = settingsHash;
    // Foam lifetime uses game seconds, independent of the swell speed multiplier.
    m_submittedTime = simulated ? m_simulation->simulationTime() : f[4];
    ++m_generation;
  }
  RasterGeometry geometry;
  geometry.externalMaterial = material;
  geometry.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  geometry.cullMode = VK_CULL_MODE_NONE;
  geometry.frontFace = VK_FRONT_FACE_CLOCKWISE;
  geometry.vertexCount = WATER_RENDER_VERTEX_COUNT;
  const DxvkBufferSlice vb(m_vertices), ib(m_indices);
  geometry.positionBuffer = RasterBuffer {vb, 0, 36, VK_FORMAT_R32G32B32_SFLOAT};
  geometry.normalBuffer = RasterBuffer {vb, 12, 36, VK_FORMAT_R32G32B32_SFLOAT};
  geometry.texcoordBuffer = RasterBuffer {vb, 24, 36, VK_FORMAT_R32G32_SFLOAT};
  geometry.color0Buffer = RasterBuffer {vb, 32, 36, VK_FORMAT_B8G8R8A8_UNORM};
  geometry.indexCount = WATER_RENDER_INDEX_COUNT;
  geometry.indexBuffer = RasterBuffer {ib, 0, 4, VK_INDEX_TYPE_UINT32};
  geometry.hashes[HashComponents::Indices] = kMesh + 3;
  geometry.hashes[HashComponents::GeometryDescriptor] = kMesh;
  geometry.hashes[HashComponents::VertexLayout] = kMesh + 1;
  geometry.hashes[HashComponents::VertexTexcoord] = m_topologyHash;
  geometry.hashes[HashComponents::VertexPosition] = kMesh ^ (m_generation * 0x9E3779B97F4A7C15ull);
  geometry.hashes.precombine();
  geometry.boundingBox.minPos = Vector3(f[0]-horizontalPadding, f[1]-horizontalPadding, f[2]-verticalPadding);
  geometry.boundingBox.maxPos = Vector3(f[0]+f[12]+horizontalPadding, f[1]+f[12]+horizontalPadding, f[2]+verticalPadding);
  if (!m_registered || !assets->registerGeneratedMesh(mesh, {geometry})) {
    assets->registerExternalMesh(mesh, {geometry});
    m_registered = true;
    Logger::info(str::format("[Water GPU Surface] registered: samples=", gpu::kVertexCount,
      " renderVertices=", WATER_RENDER_VERTEX_COUNT,
      " indices=", WATER_RENDER_INDEX_COUNT, " inputBytes=", sizeof(m_frame.data),
      "; GPU deformation, terrain shoreline waves and wake foam"));
  }
  auto state = std::make_unique<ExternalDrawState>();
  state->drawCall.cameraType = CameraType::Main;
  state->drawCall.modifyTransformData().objectToWorld = Matrix4();
  state->drawCall.modifyTransformData().texgenMode = TexGenMode::None;
  state->drawCall.modifyCategoryFlags().set(InstanceCategories::IgnoreAntiCulling);
  state->drawCall.modifyCategoryFlags().set(InstanceCategories::AnimatedWater);
  state->mesh = mesh;
  state->cameraType = CameraType::Main;
  state->categories.set(InstanceCategories::IgnoreAntiCulling);
  state->categories.set(InstanceCategories::AnimatedWater);
  state->doubleSided = true;
  scene.submitExternalDraw(ctx, std::move(state));
  m_simulation->submitFoam(ctx, m_input, m_positions, simulated, verticalDisplacement);
}
}
