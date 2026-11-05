#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>
#include <condition_variable>
#include <functional>
#include <deque>
#include <mutex>
#include <thread>

#include "rtx_grass.h"
#include "rtx_grass_gpu.h"
#include "rtx_grass_placement.h"
#include "../../util/util_env.h"
#include "rtx_mod_usd.h"
#include "rtx_scene_manager.h"
#include "rtx_texture_manager.h"
#include "../dxvk_device.h"
#include "../dxvk_context.h"

namespace dxvk {
namespace {
  constexpr size_t kMaxPatchPoints = GrassPatchBuilder::kMaxPatchPoints;
  constexpr size_t kMaxCachedPoints = 1048576;
  constexpr size_t kMaxTriangles = 131072;
  constexpr size_t kMaxCachedTriangles = 1048576;

  std::string filenameLower(std::string name) {
    std::replace(name.begin(), name.end(), '\\', '/');
    name = name.substr(name.find_last_of('/') + 1);
    const auto begin = name.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return {}; }
    name = name.substr(begin, name.find_last_not_of(" \t\r\n") - begin + 1);
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return c >= 'A' && c <= 'Z' ? char(c + 'a' - 'A') : char(c);
    });
    return name;
  }

  struct Triangle { Vector3 p[3]; Vector3 normal; float area; uint64_t seed; };

  bool finite(const Vector3& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
           std::abs(p.x) < 1e7f && std::abs(p.y) < 1e7f && std::abs(p.z) < 1e7f;
  }

  bool nearby(const AxisAlignedBoundingBox& box, const Matrix4& transform, const Vector3& camera, float radius) {
    if (!box.isValid()) { return false; }
    AxisAlignedBoundingBox world;
    for (unsigned i = 0; i < 8; ++i) {
      const auto p = (transform * Vector4((i & 1) ? box.maxPos.x : box.minPos.x,
        (i & 2) ? box.maxPos.y : box.minPos.y, (i & 4) ? box.maxPos.z : box.minPos.z, 1)).xyz();
      if (!finite(p)) { return false; }
      world.minPos = min(world.minPos, p);
      world.maxPos = max(world.maxPos, p);
    }
    const float dx = std::max({ world.minPos.x - camera.x, 0.0f, camera.x - world.maxPos.x });
    const unsigned horizontal = RtxOptions::zUp() ? 1u : 2u;
    const float dy = std::max({ world.minPos[horizontal] - camera[horizontal], 0.0f, camera[horizontal] - world.maxPos[horizontal] });
    return dx * dx + dy * dy <= radius * radius;
  }

  bool readTriangles(const RasterGeometry& geometry, const Matrix4& transform, float slope,
                     std::vector<Triangle>& triangles) {
    if (geometry.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || geometry.numBonesPerVertex ||
        geometry.calculatePrimitiveCount() > kMaxTriangles || !geometry.positionBuffer.defined() ||
        (geometry.positionBuffer.vertexFormat() != VK_FORMAT_R32G32B32_SFLOAT &&
         geometry.positionBuffer.vertexFormat() != VK_FORMAT_R32G32B32A32_SFLOAT) ||
        geometry.positionBuffer.stride() < 12) { return false; }
    const auto& vb = geometry.positionBuffer;
    const auto& ib = geometry.indexBuffer;
    const bool indexed = ib.defined();
    const uint32_t count = indexed ? geometry.indexCount : geometry.vertexCount;
    const uint32_t indexSize = indexed && ib.indexType() == VK_INDEX_TYPE_UINT32 ? 4 : 2;
    const uint32_t indexStride = indexed ? std::max(indexSize, ib.stride()) : 0;
    if (indexed && ib.indexType() != VK_INDEX_TYPE_UINT32 && ib.indexType() != VK_INDEX_TYPE_UINT16) { return false; }
    if (!geometry.vertexCount || !count || vb.isPendingGpuWrite() || (indexed && ib.isPendingGpuWrite()) ||
        vb.offsetFromSlice() + uint64_t(geometry.vertexCount - 1) * vb.stride() + 12 > vb.length() ||
        (indexed && ib.offsetFromSlice() + uint64_t(count - 1) * indexStride + indexSize > ib.length())) { return false; }
    const auto* vertices = static_cast<const uint8_t*>(vb.mapPtr(vb.offsetFromSlice()));
    const auto* indices = indexed ? static_cast<const uint8_t*>(ib.mapPtr(ib.offsetFromSlice())) : nullptr;
    if (!vertices || (indexed && !indices)) { return false; }
    const bool mirrored = dot(cross(transform[0].xyz(), transform[1].xyz()), transform[2].xyz()) < 0;
    for (uint32_t i = 0; i + 2 < count; i += 3) {
      Triangle tri;
      Vector3 localPoints[3];
      uint32_t firstIndex = 0;
      for (unsigned j = 0; j < 3; ++j) {
        uint32_t index = i + j;
        if (indexed) {
          index = 0;
          std::memcpy(&index, indices + size_t(i + j) * indexStride, indexSize);
        }
        if (index >= geometry.vertexCount) { return false; }
        Vector3 p;
        std::memcpy(&p, vertices + size_t(index) * vb.stride(), 12);
        localPoints[j] = p;
        if (j == 0) { firstIndex = index; }
        tri.p[j] = (transform * Vector4(p, 1)).xyz();
        if (!finite(tri.p[j])) { return false; }
      }
      Vector3 n = cross(tri.p[1] - tri.p[0], tri.p[2] - tri.p[0]);
      const float nLength = std::sqrt(dot(n, n));
      if (nLength < 1e-8f) { continue; }
      n /= nLength;
      bool flip = geometry.frontFace == VK_FRONT_FACE_COUNTER_CLOCKWISE;
      const auto& nb = geometry.normalBuffer;
      if (nb.defined() && !nb.isPendingGpuWrite()) {
        const auto* normalData = static_cast<const uint8_t*>(nb.mapPtr(nb.offsetFromSlice()));
        const size_t size = nb.vertexFormat() == VK_FORMAT_R32G32B32_SFLOAT ? 12 : 4;
        if (normalData && nb.offsetFromSlice() + uint64_t(firstIndex) * nb.stride() + size <= nb.length()) {
          Vector3 normal(0.0f);
          if (nb.vertexFormat() == VK_FORMAT_R32G32B32_SFLOAT) {
            std::memcpy(&normal, normalData + size_t(firstIndex) * nb.stride(), 12);
          } else if (nb.vertexFormat() == VK_FORMAT_R32_UINT) {
            uint32_t packed;
            std::memcpy(&packed, normalData + size_t(firstIndex) * nb.stride(), 4);
            normal.x = float(packed & 0xffffu) / 65535.0f * 2 - 1;
            normal.y = float(packed >> 16) / 65535.0f * 2 - 1;
            normal.z = 1 - std::abs(normal.x) - std::abs(normal.y);
            const float t = std::max(0.0f, -normal.z);
            normal.x += normal.x >= 0 ? -t : t;
            normal.y += normal.y >= 0 ? -t : t;
          }
          if (dot(normal, normal) > 1e-8f && finite(normal)) {
            flip = dot(cross(localPoints[1] - localPoints[0], localPoints[2] - localPoints[0]), normal) < 0;
          }
        }
      }
      if (flip != mirrored) { n = -n; }
      if (n.z < slope) { continue; }
      tri.normal = n;
      tri.area = nLength * 0.5f;
      tri.seed = XXH64(tri.p, sizeof(tri.p), 0x4752415353ull);
      triangles.push_back(tri);
    }
    return true;
  }
}

struct GrassSystem::Job {
  uint64_t key = 0;
  uint64_t generation = 0;
  std::vector<std::array<Vector3, 3>> triangles;
  Matrix4 world;
  float slope = 0;
  float density = 0;
  float margin = 0;
  bool inset = false;
  bool zUp = true;
  std::atomic<bool> cancelled { false };
  std::atomic<bool> ready { false };
  Patch result;
  std::string error;

  void build() {
    try {
      ScopedCpuProfileZone();
      result.points = std::make_shared<const std::vector<GrassRoot>>(
        GrassPatchBuilder::build(triangles, world, slope, density, margin, inset, zUp, cancelled));
    } catch (const std::exception& exception) {
      error = exception.what();
    } catch (...) {
      error = "Unknown patch generation failure";
    }
    ready.store(true, std::memory_order_release);
  }
};

struct GrassSystem::RootJob {
  uint64_t generation = 0;
  std::vector<uint64_t> keys;
  std::vector<std::shared_ptr<const std::vector<GrassRoot>>> patches;
  std::vector<GrassRoot> roots;
  std::atomic<bool> ready { false };
  std::atomic<bool> cancelled { false };
  std::string error;

  void build() {
    try {
      ScopedCpuProfileZone();
      size_t count = 0;
      for (const auto& patch : patches) { count += patch->size(); }
      roots.reserve(count);
      for (const auto& patch : patches) {
        if (cancelled.load(std::memory_order_relaxed)) { break; }
        roots.insert(roots.end(), patch->begin(), patch->end());
      }
      const auto identity = [](const GrassRoot& root) { return (uint64_t(root.identityHi) << 32) | root.identityLo; };
      std::sort(roots.begin(), roots.end(), [&](const auto& a, const auto& b) { return identity(a) < identity(b); });
      roots.erase(std::unique(roots.begin(), roots.end(), [&](const auto& a, const auto& b) { return identity(a) == identity(b); }), roots.end());
    } catch (const std::exception& exception) {
      error = exception.what();
    } catch (...) {
      error = "Unknown root packing failure";
    }
    ready.store(true, std::memory_order_release);
  }
};

struct GrassSystem::Workers {
  std::mutex m_mutex;
  std::condition_variable wake;
  std::deque<std::function<void()>> queue;
  std::array<std::thread, 2> threads;
  bool stopped = false;

  Workers() {
    for (auto& thread : threads) {
      thread = std::thread([this] {
        env::setThreadName("rtx-grass");
        for (;;) {
          std::function<void()> job;
          {
            std::unique_lock<std::mutex> lock(m_mutex);
            wake.wait(lock, [this] { return stopped || !queue.empty(); });
            if (stopped) { return; }
            job = std::move(queue.front());
            queue.pop_front();
          }
          job();
        }
      });
    }
  }
  ~Workers() {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      stopped = true;
    }
    wake.notify_all();
    for (auto& thread : threads) { thread.join(); }
  }
  void schedule(std::function<void()> job) {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      queue.push_back(std::move(job));
    }
    wake.notify_one();
  }
};

GrassSystem::GrassSystem() = default;
GrassSystem::~GrassSystem() {
  for (const auto& job : m_jobs) { job->cancelled = true; }
  if (m_rootJob) { m_rootJob->cancelled = true; }
  m_workers.reset();
}

bool GrassSystem::matchesOriginalName(const std::string* name) const {
  if (!name || name->empty() || originalTextures().empty()) { return false; }
  std::istringstream rules(originalTextures());
  std::string rule;
  while (std::getline(rules, rule, ';')) {
    if (filenameLower(rule) == *name) { return true; }
  }
  return false;
}

void GrassSystem::clear() {
  ++m_generation;
  for (const auto& job : m_jobs) { job->cancelled = true; }
  if (m_rootJob) { m_rootJob->cancelled = true; }
  m_jobs.clear();
  m_rootJob.reset();
  m_patches.clear();
  m_sources.clear();
  m_activePatches.clear();
  m_candidateCount = 0;
  if (m_gpu) { m_gpu->clear(); }
  m_visibleClumps = m_matchedSurfaces = m_skippedSurfaces = 0;
}

void GrassSystem::update(const Rc<DxvkContext>& ctx, SceneManager& scene) {
  ScopedCpuProfileZone();
  const auto updateStart = std::chrono::steady_clock::now();
  m_createdClumps = 0;
  m_updateMilliseconds = 0;
  const uint32_t frame = ctx->getDevice()->getCurrentFrameId();
  m_frame = frame;
  m_visibleClumps = m_matchedSurfaces = m_skippedSurfaces = 0;
  m_limited = false;
  if (!enabled()) {
    if (m_wasEnabled) { clear(); }
    m_wasEnabled = false;
    m_status = "Disabled";
    return;
  }
  m_wasEnabled = true;
  if (m_loadedPath != assetPath() || m_reloadVersion != reloadVersion()) {
    clear();
    m_asset.reset();
    m_budget = 0;
    m_loadedPath = assetPath();
    m_reloadVersion = reloadVersion();
    try {
      if (m_loadedPath.empty()) { throw DxvkError("Set the grass USD path"); }
      wchar_t executable[32768] = {};
      GetModuleFileNameW(nullptr, executable, 32768);
      auto folder = std::filesystem::path(executable).parent_path();
      if (filenameLower(folder.filename().string()) == ".trex") {
        folder = folder.parent_path();
      }
      const auto resolved = (folder / std::filesystem::path(m_loadedPath)).lexically_normal();
      m_resolvedPath = resolved.string();
      m_asset = UsdMod::loadStandaloneMesh(ctx, resolved);
      m_boundWidth = m_boundHeight = 0;
      for (const auto& mesh : m_asset->replacements) {
        const auto& box = mesh.geometry->data.boundingBox;
        for (unsigned i = 0; i < 8; ++i) {
          const auto p = (mesh.replacementToObject * Vector4((i & 1) ? box.maxPos.x : box.minPos.x,
            (i & 2) ? box.maxPos.y : box.minPos.y, (i & 4) ? box.maxPos.z : box.minPos.z, 1)).xyz();
          m_boundWidth = std::max(m_boundWidth, std::sqrt(p.x * p.x + p.y * p.y));
          m_boundHeight = std::max(m_boundHeight, std::abs(p.z));
        }
      }
      m_status = "Loaded";
    } catch (const DxvkError& e) {
      m_status = e.message();
      Logger::err(str::format("[Grass] ", m_status));
    } catch (const std::exception& e) {
      m_status = e.what();
      Logger::err(str::format("[Grass] ", m_status));
    }
  }
  if (!m_asset) { return; }
  if (!m_gpu) { m_gpu = std::make_unique<GrassGpu>(ctx->getDevice().ptr()); }
  if (m_rules != diffuseTextures() || m_originalRules != originalTextures()) {
    m_rules = diffuseTextures(); m_originalRules = originalTextures(); m_names.clear(); clear();
    std::istringstream rules(m_rules);
    std::string name;
    while (std::getline(rules, name, ';')) {
      name = filenameLower(name);
      if (!name.empty()) { m_names.push_back(name); }
    }
  }
  if (m_names.empty() && m_originalRules.empty()) { m_status = "Add replacement or original ground texture names"; return; }
  const float d = std::clamp(density(), 0.01f, 10.0f);
  const float h = std::clamp(height(), 0.01f, 10.0f);
  const float w = std::clamp(width(), 0.01f, 10.0f);
  const float slope = std::cos(std::clamp(maxSlopeDegrees(), 0.0f, 80.0f) * 0.01745329252f);
  if (!std::isfinite(d) || !std::isfinite(h) || !std::isfinite(w) || !std::isfinite(slope) || !std::isfinite(radius())) {
    m_status = "Invalid numeric setting"; return;
  }
  if (m_density != d || m_width != w || m_slope != slope || m_inset != keepInsideSurface() || m_zUp != RtxOptions::zUp()) {
    if (m_zUp != RtxOptions::zUp()) { m_horizontalForward = RtxOptions::zUp() ? Vector3(0, 1, 0) : Vector3(0, 0, 1); }
    clear(); m_density = d; m_width = w; m_slope = slope; m_inset = keepInsideSurface(); m_zUp = RtxOptions::zUp();
  }
  for (auto it = m_jobs.begin(); it != m_jobs.end();) {
    auto& job = **it;
    if (!job.ready.load(std::memory_order_acquire)) { ++it; continue; }
    if (!job.cancelled && job.generation == m_generation && job.error.empty()) {
      job.result.lastFrame = frame;
      m_patches.emplace(job.key, std::move(job.result));
    } else if (!job.error.empty()) {
      Logger::err(str::format("[Grass] Patch worker: ", job.error));
    }
    it = m_jobs.erase(it);
  }
  for (auto it = m_sources.begin(); it != m_sources.end();) {
    if (frame - it->second.lastFrame > 120) { it = m_sources.erase(it); }
    else { ++it; }
  }
  size_t cachedTriangles = 0;
  for (const auto& source : m_sources) { cachedTriangles += source.second.triangles.size(); }
  size_t cachedPoints = 0;
  for (auto it = m_patches.begin(); it != m_patches.end();) {
    if (frame - it->second.lastFrame > 120) { it = m_patches.erase(it); }
    else { cachedPoints += it->second.points->size(); ++it; }
  }
  const float r = std::clamp(radius(), 1.0f, 200.0f);
  const Vector3 camera = scene.getCamera().getPosition();
  const Vector3 direction = scene.getCamera().getDirection();
  // The engine anti-culling frustum is not updated when anti-culling is disabled.
  // Derive our own world-space side planes from the actual camera instead.
  const Matrix4 worldToClip = scene.getCamera().getViewToProjectionf() * scene.getCamera().getWorldToViewf();
  std::array<Vector4, 5> viewPlanes;
  const Vector4 rowW(worldToClip[0][3], worldToClip[1][3], worldToClip[2][3], worldToClip[3][3]);
  for (unsigned axis = 0; axis < 2; ++axis) {
    const Vector4 row(worldToClip[0][axis], worldToClip[1][axis], worldToClip[2][axis], worldToClip[3][axis]);
    viewPlanes[axis * 2] = rowW + row;
    viewPlanes[axis * 2 + 1] = rowW - row;
  }
  viewPlanes[4] = rowW;
  for (auto& plane : viewPlanes) {
    const float length = std::sqrt(dot(plane.xyz(), plane.xyz()));
    if (length > 1e-6f) { plane /= length; }
  }
  const Vector3 horizontalDirection = m_zUp ? Vector3(direction.x, direction.y, 0) : Vector3(direction.x, 0, direction.z);
  const float horizontalLengthSquared = dot(horizontalDirection, horizontalDirection);
  // Keep the last heading when looking vertically, where projected forward is undefined.
  if (std::isfinite(horizontalLengthSquared) && horizontalLengthSquared > 1e-6f) {
    m_horizontalForward = horizontalDirection / std::sqrt(horizontalLengthSquared);
  }
  const auto& textures = ctx->getCommonObjects()->getTextureManager().getTextureTable();
  std::vector<int8_t> textureMatches(textures.size(), -1);
  std::vector<uint64_t> activePatches;
  const auto& instances = scene.getInstanceTable();
  unsigned builds = 0;
  std::unordered_set<uint64_t> visitedPatches;
  for (const auto* instance : instances) {
    auto* owner = const_cast<RtInstance*>(instance)->getPrimInstanceOwner().getReplacementInstance();
    if ((owner && owner->proceduralGrass) || instance->isHidden() || instance->getMaterialType() != MaterialDataType::Opaque) { continue; }
    bool match = !m_originalRules.empty() && (!owner || !owner->activeReplacements) &&
      matchesOriginalName(scene.getOriginalTextureName(instance->surface.originalTextureHash));
    const auto textureIndex = instance->getAlbedoOpacityTextureIndex();
    if (textureIndex < textures.size() && !m_names.empty()) {
      auto& cached = textureMatches[textureIndex];
      if (cached < 0) {
        cached = 0;
        const auto& managed = textures[textureIndex].getManagedTexture();
        if (managed != nullptr && managed->m_assetData != nullptr && managed->m_assetData->info().filename) {
          const auto name = filenameLower(managed->m_assetData->info().filename);
          cached = std::find(m_names.begin(), m_names.end(), name) != m_names.end() ? 1 : 0;
        }
      }
      match |= cached != 0;
    }
    if (!match) { continue; }
    const auto* blas = instance->getBlas();
    if (!blas) { continue; }
    const auto& geometry = blas->input.getGeometryData();
    if (geometry.numBonesPerVertex) { ++m_skippedSurfaces; continue; }
    const auto collect = [&](const Matrix4& world) {
      if (!nearby(geometry.boundingBox, world, camera, r)) { return; }
      ++m_matchedSurfaces;
      const uint64_t geomHash = geometry.hashes[HashComponents::VertexPosition];
      const uint64_t key = XXH64(&world, sizeof(world), XXH64(&geomHash, sizeof(geomHash), geometry.hashes[HashComponents::Indices]));
      if (!visitedPatches.insert(key).second) { return; }
      auto found = m_patches.find(key);
      if (found == m_patches.end()) {
        if (std::any_of(m_jobs.begin(), m_jobs.end(), [key](const auto& job) { return job->key == key && !job->cancelled; })) { return; }
        if (builds >= 2 || m_jobs.size() >= 4 || cachedPoints + (m_jobs.size() + 1) * kMaxPatchPoints > kMaxCachedPoints) {
          m_limited = true; return;
        }
        const auto sourceKey = XXH64(&geomHash, sizeof(geomHash), geometry.hashes[HashComponents::Indices]);
        auto source = m_sources.find(sourceKey);
        if (source == m_sources.end()) {
          std::vector<Triangle> triangles;
          if (!readTriangles(geometry, Matrix4(), -2.0f, triangles)) { ++m_skippedSurfaces; return; }
          if (cachedTriangles + triangles.size() > kMaxCachedTriangles) { m_limited = true; return; }
          cachedTriangles += triangles.size();
          Source captured;
          captured.lastFrame = frame;
          for (const auto& tri : triangles) {
            auto b = tri.p[1], c = tri.p[2];
            if (dot(cross(b - tri.p[0], c - tri.p[0]), tri.normal) < 0) { std::swap(b, c); }
            captured.triangles.push_back({ tri.p[0], b, c });
          }
          source = m_sources.emplace(sourceKey, std::move(captured)).first;
        }
        if (source == m_sources.end()) { ++m_skippedSurfaces; return; }
        source->second.lastFrame = frame;
        auto job = std::make_shared<Job>();
        job->key = key;
        job->generation = m_generation;
        job->triangles = source->second.triangles;
        job->world = world;
        job->slope = slope;
        job->density = d;
        job->margin = m_boundWidth * w;
        job->inset = m_inset;
        job->zUp = m_zUp;
        if (!m_workers) {
          m_workers = std::make_unique<Workers>();
        }
        m_workers->schedule([job] { job->build(); });
        m_jobs.push_back(job);
        ++builds;
        return;
      }
      found->second.lastFrame = frame;
      activePatches.push_back(key);
    };
    const auto& transforms = instance->surface.instancesToObject;
    if (transforms) {
      for (const auto& transform : *transforms) { collect(instance->surface.objectToWorld * transform); }
    } else { collect(instance->surface.objectToWorld); }
  }
  std::sort(activePatches.begin(), activePatches.end());
  if (m_rootJob && m_rootJob->ready.load(std::memory_order_acquire)) {
    if (!m_rootJob->cancelled && m_rootJob->generation == m_generation && m_rootJob->error.empty() && m_rootJob->keys == activePatches) {
      m_candidateCount = m_rootJob->roots.size();
      m_gpu->setRoots(std::move(m_rootJob->roots));
      m_activePatches = std::move(m_rootJob->keys);
    } else if (!m_rootJob->error.empty()) {
      Logger::err(str::format("[Grass] Root packing: ", m_rootJob->error));
    }
    m_rootJob.reset();
  }
  if (activePatches.empty()) {
    if (m_rootJob) { m_rootJob->cancelled = true; }
    if (m_candidateCount) { m_gpu->setRoots({}); }
    m_candidateCount = 0;
    m_activePatches.clear();
  } else if (activePatches != m_activePatches && !m_rootJob) {
    auto job = std::make_shared<RootJob>();
    job->generation = m_generation;
    job->keys = std::move(activePatches);
    for (uint64_t key : job->keys) { job->patches.push_back(m_patches.at(key).points); }
    m_workers->schedule([job] { job->build(); });
    m_rootJob = std::move(job);
  }
  const uint32_t budget = std::min(static_cast<uint32_t>(m_candidateCount), uint32_t(std::clamp(maxInstances(), 1, 512)));
  if (budget != m_budget) {
    // Replacing the bucket invalidates the preserve path when its slot count changes.
    auto asset = std::make_shared<ReplacementBucket>();
    asset->replacements = std::vector<AssetReplacement>(m_asset->replacements);
    for (auto& mesh : asset->replacements) {
      mesh.instancesToObject = std::make_shared<std::vector<Matrix4>>(budget);
    }
    m_asset = std::move(asset);
    m_budget = budget;
  }
  GrassConstants constants {};
  constants.cameraRadius = { camera.x, camera.y, camera.z, r };
  constants.forwardFade = { m_horizontalForward.x, m_horizontalForward.y, m_horizontalForward.z, std::isfinite(fadeDistance()) ? std::clamp(fadeDistance(), 0.01f, 200.0f) : 5.0f };
  constants.dimensions = { w, h, std::isfinite(groundOffset()) ? std::clamp(groundOffset(), -1.0f, 1.0f) : 0.0f, 0 };
  constants.bounds = { m_boundWidth, m_boundHeight, 0, 0 };
  memcpy(constants.viewPlanes, viewPlanes.data(), sizeof(constants.viewPlanes));
  constants.budget = budget;
  constants.frontRear = frontRearRange();
  constants.zUp = m_zUp;
  constants.frustumCull = frustumCull();
  m_gpu->setFrame(constants);
  if (budget) {
    m_createdClumps = scene.submitGrass(ctx, m_asset, 0x4750554752415353ull, Matrix4()) ? 1 : 0;
  }
  m_visibleClumps = budget;
  m_updateMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - updateStart).count();
  m_status = !m_jobs.empty() || m_rootJob ? "Building patches in background" : m_matchedSurfaces ? "Active (GPU instances)" : "No matching ground surfaces in range";
  if (frame % 600 == 0) {
    Logger::info(str::format("[Grass] Surfaces=", m_matchedSurfaces, " candidates=", m_candidateCount, " gpuBudget=", budget,
      " skipped=", m_skippedSurfaces, " jobs=", m_jobs.size(), " updateMs=", m_updateMilliseconds));
  }
}
}
