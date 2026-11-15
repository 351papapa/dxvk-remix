#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <Windows.h>

#include "numos/gpu.h"
#include "numos/lib/atmosphere_args.h"
#include "numos/util/camera.h"

#include "rtx_numos_cloud_probe.h"
#include "rtx_scene_manager.h"
#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../shaders/rtx/pass/instance_definitions.h"
#include "../../util/log/log.h"
#include "../../util/util_env.h"
#include "../../util/util_string.h"

namespace dxvk {
  namespace {
    constexpr uint32_t kMagic = 0x43535031u;
    constexpr uint32_t kPixels = 8;
    constexpr uint32_t kWords = 8256;
    constexpr size_t kProbeBytes = size_t(kWords) * sizeof(uint32_t);
    constexpr size_t kArgsOffset = kProbeBytes;
    constexpr size_t kCameraOffset = (kArgsOffset + sizeof(numos::AtmosphereArgs) + 15) & ~size_t(15);
    constexpr size_t kReadBytes = kCameraOffset + sizeof(numos::Camera);
    constexpr uint32_t kMaxSurfaces = 65536;

    void logProbe(const std::string& value) {
      Logger::info("[NumosCloudProbe] " + value);
    }

    float floatWord(uint32_t value) {
      float result;
      std::memcpy(&result, &value, sizeof(result));
      return result;
    }

    uint64_t hashBytes(const void* data, size_t size) {
      uint64_t hash = 14695981039346656037ull;
      const auto* bytes = static_cast<const uint8_t*>(data);
      for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ull;
      }
      return hash;
    }

    std::string jsonString(const std::string& value) {
      std::ostringstream out;
      out << '"';
      for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
          out << '\\' << char(c);
        } else if (c < 32) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << uint32_t(c) << std::dec;
        } else {
          out << char(c);
        }
      }
      out << '"';
      return out.str();
    }

    bool wideEnvironment(const wchar_t* name, std::wstring& value) {
      const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
      if (!size) {
        return false;
      }
      value.resize(size);
      const DWORD copied = GetEnvironmentVariableW(name, value.data(), size);
      if (!copied || copied >= size) {
        return false;
      }
      value.resize(copied);
      return true;
    }

    // Mirrors the existing cloud march lattice, including its non-raster Bayer order.
    std::array<uint32_t, 2> snapPixel(float u, float v, uint32_t width, uint32_t height,
      uint32_t frame, uint32_t interleave) {
      uint32_t cell = 1, px = 0, py = 0;
      if (interleave & 0x40000u) {
        if ((interleave & 0xC0000u) == 0xC0000u) {
          constexpr uint32_t order[] = { 0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12 };
          cell = 4;
          px = order[frame & 15u] & 3u;
          py = order[frame & 15u] >> 2u;
        } else {
          cell = 2;
          const uint32_t i = frame & 3u;
          px = i == 1u || i == 2u ? 1u : 0u;
          py = i == 1u || i == 3u ? 1u : 0u;
        }
      }
      const uint32_t x = std::min(uint32_t(u * float(width)), width - 1);
      const uint32_t y = std::min(uint32_t(v * float(height)), height - 1);
      const uint32_t maxCellX = (width - 1 - px) / cell;
      const uint32_t maxCellY = (height - 1 - py) / cell;
      return { std::min(x / cell, maxCellX) * cell + px, std::min(y / cell, maxCellY) * cell + py };
    }

    const numos::Binding* findBinding(const numos::DispatchDesc& desc, uint32_t slot) {
      for (uint32_t i = 0; i < desc.bindingCount; ++i) {
        if (desc.bindings[i].slot == slot) {
          return &desc.bindings[i];
        }
      }
      return nullptr;
    }

    bool fits(const Rc<DxvkBuffer>& buffer, const numos::Binding& binding, size_t size) {
      return buffer != nullptr && binding.offset <= buffer->info().size &&
        size <= buffer->info().size - binding.offset && size <= binding.size;
    }

    std::string surfaceSnapshot(const SceneManager& scene, const RtInstance* instance,
      uint32_t slot, uint32_t firstIndexOffset) {
      if (instance == nullptr) {
        return "identity=unresolved nullOrderedSlot=1";
      }
      const auto& s = instance->surface;
      const auto& a = s.alphaState;
      const auto* owner = const_cast<RtInstance*>(instance)->getPrimInstanceOwner().getReplacementInstance();
      const bool proceduralGrass = owner != nullptr && owner->proceduralGrass;
      Matrix4 effective = s.objectToWorld;
      bool transformResolved = !proceduralGrass;
      if (s.instancesToObject && !proceduralGrass) {
        const size_t index = size_t(slot) - s.surfaceIndexOfFirstInstance;
        transformResolved = s.surfaceIndexOfFirstInstance <= slot && index < s.instancesToObject->size();
        if (transformResolved) {
          effective = effective * (*s.instancesToObject)[index];
        }
      }
      std::ostringstream line;
      line << std::setprecision(9) << "identity=cpuOrderedSurface instanceId=" << instance->getId()
        << " cacheIdentity=" << instance->getCacheIdentity() << " vectorId=" << instance->getVectorIdx()
        << " cpuSurfaceIndex=" << instance->getSurfaceIndex() << " associatedGeometryHash=" << std::hex << s.associatedGeometryHash
        << " originalTextureHash=" << s.originalTextureHash << " materialHash=" << instance->getMaterialHash()
        << " materialDataHash=" << instance->getMaterialDataHash() << " indexHash=" << instance->getIndexHash()
        << " texcoordHash=" << instance->getTexcoordHash() << std::dec
        << " firstIndex=" << s.firstIndex << " splitFirstIndexOffset=" << firstIndexOffset
        << " materialCacheIndex=" << s.surfaceMaterialIndex << " materialType=" << uint32_t(instance->getMaterialType())
        << " albedoSlot=" << instance->getAlbedoOpacityTextureIndex() << " samplerSlot=" << instance->getSamplerIndex()
        << " secondaryOpacitySlot=" << instance->getSecondaryOpacityTextureIndex()
        << " hidden=" << instance->isHidden() << " cameraMain=" << instance->isCameraRegistered(CameraType::Main)
        << " cameraSky=" << instance->isCameraRegistered(CameraType::Sky) << " category=" << instance->getCategoryFlags().raw()
        << " geometryFlags=" << instance->getGeometryFlags() << " instanceFlags=" << instance->getVkInstance().flags
        << " instanceMask=" << instance->getVkInstance().mask
        << " alphaTest=" << uint32_t(a.alphaTestType) << " alphaRef=" << uint32_t(a.alphaTestReferenceValue)
        << " fullyOpaque=" << a.isFullyOpaque << " blendingDisabled=" << a.isBlendingDisabled
        << " blend=" << uint32_t(a.blendType) << " emissiveBlend=" << a.emissiveBlend
        << " particle=" << a.isParticle << " decal=" << a.isDecal
        << " clipEnabled=" << s.isClipPlaneEnabled << " clip=" << s.clipPlane.x << ',' << s.clipPlane.y << ',' << s.clipPlane.z << ',' << s.clipPlane.w
        << " instanced=" << bool(s.instancesToObject) << " instancesFromHost=" << s.instancesFromHost
        << " proceduralGrass=" << proceduralGrass << " prototypeIdentityOnly=" << proceduralGrass
        << " transformResolved=" << transformResolved << " gpuTransformVerified=0 transform=";
      for (uint32_t c = 0; c < 4; ++c) {
        for (uint32_t r = 0; r < 4; ++r) {
          line << effective[c][r] << ',';
        }
      }
      const auto* material = scene.getSurfaceMaterialOrNull(s.surfaceMaterialIndex);
      if (material != nullptr) {
        line << " resolvedMaterialHash=" << std::hex << material->getHash() << std::dec;
        if (material->getType() == RtSurfaceMaterialType::Opaque) {
          const auto color = material->getOpaqueSurfaceMaterial().getAlbedoOpacityConstant();
          line << " albedoOpacityConstant=" << color.x << ',' << color.y << ',' << color.z << ',' << color.w;
        }
      }
      const auto* blas = instance->getBlas();
      if (blas != nullptr) {
        line << " drawId=" << blas->input.drawCallID << " drawCamera=" << uint32_t(blas->input.cameraType)
          << " sourceGeometryHash=" << std::hex << blas->input.getGeometryData().getHashForRule<rules::FullGeometryHash>()
          << " legacyDrawHash=" << blas->input.getGeometryData().legacyDrawHash
          << " legacyGeometryHash=" << blas->input.getGeometryData().legacyGeometryHash
          << " legacyMaterialHash=" << blas->input.getGeometryData().legacyMaterialHash
          << " sourceMaterialHash=" << blas->input.getMaterialData().getHash()
          << " sourceTextureHash=" << blas->input.getMaterialData().getColorTexture().getImageHash()
          << " sourceTexture2Hash=" << blas->input.getMaterialData().getColorTexture2().getImageHash() << std::dec
          << " sourceVertexShader=" << blas->input.usesVertexShader << " raytracedRenderTarget=" << blas->input.isUsingRaytracedRenderTarget
          << " primitiveCount=" << blas->modifiedGeometryData.calculatePrimitiveCount();
      }
      return line.str();
    }
  }

  struct NumosCloudProbe::Impl {
    struct Upload {
      uint64_t handle;
      size_t offset;
      std::vector<uint8_t> bytes;
    };
    DxvkDevice* device;
    bool enabled = false, sawAbsent = false, armed = false, captured = false, pending = false, active = false;
    bool armWarning = false, haveFrame = false;
    uint64_t frame = 0, captureFrame = 0;
    uint32_t dispatches = 0, pixelCount = kPixels;
    std::filesystem::path armFile, outputDirectory;
    std::chrono::steady_clock::time_point nextPoll {};
    std::array<std::array<float, 2>, kPixels> uv {{{.35f,.3f},{.45f,.3f},{.55f,.3f},{.65f,.3f},{.35f,.55f},{.45f,.55f},{.55f,.55f},{.65f,.55f}}};
    std::vector<Upload> uploads;
    std::vector<std::string> surfaceMetadata;
    std::array<uint32_t, 64> header {};
    Rc<DxvkBuffer> storage, readback, argsBuffer, cameraBuffer;
    Rc<DxvkGpuQuery> ready;
    size_t argsOffset = 0, cameraOffset = 0;
    numos::AtmosphereArgs args {};
    numos::Camera camera {};

    explicit Impl(DxvkDevice* pDevice) : device(pDevice) {
      enabled = env::getEnvVar("RTX_NUMOS_CLOUD_PROBE") == "1";
      if (!enabled) {
        return;
      }
      std::wstring path;
      if (!wideEnvironment(L"RTX_NUMOS_CLOUD_PROBE_ARM_FILE", path)) {
        enabled = false;
        logProbe("disabled reason=missingOrInvalidArmFile");
        return;
      }
      armFile = path;
      if (!armFile.is_absolute()) {
        enabled = false;
        logProbe("disabled reason=armFileMustBeAbsolute");
        return;
      }
      if (!wideEnvironment(L"RTX_NUMOS_CLOUD_PROBE_DIRECTORY", path)) {
        enabled = false;
        logProbe("disabled reason=missingOutputDirectory");
        return;
      }
      outputDirectory = path;
      std::error_code directoryError;
      if (!outputDirectory.is_absolute() || !std::filesystem::is_directory(outputDirectory, directoryError)) {
        enabled = false;
        logProbe("disabled reason=outputDirectoryMustExistAndBeAbsolute");
        return;
      }
      std::string pixels = env::getEnvVar("RTX_NUMOS_CLOUD_PROBE_PIXELS");
      if (!pixels.empty()) {
        std::replace(pixels.begin(), pixels.end(), ';', ' ');
        std::replace(pixels.begin(), pixels.end(), ',', ' ');
        std::istringstream input(pixels);
        std::vector<float> values;
        float value;
        while (input >> value) {
          values.push_back(value);
        }
        if (!input.eof() || values.empty() || values.size() % 2 || values.size() > 2 * kPixels) {
          enabled = false;
        }
        pixelCount = uint32_t(values.size() / 2);
        for (uint32_t i = 0; enabled && i < pixelCount; ++i) {
          const float x = values[2 * i], y = values[2 * i + 1];
          if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 || x >= 1 || y >= 1) {
            enabled = false;
            break;
          }
          uv[i] = {x, y};
        }
        if (!enabled) {
          logProbe("disabled reason=invalidNormalizedPixels");
          return;
        }
      }
      logProbe("enabled ABI=1 oneFrame=1 waitingForObservedMarkerAbsence=1");
    }

    std::filesystem::path capturePath(const char* suffix) const {
      return outputDirectory / ("cloud-probe-frame-" + std::to_string(captureFrame) + suffix);
    }

    void writeSnapshot() {
      std::ofstream out(capturePath(".metadata.json"), std::ios::binary);
      out << "{\"abi\":1,\"deviceFrame\":" << captureFrame << ",\"cloudFrame\":" << args.cloudRenderFrameIdx
        << ",\"gpuTransformVerified\":false,\"surfaces\":[";
      for (size_t i = 0; i < surfaceMetadata.size(); ++i) {
        if (i) { out << ','; }
        out << "{\"surfaceSlot\":" << i << ",\"detail\":" << jsonString(surfaceMetadata[i]) << '}';
      }
      out << "]}\n";
      out.flush();
      logProbe(str::format("metadataWritten deviceFrame=", captureFrame, " success=", bool(out), " surfaces=", surfaceMetadata.size()));
    }

    void pollArm() {
      const auto now = std::chrono::steady_clock::now();
      if (armed || captured || now < nextPoll) {
        return;
      }
      nextPoll = now + std::chrono::milliseconds(250);
      const DWORD attributes = GetFileAttributesW(armFile.c_str());
      const DWORD error = attributes == INVALID_FILE_ATTRIBUTES ? GetLastError() : ERROR_SUCCESS;
      if (attributes == INVALID_FILE_ATTRIBUTES && (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND)) {
        if (!sawAbsent) {
          logProbe("armFileAbsent waitingForNewFile=1");
        }
        sawAbsent = true;
      } else if (attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY) && sawAbsent) {
        armed = true;
        logProbe(str::format("armed deviceFrame=", frame, " markerLeftUnchanged=1"));
      } else if (!armWarning) {
        armWarning = true;
        logProbe(str::format("armWaiting sawAbsent=", sawAbsent, " attributes=", attributes, " error=", error));
      }
    }

    void readResult() {
      const auto* bytes = static_cast<const uint8_t*>(readback->mapPtr(0));
      const auto* words = reinterpret_cast<const uint32_t*>(bytes);
      const bool argsEqual = std::memcmp(bytes + kArgsOffset, &args, sizeof(args)) == 0;
      const bool cameraEqual = std::memcmp(bytes + kCameraOffset, &camera, sizeof(camera)) == 0;
      const bool headerEqual = std::memcmp(words, header.data(), 24 * sizeof(uint32_t)) == 0;
      std::ofstream raw(capturePath(".bin"), std::ios::binary);
      raw.write(reinterpret_cast<const char*>(bytes), kProbeBytes);
      raw.flush();
      std::ofstream constants(capturePath(".constants.bin"), std::ios::binary);
      constants.write(reinterpret_cast<const char*>(bytes + kArgsOffset), kReadBytes - kArgsOffset);
      constants.flush();
      logProbe(str::format("binaryWritten deviceFrame=", captureFrame, " probeBytes=", kProbeBytes,
        " probeSuccess=", bool(raw), " constantsBytes=", kReadBytes - kArgsOffset, " constantsSuccess=", bool(constants),
        " cameraOffsetInConstants=", kCameraOffset - kArgsOffset));
      logProbe(str::format("readback deviceFrame=", captureFrame, " readyAt=", frame, " cloudFrame=", args.cloudRenderFrameIdx,
        " dispatches=", dispatches, " headerEqual=", headerEqual, " argsEqual=", argsEqual, " cameraEqual=", cameraEqual,
        " cpuArgsHash=", hashBytes(&args, sizeof(args)), " gpuArgsHash=", hashBytes(bytes + kArgsOffset, sizeof(args)),
        " cpuCameraHash=", hashBytes(&camera, sizeof(camera)), " gpuCameraHash=", hashBytes(bytes + kCameraOffset, sizeof(camera))));
      // Raw constant words retain all frame-specific settings and matrices without a lossy text schema.
      for (const auto& block : { std::make_pair(kArgsOffset, sizeof(args)), std::make_pair(kCameraOffset, sizeof(camera)) }) {
        std::ostringstream line;
        line << "deviceFrame=" << captureFrame << (block.first == kArgsOffset ? " gpuArgsWords=" : " gpuCameraWords=") << std::hex;
        for (size_t i = 0; i < block.second / sizeof(uint32_t); ++i) {
          uint32_t word;
          std::memcpy(&word, bytes + block.first + i * sizeof(word), sizeof(word));
          line << word << ',';
        }
        logProbe(line.str());
      }
      std::unordered_set<uint32_t> mapped;
      for (uint32_t p = 0; p < pixelCount; ++p) {
        logProbe(str::format("deviceFrame=", captureFrame, " pixelSlot=", p, " pixel=", words[8 + p * 2], ",", words[9 + p * 2],
          " claimed=", words[24 + p], " total=", words[32 + p], " hits=", words[40 + p], " misses=", words[48 + p], " beyondPrefix=", words[56 + p]));
        for (uint32_t r = 0; r < 32; ++r) {
          if (r < 30 && r >= std::min(words[32 + p], 30u)) {
            continue;
          }
          if ((r == 30 && !words[40 + p]) || (r == 31 && !words[48 + p])) {
            continue;
          }
          const auto* v = words + 64 + p * 1024 + r * 32;
          const uint64_t surface = uint64_t(v[16] & CUSTOM_INDEX_SURFACE_MASK) + v[18];
          std::ostringstream line;
          line << std::setprecision(9) << "deviceFrame=" << captureFrame << " pixelSlot=" << p << " record=" << r
            << " hit=" << v[0] << " stage=" << v[1] << " query=" << v[2] << " cloudFrame=" << v[3]
            << " origin=" << floatWord(v[8]) << ',' << floatWord(v[9]) << ',' << floatWord(v[10])
            << " direction=" << floatWord(v[12]) << ',' << floatWord(v[13]) << ',' << floatWord(v[14])
            << " hitT=" << floatWord(v[15]) << " customId=" << v[16] << " instanceIndex=" << v[17]
            << " geometryIndex=" << v[18] << " primitiveIndex=" << v[19] << " surfaceSlot=" << surface << " raw=" << std::hex;
          for (uint32_t i = 0; i < 32; ++i) {
            line << v[i] << ',';
          }
          logProbe(line.str());
          if (v[0] && surface <= UINT32_MAX && mapped.insert(uint32_t(surface)).second) {
            const std::string identity = surface < surfaceMetadata.size() ? surfaceMetadata[size_t(surface)] : "identity=unresolved outsideCpuSnapshot=1";
            logProbe(str::format("deviceFrame=", captureFrame, " surfaceSlot=", surface, " ", identity));
          }
        }
      }
      logProbe(str::format("complete deviceFrame=", captureFrame, " mappedHitSurfaces=", mapped.size(), " noRearm=1"));
      surfaceMetadata.clear();
    }
  };

  NumosCloudProbe::NumosCloudProbe(DxvkDevice* device) : m_impl(std::make_unique<Impl>(device)) { }
  NumosCloudProbe::~NumosCloudProbe() {
    if (m_impl->enabled) {
      logProbe(str::format("shutdown captured=", m_impl->captured, " pendingDropped=", m_impl->pending));
    }
  }
  bool NumosCloudProbe::enabled() const { return m_impl->enabled; }

  void NumosCloudProbe::beginFrame(DxvkContext&, uint64_t frame) {
    auto& p = *m_impl;
    if (!p.enabled) {
      return;
    }
    if (p.haveFrame && p.frame == frame) {
      return;
    }
    p.haveFrame = true;
    p.frame = frame;
    if (p.pending && frame >= p.captureFrame && frame - p.captureFrame >= 8 && !p.readback->isInUse(DxvkAccess::Read)) {
      DxvkQueryData data;
      const auto status = p.ready->getData(data);
      if (status == DxvkGpuQueryStatus::Available) {
        p.readResult();
        p.pending = false;
      } else if (status != DxvkGpuQueryStatus::Pending) {
        logProbe("readbackDropped reason=queryUnavailable");
        p.pending = false;
      }
    }
    p.uploads.clear();
    p.pollArm();
  }

  void NumosCloudProbe::noteConstants(uint64_t handle, size_t offset, const void* data, size_t size) {
    auto& p = *m_impl;
    if (!p.enabled || !p.armed || p.captured || data == nullptr || p.uploads.size() >= 256 ||
      (size != sizeof(numos::AtmosphereArgs) && size != sizeof(numos::Camera))) {
      return;
    }
    const auto* bytes = static_cast<const uint8_t*>(data);
    p.uploads.push_back({handle, offset, std::vector<uint8_t>(bytes, bytes + size)});
  }

  Rc<DxvkBuffer> NumosCloudProbe::beforeCloudDispatch(DxvkContext& ctx, const numos::DispatchDesc& desc,
    const Rc<DxvkBuffer>& argsBuffer, const Rc<DxvkBuffer>& cameraBuffer,
    const SceneManager& scene, uint32_t width, uint32_t height) {
    auto& p = *m_impl;
    if (!p.enabled || !p.armed || p.captured || width < 4 || height < 4) {
      return nullptr;
    }
    if (p.active) {
      ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      ++p.dispatches;
      return p.storage;
    }
    const auto* a = findBinding(desc, 0);
    const auto* c = findBinding(desc, 17);
    if (a == nullptr || c == nullptr || !fits(argsBuffer, *a, sizeof(p.args)) || !fits(cameraBuffer, *c, sizeof(p.camera))) {
      logProbe("captureDeferred reason=missingConstantsBinding");
      return nullptr;
    }
    auto findUpload = [&](const numos::Binding& b, void* dst, size_t size) {
      for (auto it = p.uploads.rbegin(); it != p.uploads.rend(); ++it) {
        if (it->handle == b.resource && it->offset == b.offset && it->bytes.size() == size) {
          std::memcpy(dst, it->bytes.data(), size);
          return true;
        }
      }
      return false;
    };
    if (!findUpload(*a, &p.args, sizeof(p.args)) || !findUpload(*c, &p.camera, sizeof(p.camera))) {
      logProbe("captureDeferred reason=missingSameFrameCpuUpload");
      return nullptr;
    }
    const auto& ordered = scene.getAccelManager().getOrderedInstances();
    const auto& offsets = scene.getAccelManager().getOrderedInstanceFirstIndexOffsets();
    p.surfaceMetadata.reserve(std::min<size_t>(ordered.size(), kMaxSurfaces));
    for (size_t i = 0; i < std::min<size_t>(ordered.size(), kMaxSurfaces); ++i) {
      p.surfaceMetadata.push_back(surfaceSnapshot(scene, ordered[i], uint32_t(i), i < offsets.size() ? offsets[i] : 0));
    }
    p.header = {};
    p.header[0] = kMagic;
    p.header[1] = p.header[2] = 1;
    p.header[3] = p.args.cloudRenderFrameIdx;
    p.header[4] = p.pixelCount;
    p.header[5] = p.header[6] = 32;
    for (uint32_t i = 0; i < p.pixelCount; ++i) {
      const auto xy = snapPixel(p.uv[i][0], p.uv[i][1], width, height, p.args.cloudRenderFrameIdx, p.args.cloudScreenInterleave);
      p.header[8 + i * 2] = xy[0];
      p.header[9 + i * 2] = xy[1];
    }
    DxvkBufferCreateInfo info {};
    info.size = kProbeBytes + 256;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    p.storage = p.device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "NUMOS cloud hit probe");
    info.size = kReadBytes + 256;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    p.readback = p.device->createBuffer(info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      DxvkMemoryStats::Category::RTXBuffer, "NUMOS cloud hit probe readback");
    p.ready = p.device->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0, 0);
    std::array<uint32_t, kWords> initial {};
    std::copy(p.header.begin(), p.header.end(), initial.begin());
    ctx.writeToBuffer(p.storage, 0, sizeof(initial), initial.data());
    ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    p.argsBuffer = argsBuffer;
    p.cameraBuffer = cameraBuffer;
    p.argsOffset = a->offset;
    p.cameraOffset = c->offset;
    p.captureFrame = p.frame;
    p.writeSnapshot();
    p.active = true;
    p.dispatches = 1;
    logProbe(str::format("capture deviceFrame=", p.frame, " cloudFrame=", p.args.cloudRenderFrameIdx,
      " extent=", width, ",", height, " interleave=", p.args.cloudScreenInterleave, " snapshots=", p.surfaceMetadata.size(),
      " orderedSurfaces=", ordered.size(), " truncated=", ordered.size() > kMaxSurfaces,
      " sameFrameSnapshot=1 gpuOwnedSlotsMayDiffer=1"));
    return p.storage;
  }

  void NumosCloudProbe::endFrame(DxvkContext& ctx) {
    auto& p = *m_impl;
    if (!p.active) {
      return;
    }
    ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    ctx.copyBuffer(p.readback, 0, p.storage, 0, kProbeBytes);
    ctx.copyBuffer(p.readback, kArgsOffset, p.argsBuffer, p.argsOffset, sizeof(p.args));
    ctx.copyBuffer(p.readback, kCameraOffset, p.cameraBuffer, p.cameraOffset, sizeof(p.camera));
    ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    ctx.writeTimestamp(p.ready);
    p.pending = p.captured = true;
    p.active = false;
    p.argsBuffer = nullptr;
    p.cameraBuffer = nullptr;
    logProbe(str::format("queued deviceFrame=", p.captureFrame, " dispatches=", p.dispatches, " delayFrames=8"));
  }
}
