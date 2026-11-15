#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <Windows.h>

#include "numos/gpu.h"
#include "numos/lib/aerial_perspective_binding_indices.h"
#include "numos/lib/aerial_perspective_light.h"
#include "numos/lib/atmosphere_args.h"

#include "rtx_numos_haze_probe.h"
#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../../util/log/log.h"
#include "../../util/util_env.h"
#include "../../util/util_string.h"

namespace dxvk {
  namespace {
    constexpr uint32_t kFrameLimit = 300;
    constexpr uint32_t kLightLimit = 4096;
    constexpr uint32_t kColumnWords = 17;
    constexpr uint32_t kColumnMarker = 0x80000000u;
    constexpr uint32_t kSlotCount = 4;
    constexpr uint32_t kReadDelayFrames = 8;
    constexpr size_t kLightOffset = (sizeof(numos::AtmosphereArgs) + 15) & ~size_t(15);
    static_assert(sizeof(numos::AerialPerspectiveLight) == 64, "NUMOS AP light layout changed");
    static_assert(AERIAL_PERSPECTIVE_LIGHT_CLUSTER_STRIDE == kColumnWords, "NUMOS AP column layout changed");

    // Recycled timestamp queries can still report the old result before the init
    // command buffer resets them. Resource retirement additionally proves that the
    // current copy commands have finished; neither condition blocks the CPU.
    bool canPollReadback(uint64_t currentFrame, uint64_t capturedFrame, bool bufferInUse) {
      return currentFrame >= capturedFrame && currentFrame - capturedFrame >= kReadDelayFrames && !bufferInUse;
    }

    enum class ArmFileStatus { Missing, File, Other, Error };
    enum class ArmFileEvent { None, Absent, Armed };

    // A pre-existing marker must disappear before a later creation can arm this instance.
    struct ArmFileGate {
      using Clock = std::chrono::steady_clock;
      Clock::time_point nextPoll = Clock::time_point::min();
      bool sawAbsent = false;
      bool armed = false;

      bool pollDue(Clock::time_point now) {
        if (armed || now < nextPoll) {
          return false;
        }
        nextPoll = now + std::chrono::milliseconds(250);
        return true;
      }

      ArmFileEvent observe(ArmFileStatus status) {
        if (armed) {
          return ArmFileEvent::None;
        }
        if (status == ArmFileStatus::Missing && !sawAbsent) {
          sawAbsent = true;
          return ArmFileEvent::Absent;
        }
        if (status == ArmFileStatus::File && sawAbsent) {
          armed = true;
          return ArmFileEvent::Armed;
        }
        return ArmFileEvent::None;
      }
    };

    // Read UTF-16 with a sized buffer so a long marker path cannot silently become an empty gate.
    bool readArmFileEnvironment(std::wstring& value) {
      constexpr const wchar_t* name = L"RTX_NUMOS_HAZE_PROBE_ARM_FILE";
      SetLastError(ERROR_SUCCESS);
      const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
      if (!size) {
        const DWORD error = GetLastError();
        value.clear();
        return error == ERROR_ENVVAR_NOT_FOUND || error == ERROR_SUCCESS;
      }
      value.resize(size);
      const DWORD copied = GetEnvironmentVariableW(name, value.data(), size);
      if (!copied || copied >= size) {
        value.clear();
        return false;
      }
      value.resize(copied);
      return true;
    }

    ArmFileStatus readArmFileStatus(const std::filesystem::path& path, DWORD& error) {
      const DWORD attributes = GetFileAttributesW(path.c_str());
      if (attributes == INVALID_FILE_ATTRIBUTES) {
        error = GetLastError();
        return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND
          ? ArmFileStatus::Missing : ArmFileStatus::Error;
      }
      error = ERROR_SUCCESS;
      return attributes & FILE_ATTRIBUTE_DIRECTORY ? ArmFileStatus::Other : ArmFileStatus::File;
    }

    bool probeFrameEligible(uint64_t frame, uint64_t start, uint32_t attempts, bool manualArm, bool armed) {
      return (!manualArm || armed) && frame >= start && attempts < kFrameLimit;
    }

    uint32_t screenTile(float coordinate, uint32_t froxelSize, uint32_t tileCount) {
      const uint32_t froxel = uint32_t(coordinate * froxelSize);
      return std::min(froxel / AERIAL_PERSPECTIVE_LIGHT_TILE_SIZE, tileCount - 1);
    }

    struct DecodedColumn {
      bool valid = true;
      bool grouped = false;
      uint32_t candidateCount = 0;
      uint32_t groupSize = 1;
      std::vector<uint32_t> indices;
    };

    // Group bits conservatively represent every light in the group, not K exact indices.
    DecodedColumn decodeColumn(const std::array<uint32_t, kColumnWords>& words, uint32_t lightCount) {
      DecodedColumn out;
      out.grouped = (words[0] & kColumnMarker) != 0;
      out.candidateCount = words[0] & ~kColumnMarker;
      if (!out.grouped) {
        out.valid = out.candidateCount <= 16;
        for (uint32_t i = 0; i < std::min(out.candidateCount, 16u); ++i) {
          if (words[i + 1] < lightCount) {
            out.indices.push_back(words[i + 1]);
          } else {
            out.valid = false;
          }
        }
        return out;
      }
      out.groupSize = std::max((lightCount + 511u) / 512u, 1u);
      if (out.candidateCount == 0) {
        return out;
      }
      out.valid = out.candidateCount <= lightCount;
      uint32_t groups = 0;
      for (uint32_t bit = 0; bit < 512; ++bit) {
        if ((words[1 + bit / 32] & (1u << (bit % 32))) == 0) {
          continue;
        }
        const uint32_t first = bit * out.groupSize;
        if (first >= lightCount) {
          out.valid = false;
          continue;
        }
        ++groups;
        for (uint32_t i = first; i < std::min(first + out.groupSize, lightCount); ++i) {
          out.indices.push_back(i);
        }
      }
      out.valid &= groups <= out.candidateCount && out.candidateCount <= out.indices.size();
      return out;
    }

    uint64_t bytesHash(const void* data, size_t size) {
      const auto* bytes = static_cast<const uint8_t*>(data);
      uint64_t hash = 14695981039346656037ull;
      for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ull;
      }
      return hash;
    }

    const numos::Binding* findBinding(const numos::DispatchDesc& desc, uint32_t slot) {
      for (uint32_t i = 0; i < desc.bindingCount; ++i) {
        if (desc.bindings[i].slot == slot) {
          return &desc.bindings[i];
        }
      }
      return nullptr;
    }

    bool fits(const numos::Binding& binding, const Rc<DxvkBuffer>& buffer, size_t offset, size_t size) {
      if (buffer == nullptr || binding.offset > buffer->info().size) {
        return false;
      }
      const size_t available = std::min<size_t>(buffer->info().size - binding.offset, binding.size);
      return offset <= available && size <= available - offset;
    }

    void logProbe(const std::string& text) {
      Logger::info("[NumosHazeProbe] " + text);
    }
  }

  struct NumosHazeProbe::Impl {
    struct ConstantUpload {
      uint64_t handle;
      size_t offset;
      numos::AtmosphereArgs args;
    };
    struct Slot {
      Rc<DxvkBuffer> readback;
      Rc<DxvkGpuQuery> ready;
      bool pending = false;
      uint64_t frameId = 0;
      uint64_t uploadFrame = 0;
      numos::AtmosphereArgs args {};
      std::vector<uint8_t> cpuLights;
      std::array<std::array<uint32_t, 2>, 3> tiles {};
      size_t lightBytes = 0;
      size_t columnOffset = 0;
    };

    DxvkDevice* device;
    bool enabled = false;
    bool eligible = false;
    bool sawCull = false;
    bool haveFrame = false;
    bool summarized = false;
    uint64_t frameId = 0;
    uint64_t startFrame = 1800;
    bool manualArm = false;
    bool armErrorReported = false;
    bool armExistingReported = false;
    std::filesystem::path armFile;
    ArmFileGate armGate;
    uint64_t lightHandle = 0;
    uint64_t uploadFrame = 0;
    uint32_t attempts = 0;
    uint32_t queued = 0;
    uint32_t completed = 0;
    uint32_t dropped = 0;
    uint32_t noCull = 0;
    std::array<std::array<float, 2>, 3> normalizedTiles {{{ .25f, .5f }, { .5f, .5f }, { .75f, .5f }}};
    std::vector<ConstantUpload> constants;
    std::vector<uint8_t> cpuLights;
    std::array<Slot, kSlotCount> slots;

    explicit Impl(DxvkDevice* pDevice) : device(pDevice) {
      enabled = env::getEnvVar("RTX_NUMOS_HAZE_PROBE") == "1";
      if (!enabled) {
        return;
      }
      const std::string start = env::getEnvVar("RTX_NUMOS_HAZE_PROBE_START_FRAME");
      if (!start.empty()) {
        char* end = nullptr;
        const auto parsed = std::strtoull(start.c_str(), &end, 10);
        if (end != start.c_str() && *end == '\0' && start[0] != '-') {
          startFrame = parsed;
        } else {
          logProbe("invalid START_FRAME; using 1800");
        }
      }
      std::string tileText = env::getEnvVar("RTX_NUMOS_HAZE_PROBE_TILES");
      if (!tileText.empty()) {
        std::replace(tileText.begin(), tileText.end(), ',', ' ');
        std::replace(tileText.begin(), tileText.end(), ';', ' ');
        std::istringstream input(tileText);
        auto parsed = normalizedTiles;
        bool valid = true;
        for (auto& tile : parsed) {
          valid &= bool(input >> tile[0] >> tile[1]);
          valid &= std::isfinite(tile[0]) && std::isfinite(tile[1]) &&
                   tile[0] >= 0.f && tile[0] <= 1.f && tile[1] >= 0.f && tile[1] <= 1.f;
        }
        std::string extra;
        valid &= !(input >> extra);
        if (valid) {
          normalizedTiles = parsed;
        } else {
          logProbe("invalid TILES; using three default normalized screen points");
        }
      }
      std::wstring marker;
      if (!readArmFileEnvironment(marker)) {
        enabled = false;
        logProbe("disabled reason=armFileEnvironmentReadFailed");
        return;
      }
      if (!marker.empty()) {
        try {
          armFile = std::filesystem::path(marker);
          if (!armFile.is_absolute()) {
            enabled = false;
            logProbe("disabled reason=armFilePathMustBeAbsolute");
            return;
          }
        } catch (const std::exception&) {
          enabled = false;
          logProbe("disabled reason=invalidArmFilePath");
          return;
        }
        manualArm = true;
        logProbe(str::format("waitingForArmFile path=", str::fromws(marker.c_str()),
          " pollMs=250 mustObserveAbsent=1 oneShot=1 startFrameIgnored=", startFrame));
        pollArmFile();
      } else {
        logArmed();
      }
    }

    void logArmed() {
      logProbe(str::format("armed startFrame=", startFrame, " frameLimit=", kFrameLimit,
        " readbackSlots=", kSlotCount, " readDelayFrames=", kReadDelayFrames, " tileCount=3 lightLimit=", kLightLimit,
        " enabledOnlyByEnvironment=1 manualArmFile=", manualArm));
    }

    void pollArmFile() {
      if (!manualArm || !armGate.pollDue(ArmFileGate::Clock::now())) {
        return;
      }
      DWORD error = ERROR_SUCCESS;
      const auto status = readArmFileStatus(armFile, error);
      const auto event = armGate.observe(status);
      if (status == ArmFileStatus::Error && !armErrorReported) {
        armErrorReported = true;
        logProbe(str::format("armFilePollError win32Error=", error, " waiting=1 budgetConsumed=0"));
      }
      if ((status == ArmFileStatus::File || status == ArmFileStatus::Other) &&
          !armGate.sawAbsent && !armExistingReported) {
        armExistingReported = true;
        logProbe("armFileExistsAtStartup waitingForObservedAbsence=1 budgetConsumed=0");
      }
      if (event == ArmFileEvent::Absent) {
        logProbe("armFileAbsent waitingForNewFile=1 budgetConsumed=0");
      } else if (event == ArmFileEvent::Armed) {
        startFrame = frameId;
        logProbe(str::format("armFileTriggered frame=", frameId, " fileLeftUnchanged=1"));
        logArmed();
      }
    }

    void drop(const char* reason) {
      ++dropped;
      logProbe(str::format("drop frame=", frameId, " reason=", reason, " dropped=", dropped));
    }

    void finishFrame() {
      if (eligible && !sawCull) {
        ++noCull;
        logProbe(str::format("frame=", frameId, " status=noCullDispatch historyReset=unknown"));
      }
    }

    void readSlot(Slot& slot) {
      const auto* bytes = static_cast<const uint8_t*>(slot.readback->mapPtr(0));
      numos::AtmosphereArgs gpuArgs {};
      std::memcpy(&gpuArgs, bytes, sizeof(gpuArgs));
      const bool argsEqual = std::memcmp(&gpuArgs, &slot.args, sizeof(gpuArgs)) == 0;
      const bool haveCpu = slot.cpuLights.size() == slot.lightBytes;
      const bool lightsEqual = haveCpu && (slot.lightBytes == 0 || std::memcmp(slot.cpuLights.data(), bytes + kLightOffset, slot.lightBytes) == 0);
      const auto& args = slot.args;
      logProbe(str::format("frame=", slot.frameId, " status=readback readyAtFrame=", frameId,
        " argsEqual=", argsEqual, " lightsEqual=", lightsEqual, " cpuPresent=", haveCpu,
        " uploadFrame=", slot.uploadFrame, " count=", args.apLocalLightCount,
        " grid=", args.apFroxelSizeX, ",", args.apFroxelSizeY, ",", args.apFroxelSlices,
        " tiles=", args.apLocalLightTilesX, ",", args.apLocalLightTilesY,
        " historyReset=", args.apHistoryReset, " historyWeightMin=", args.apHistoryWeightMin,
        " fogExtinctionPerKm=", args.fogExtinctionPerKm, " fogScaleHeightKm=", args.fogScaleHeightKm,
        " range=", args.apFroxelRange, " localShadow=", args.apLocalLightShadows,
        " cpuHash=", haveCpu ? bytesHash(slot.cpuLights.data(), slot.lightBytes) : 0,
        " gpuHash=", bytesHash(bytes + kLightOffset, slot.lightBytes)));
      std::array<DecodedColumn, 3> decoded;
      for (uint32_t t = 0; t < 3; ++t) {
        std::array<uint32_t, kColumnWords> words {};
        std::memcpy(words.data(), bytes + slot.columnOffset + t * sizeof(words), sizeof(words));
        decoded[t] = decodeColumn(words, args.apLocalLightCount);
        std::ostringstream line;
        line << "frame=" << slot.frameId << " tile=" << slot.tiles[t][0] << ',' << slot.tiles[t][1]
             << " tileSlot=" << t << " offsetsFromCpuArgs=1 valid=" << (argsEqual && decoded[t].valid)
             << " mode=" << (decoded[t].grouped ? "grouped" : "legacy")
             << " candidateCount=" << decoded[t].candidateCount << " groupSize=" << decoded[t].groupSize
             << " representedCount=" << decoded[t].indices.size() << " raw=";
        for (uint32_t word : words) {
          line << std::hex << word << ',';
        }
        line << std::dec << " indices=";
        for (uint32_t index : decoded[t].indices) {
          line << index << ',';
        }
        logProbe(line.str());
      }
      // One table per captured frame identifies even lights absent from all three columns.
      std::ostringstream table;
      table << std::setprecision(9) << "frame=" << slot.frameId << " gpuLights=";
      for (uint32_t i = 0; i < args.apLocalLightCount; ++i) {
        numos::AerialPerspectiveLight light {};
        std::memcpy(&light, bytes + kLightOffset + i * sizeof(light), sizeof(light));
        uint32_t tileMask = 0;
        for (uint32_t t = 0; t < 3; ++t) {
          if (argsEqual && decoded[t].valid && std::find(decoded[t].indices.begin(), decoded[t].indices.end(), i) != decoded[t].indices.end()) {
            tileMask |= 1u << t;
          }
        }
        table << i << ':' << light.position.x << ',' << light.position.y << ',' << light.position.z
              << '/' << light.radiance.x << ',' << light.radiance.y << ',' << light.radiance.z
              << '/' << light.equivalentRadius << '/' << light.influenceRadius
              << '/' << light.coneAxis.x << ',' << light.coneAxis.y << ',' << light.coneAxis.z
              << '/' << light.cosConeAngle << '/' << tileMask << ';';
      }
      logProbe(table.str());
      if (haveCpu && !lightsEqual) {
        std::ostringstream cpuTable;
        cpuTable << std::setprecision(9) << "frame=" << slot.frameId << " cpuLightsMismatch=";
        for (uint32_t i = 0; i < args.apLocalLightCount; ++i) {
          numos::AerialPerspectiveLight light {};
          std::memcpy(&light, slot.cpuLights.data() + i * sizeof(light), sizeof(light));
          cpuTable << i << ':' << light.position.x << ',' << light.position.y << ',' << light.position.z
                   << '/' << light.radiance.x << ',' << light.radiance.y << ',' << light.radiance.z << ';';
        }
        logProbe(cpuTable.str());
      }
      ++completed;
    }

    void poll() {
      for (Slot& slot : slots) {
        if (!slot.pending ||
            !canPollReadback(frameId, slot.frameId, slot.readback->isInUse(DxvkAccess::Read))) {
          continue;
        }
        DxvkQueryData data;
        const auto status = slot.ready->getData(data);
        if (status == DxvkGpuQueryStatus::Pending) {
          continue;
        }
        if (status == DxvkGpuQueryStatus::Available) {
          readSlot(slot);
        } else {
          ++dropped;
          logProbe(str::format("drop frame=", slot.frameId, " reason=queryUnavailable"));
        }
        slot.pending = false;
        slot.cpuLights.clear();
      }
    }
  };

  NumosHazeProbe::NumosHazeProbe(DxvkDevice* device) : m_impl(std::make_unique<Impl>(device)) { }

  NumosHazeProbe::~NumosHazeProbe() {
    auto& p = *m_impl;
    if (p.enabled) {
      p.finishFrame();
      const auto pending = std::count_if(p.slots.begin(), p.slots.end(), [](const auto& slot) { return slot.pending; });
      logProbe(str::format("shutdown captured=", p.queued, " completed=", p.completed,
        " dropped=", p.dropped, " noCull=", p.noCull, " pendingDroppedAtShutdown=", pending));
    }
  }

  void NumosHazeProbe::beginFrame(DxvkContext&, uint64_t frameId) {
    auto& p = *m_impl;
    if (!p.enabled) {
      return;
    }
    if (p.haveFrame && p.frameId == frameId) {
      return;
    }
    p.finishFrame();
    p.frameId = frameId;
    p.haveFrame = true;
    p.poll();
    p.constants.clear();
    p.pollArmFile();
    p.eligible = probeFrameEligible(frameId, p.startFrame, p.attempts, p.manualArm, p.armGate.armed);
    p.sawCull = false;
    if (p.eligible) {
      ++p.attempts;
    } else if (p.attempts == kFrameLimit && !p.summarized &&
               std::none_of(p.slots.begin(), p.slots.end(), [](const auto& slot) { return slot.pending; })) {
      p.summarized = true;
      logProbe(str::format("complete attemptedFrames=", p.attempts, " captured=", p.queued,
        " completed=", p.completed, " dropped=", p.dropped, " noCull=", p.noCull));
      p.cpuLights.clear();
    }
  }

  void NumosHazeProbe::trackBuffer(uint64_t handle, const char* debugName) {
    auto& p = *m_impl;
    if (p.enabled && debugName != nullptr && std::strcmp(debugName, "Atmosphere AP Lights") == 0) {
      p.lightHandle = handle;
      p.cpuLights.clear();
    }
  }

  void NumosHazeProbe::forgetBuffer(uint64_t handle) {
    auto& p = *m_impl;
    if (p.lightHandle == handle) {
      p.lightHandle = 0;
      p.cpuLights.clear();
    }
  }

  void NumosHazeProbe::noteConstants(uint64_t handle, size_t offset, const void* data, size_t size) {
    auto& p = *m_impl;
    if (!p.eligible || data == nullptr || size != sizeof(numos::AtmosphereArgs) || p.constants.size() >= 128) {
      return;
    }
    Impl::ConstantUpload upload { handle, offset, {} };
    std::memcpy(&upload.args, data, size);
    p.constants.push_back(upload);
  }

  void NumosHazeProbe::noteBuffer(uint64_t handle, size_t offset, const void* data, size_t size) {
    auto& p = *m_impl;
    if (!p.eligible || handle != p.lightHandle || data == nullptr) {
      return;
    }
    if (offset != 0 || size > kLightLimit * sizeof(numos::AerialPerspectiveLight) ||
        size % sizeof(numos::AerialPerspectiveLight)) {
      p.cpuLights.clear();
      return;
    }
    p.cpuLights.resize(size);
    std::memcpy(p.cpuLights.data(), data, size);
    p.uploadFrame = p.frameId;
  }

  void NumosHazeProbe::afterCull(DxvkContext& ctx, const numos::DispatchDesc& desc,
                                const Rc<DxvkBuffer>& argsBuffer, const Rc<DxvkBuffer>& lightsBuffer,
                                const Rc<DxvkBuffer>& clustersBuffer) {
    auto& p = *m_impl;
    if (!p.eligible || p.sawCull) {
      return;
    }
    p.sawCull = true;
    const auto* argsBinding = findBinding(desc, AP_CULL_ATMOSPHERE_ARGS);
    const auto* lightsBinding = findBinding(desc, AP_CULL_LIGHTS);
    const auto* clustersBinding = findBinding(desc, AP_CULL_CLUSTERS);
    if (argsBinding == nullptr || lightsBinding == nullptr || clustersBinding == nullptr) {
      p.drop("missingBinding");
      return;
    }
    const auto upload = std::find_if(p.constants.rbegin(), p.constants.rend(), [&](const auto& c) {
      return c.handle == argsBinding->resource && c.offset == argsBinding->offset;
    });
    if (upload == p.constants.rend()) {
      p.drop("missingCpuConstants");
      return;
    }
    const auto& args = upload->args;
    if (args.apLocalLightCount > kLightLimit || args.apFroxelSizeX == 0 || args.apFroxelSizeY == 0 ||
        args.apLocalLightTilesX == 0 || args.apLocalLightTilesY == 0 ||
        args.apLocalLightTilesX > 65536 || args.apLocalLightTilesY > 65536 || args.apFroxelSlices > 4096) {
      p.drop("unsupportedGridOrLightCount");
      return;
    }
    const size_t lightBytes = args.apLocalLightCount * sizeof(numos::AerialPerspectiveLight);
    if (!fits(*argsBinding, argsBuffer, 0, sizeof(args)) || !fits(*lightsBinding, lightsBuffer, 0, lightBytes)) {
      p.drop("bufferRange");
      return;
    }
    auto free = std::find_if(p.slots.begin(), p.slots.end(), [](const auto& slot) { return !slot.pending; });
    if (free == p.slots.end()) {
      p.drop("readbackRingFull");
      return;
    }
    auto& slot = *free;
    std::array<size_t, 3> sourceOffsets {};
    for (uint32_t t = 0; t < 3; ++t) {
      const auto x = screenTile(p.normalizedTiles[t][0], args.apFroxelSizeX, args.apLocalLightTilesX);
      const auto y = screenTile(p.normalizedTiles[t][1], args.apFroxelSizeY, args.apLocalLightTilesY);
      slot.tiles[t] = { x, y };
      sourceOffsets[t] = (size_t(args.apLocalLightTilesX) * args.apLocalLightTilesY * args.apFroxelSlices +
                          size_t(y) * args.apLocalLightTilesX + x) * kColumnWords * sizeof(uint32_t);
      if (!fits(*clustersBinding, clustersBuffer, sourceOffsets[t], kColumnWords * sizeof(uint32_t))) {
        p.drop("columnRange");
        return;
      }
    }
    slot.columnOffset = kLightOffset + lightBytes;
    const size_t bytes = slot.columnOffset + 3 * kColumnWords * sizeof(uint32_t);
    if (slot.readback == nullptr || slot.readback->info().size < bytes) {
      DxvkBufferCreateInfo info {};
      info.size = bytes + 256;
      info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
      slot.readback = p.device->createBuffer(info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            DxvkMemoryStats::Category::RTXBuffer, "NUMOS haze probe readback");
    }
    if (slot.ready == nullptr) {
      slot.ready = p.device->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0, 0);
    }
    slot.args = args;
    slot.frameId = p.frameId;
    slot.uploadFrame = p.uploadFrame;
    slot.lightBytes = lightBytes;
    slot.cpuLights.clear();
    if (lightsBinding->resource == p.lightHandle && lightsBinding->offset <= p.cpuLights.size() &&
        lightBytes <= p.cpuLights.size() - lightsBinding->offset) {
      slot.cpuLights.assign(p.cpuLights.begin() + lightsBinding->offset, p.cpuLights.begin() + lightsBinding->offset + lightBytes);
    }
    ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    ctx.copyBuffer(slot.readback, 0, argsBuffer, argsBinding->offset, sizeof(args));
    if (lightBytes) {
      ctx.copyBuffer(slot.readback, kLightOffset, lightsBuffer, lightsBinding->offset, lightBytes);
    }
    for (uint32_t t = 0; t < 3; ++t) {
      ctx.copyBuffer(slot.readback, slot.columnOffset + t * kColumnWords * sizeof(uint32_t), clustersBuffer,
                     clustersBinding->offset + sourceOffsets[t], kColumnWords * sizeof(uint32_t));
    }
    ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    ctx.writeTimestamp(slot.ready);
    slot.pending = true;
    ++p.queued;
  }
}
