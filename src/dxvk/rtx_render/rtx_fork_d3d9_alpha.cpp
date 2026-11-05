// Options, shader constants, dev-menu controls and [D3d9Alpha] counters for the D3D9 alpha / replacement GPU work.
// See rtx_fork_d3d9_alpha.h.

#include "rtx_fork_d3d9_alpha.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "rtx_imgui.h"
#include "rtx_instance_manager.h"
#include "rtx_opacity_micromap_manager.h"
#include "../dxvk_perf_counters.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  namespace {
    // Point-instancer opacity micromap state of one lane, summed over the interval's frames.
    struct PointInstancerLane {
      uint64_t groups = 0;         // per-geometry BLAS groups holding a point instancer
      uint64_t shared = 0;         // groups with more than one instance
      uint64_t bindTried = 0;      // groups that asked for a micromap
      uint64_t bound = 0;          // groups that got one
      uint64_t instances = 0;      // point-instancer instances
      uint64_t placements = 0;     // their placements
      uint64_t users = 0;          // instances that use opacity micromaps
      uint64_t requested = 0;      // instances linked to a micromap request
      uint64_t built = 0;          // instances whose micromap is built
      uint64_t texelsWaiting = 0;  // instances whose request waits for texel footprints
      uint64_t notCandidate = 0;   // users without a request that are not micromap candidates
      uint64_t gates[uint32_t(D3d9Alpha::OmmGate::Count)] = {};  // candidates by the step they stopped at
    };
    constexpr const char* kOmmGateNames[uint32_t(D3d9Alpha::OmmGate::Count)] = {
      "noBlas", "newInstance", "retryWait", "throttled", "noBudget", "notUser", "texturesNotResident", "blacklisted",
      "statsFull", "belowFilter", "registerFailed", "registered" };
    enum Lane : uint32_t { kDirect = 0, kLegacy = 1, kLaneCount = 2 };

    // Written on the CS thread only (scene preparation and draw submission), read by the perf log on the same thread.
    struct Stats {
      uint32_t frames = 0;
      uint64_t sharedOmm[6] = {};        // by SharedOmmOutcome
      uint64_t sharedOmmInstances = 0;   // instances of the BLASes above
      PointInstancerLane pointInstancers[kLaneCount];
      uint64_t pointInstancerMixedGroups = 0;  // groups with instances of both lanes, counted under the direct lane
      uint64_t unlinkedIdle[kLaneCount] = {};
      uint64_t unlinkedIdlePointInstancers[kLaneCount] = {};
    };
    Stats s_stats;
  }

  bool D3d9Alpha::statsActive() {
    return perf::g_perfDetailed.load(std::memory_order_relaxed);
  }

  void D3d9Alpha::noteSharedBlasOmm(SharedOmmOutcome outcome, uint32_t instanceCount) {
    s_stats.sharedOmm[uint32_t(outcome)]++;
    s_stats.sharedOmmInstances += instanceCount;
  }

  void D3d9Alpha::noteFrame() {
    s_stats.frames++;
  }

  void D3d9Alpha::notePointInstancerGroup(const std::vector<RtInstance*>& instances, const OpacityMicromapManager& omm,
                                          bool bindTried, bool bound) {
    bool direct = false;
    bool legacy = false;
    for (const RtInstance* instance : instances) {
      if (instance->surface.instancesToObject != nullptr) {
        (instance->getSceneOwnedHandle() != 0 ? direct : legacy) = true;
      }
    }
    if (!direct && !legacy) {
      return;
    }
    s_stats.pointInstancerMixedGroups += (direct && legacy) ? 1 : 0;
    PointInstancerLane& group = s_stats.pointInstancers[direct ? kDirect : kLegacy];
    group.groups++;
    group.shared += instances.size() > 1 ? 1 : 0;
    group.bindTried += bindTried ? 1 : 0;
    group.bound += bound ? 1 : 0;
    for (const RtInstance* instance : instances) {
      if (instance->surface.instancesToObject == nullptr) {
        continue;
      }
      PointInstancerLane& lane = s_stats.pointInstancers[instance->getSceneOwnedHandle() != 0 ? kDirect : kLegacy];
      const OpacityMicromapInstanceData& data = instance->getOpacityMicromapInstanceData();
      lane.instances++;
      lane.placements += instance->surface.instancesToObject->size();
      lane.users += OpacityMicromapManager::usesOpacityMicromap(*instance) ? 1 : 0;
      lane.requested += data.isOmmBuildRequestedAnyGeneration() ? 1 : 0;
      lane.built += omm.getOmmReadiness(*instance) == 2 ? 1 : 0;
      lane.texelsWaiting += data.waitsForTexelFootprints() ? 1 : 0;
      lane.notCandidate += OpacityMicromapManager::usesOpacityMicromap(*instance) &&
                           !data.isOmmBuildRequestedAnyGeneration() && !omm.isOmmCandidate(*instance) ? 1 : 0;
    }
  }

  void D3d9Alpha::noteOmmCandidateGate(const RtInstance& instance, OmmGate gate) {
    if (instance.surface.instancesToObject == nullptr) {
      return;
    }
    s_stats.pointInstancers[instance.getSceneOwnedHandle() != 0 ? kDirect : kLegacy].gates[uint32_t(gate)]++;
  }

  void D3d9Alpha::noteOmmUnlinkedIdle(const RtInstance& instance) {
    const uint32_t lane = instance.getSceneOwnedHandle() != 0 ? kDirect : kLegacy;
    s_stats.unlinkedIdle[lane]++;
    s_stats.unlinkedIdlePointInstancers[lane] += instance.surface.instancesToObject != nullptr ? 1 : 0;
  }

  void D3d9Alpha::logStats(const OpacityMicromapManager* omm) {
    const double n = std::max(1.0, double(s_stats.frames));
    const auto perFrame = [n](uint64_t v) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.2f", double(v) / n);
      return std::string(buf);
    };
    const auto& so = s_stats.sharedOmm;

    // sharedOmm*: per frame, multi-instance per-geometry BLASes whose instances use OMMs, by outcome.
    // wouldBind = compatible while rtx.opacityMicromap.bindSharedBlas is off. builtNeverBound: micromaps in
    // the cache that no BLAS ever bound (now).
    Logger::info(str::format("[D3d9Alpha] frames=", s_stats.frames,
      " | sharedOmmBlas bound=", perFrame(so[uint32_t(SharedOmmOutcome::Bound)]),
      " notReady=", perFrame(so[uint32_t(SharedOmmOutcome::NotReady)]),
      " wouldBind=", perFrame(so[uint32_t(SharedOmmOutcome::OptionOff)]),
      " mixedUsers=", perFrame(so[uint32_t(SharedOmmOutcome::MixedUsers)]),
      " hashMismatch=", perFrame(so[uint32_t(SharedOmmOutcome::HashMismatch)]),
      " twoState=", perFrame(so[uint32_t(SharedOmmOutcome::TwoState)]),
      " instances=", perFrame(s_stats.sharedOmmInstances),
      " builtNeverBound=", omm ? omm->countBuiltNeverBound() : 0,
      " | bindSharedBlas=", bindSharedBlas() ? 1 : 0));

    // Per frame, by lane. users, requested, built and texelsWaiting count point-instancer instances. unlinkedIdle
    // counts every instance a bake unlinked from its micromap requests (pointInstancers: those that are one).
    // notCandidate counts users without a request that the micromap manager no longer visits. gates counts the
    // candidates the manager visited by the step they stopped at.
    const auto laneLine = [&](uint32_t lane) {
      const PointInstancerLane& l = s_stats.pointInstancers[lane];
      std::string line = str::format("groups=", perFrame(l.groups), " shared=", perFrame(l.shared),
        " bindTried=", perFrame(l.bindTried), " bound=", perFrame(l.bound),
        " instances=", perFrame(l.instances), " placements=", perFrame(l.placements),
        " users=", perFrame(l.users), " requested=", perFrame(l.requested), " built=", perFrame(l.built),
        " texelsWaiting=", perFrame(l.texelsWaiting),
        " unlinkedIdle=", perFrame(s_stats.unlinkedIdle[lane]),
        " (pointInstancers ", perFrame(s_stats.unlinkedIdlePointInstancers[lane]), ")",
        " notCandidate=", perFrame(l.notCandidate), " gates:");
      for (uint32_t g = 0; g < uint32_t(OmmGate::Count); ++g) {
        line += str::format(" ", kOmmGateNames[g], "=", perFrame(l.gates[g]));
      }
      return line;
    };
    if (s_stats.pointInstancers[kDirect].groups + s_stats.pointInstancers[kLegacy].groups +
        s_stats.unlinkedIdle[kDirect] + s_stats.unlinkedIdle[kLegacy] != 0) {
      Logger::info(str::format("[D3d9Alpha] pointInstancers direct: ", laneLine(kDirect),
        " | legacy: ", laneLine(kLegacy),
        " | mixedGroups=", perFrame(s_stats.pointInstancerMixedGroups)));
    }

    s_stats = Stats{};
  }

} // namespace dxvk
