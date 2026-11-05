#pragma once

// GPU/BLAS work for alpha-tested/blended content and replacement assets on the standard D3D9 path.
//
// The controls are in the dev menu (Rendering, "D3D9 Alpha / Replacements", next to "Opacity Micromap"):
// - opacityMicromap.bindSharedBlas: per-geometry BLASes shared by several instances bind the opacity micromap their
//   instances all request (OpacityMicromapManager::checkSharedBlasOmm).
//
// [D3d9Alpha] counters are written with the detailed performance log (rtx.profile.detailedLogging) and are not
// collected while it is off.

#include <cstdint>
#include <vector>

#include "rtx_option.h"

namespace dxvk {

  class OpacityMicromapManager;
  class RtInstance;

  class D3d9Alpha {
  public:
    RTX_OPTION("rtx.opacityMicromap", bool, bindSharedBlas, true,
               "Lets a per-geometry BLAS shared by several instances (meshes above rtx.minPrimsInDynamicBLAS that are "
               "drawn more than once, such as repeated foliage and fences) use an opacity micromap, when every instance "
               "asks for the same one (same source hash, texture factor and colour flags). Unknown micro-triangles keep "
               "the exact alpha test, so the result is the same as for single-instance geometry.");

    // Counters, collected only while detailed performance logging is on.
    static bool statsActive();

    // AccelManager: one call per multi-instance per-geometry BLAS with OMM users (result of checkSharedBlasOmm).
    enum class SharedOmmOutcome : uint8_t { Bound, NotReady, MixedUsers, HashMismatch, TwoState, OptionOff };
    static void noteSharedBlasOmm(SharedOmmOutcome outcome, uint32_t instanceCount);
    static void noteFrame();

    // AccelManager: one call per per-geometry BLAS group that holds a point instancer. bindTried says whether the
    // group asked for a micromap (one instance, or a compatible shared BLAS), bound whether it got one. The counters
    // are split by lane: direct (scene-owned instances) and legacy.
    static void notePointInstancerGroup(const std::vector<RtInstance*>& instances, const OpacityMicromapManager& omm,
                                        bool bindTried, bool bound);
    // OpacityMicromapManager: a bake found the instance's texel footprints missing while the instance was not updated
    // this frame, so the instance was unlinked from its micromap requests.
    static void noteOmmUnlinkedIdle(const RtInstance& instance);
    // OpacityMicromapManager::processOmmCandidates: the step a point-instancer candidate stopped at this frame.
    enum class OmmGate : uint8_t {
      NoBlas, NewInstance, RetryWait, Throttled, NoBudget, NotUser, TexturesNotResident, Blacklisted, StatsFull,
      BelowFilter, RegisterFailed, Registered, Count
    };
    static void noteOmmCandidateGate(const RtInstance& instance, OmmGate gate);

    // Writes and resets the [D3d9Alpha] lines (rtx perf log, after [OmmChurn]).
    static void logStats(const OpacityMicromapManager* omm);
  };

} // namespace dxvk
