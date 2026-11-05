// The fork_hooks functions of the particle manager's spawn path (see rtx_fork_hooks.h).
//
// RtxParticleSystemManager::spawnParticles records the emitter as a slot index into InstanceManager::m_instances,
// but writeSpawnContextsToGpu reads it inside SceneManager::prepareSceneData, after garbageCollection() has
// swap-removed instances. An emitter appended late in the frame is the back element that moves, so the index can be stale.

#include "rtx_fork_hooks.h"

#include "rtx_constants.h"        // kInvalidInstanceId
#include "rtx_instance_manager.h" // RtInstance::getId

#include "../../util/log/log.h"
#include "../../util/util_once.h"
#include "../../util/util_string.h"

namespace dxvk {
namespace fork_hooks {

  // The recorded index is only a hint. It is validated against the stable RtInstance id, with a scan as the
  // fallback when it has drifted. The scan runs at most once per spawn context.
  const RtInstance* resolveSpawnEmitterInstance(
      const std::vector<RtInstance*>& instanceTable,
      uint32_t recordedVectorIdx,
      uint64_t recordedInstanceUid) {
    const bool hasUid = recordedInstanceUid != kInvalidInstanceId;

    // Fast path: the index still points at the instance that was recorded.
    if (recordedVectorIdx < instanceTable.size()) {
      const RtInstance* candidate = instanceTable[recordedVectorIdx];
      if (candidate != nullptr && (!hasUid || candidate->getId() == recordedInstanceUid)) {
        return candidate;
      }
    }

    // Instances created by the renderer (view model / player model copies) can carry kInvalidInstanceId.
    // They never spawn particles, since they are created after simulate().
    if (!hasUid) {
      return nullptr;
    }

    for (const RtInstance* candidate : instanceTable) {
      if (candidate != nullptr && candidate->getId() == recordedInstanceUid) {
        // Recoverable, and expected on any frame where garbage collection ran, so
        // report it once per process rather than per frame.
        ONCE(Logger::info(str::format(
          "[RTX Particles] spawn context index drifted: recorded slot ", recordedVectorIdx,
          " no longer holds emitter id ", recordedInstanceUid, "; recovered at slot ",
          candidate->getVectorIdx(), " (instance table size ", instanceTable.size(), ")")));
        return candidate;
      }
    }

    // Unlike the drift case above, the spawn is discarded here, so it is logged.
    ONCE(Logger::warn(str::format(
      "[RTX Particles] spawn context dropped: emitter instance id ", recordedInstanceUid,
      " (recorded slot ", recordedVectorIdx, ") is no longer in the instance table (size ",
      instanceTable.size(), "); this frame's spawn is discarded")));

    return nullptr;
  }

}  // namespace fork_hooks
}  // namespace dxvk
