// The fork_hooks functions of LightManager's external-light and persistent-light paths (see rtx_fork_hooks.h).
// They access LightManager's private members, so LightManager names them friends (rtx_light_manager.h).

#include "rtx_fork_hooks.h"

#include "rtx_light_manager.h"   // LightManager (full definition for friend access)
#include "rtx_lights.h"          // RtLight, kInvalidExternallyTrackedLightId
#include "rtx_options.h"         // RtxOptions::getNumFramesToPutLightsToSleep
#include "rtx_types.h"           // ReplacementInstance, PrimInstance

namespace dxvk {
namespace fork_hooks {

  // Applies external-light commands before linearization, preserving each handle's lifetime.
  void flushPendingLightMutations(LightManager& mgr) {
    auto eraseAnalytic = [&mgr](remixapi_LightHandle handle) {
      auto found = mgr.m_externalLights.find(handle);
      if (found != mgr.m_externalLights.end()) {
        found->second.getPrimInstanceOwner().setReplacementInstance(
          nullptr, ReplacementInstance::kInvalidReplacementIndex,
          &found->second, PrimInstance::Type::Light);
        mgr.m_externalLights.erase(found);
      }
      mgr.m_externalActiveLightList.erase(handle);
    };
    auto eraseDome = [&mgr](remixapi_LightHandle handle) {
      mgr.m_externalDomeLights.erase(handle);
      if (mgr.m_externalActiveDomeLight == handle) {
        mgr.m_externalActiveDomeLight = nullptr;
      }
    };
    auto activate = [&mgr](remixapi_LightHandle handle) {
      if (mgr.m_externalLights.find(handle) != mgr.m_externalLights.end()) {
        mgr.m_externalActiveLightList.insert(handle);
      } else if (mgr.m_externalDomeLights.find(handle) != mgr.m_externalDomeLights.end()
                 && mgr.m_externalActiveDomeLight == nullptr) {
        mgr.m_externalActiveDomeLight = handle;
      }
    };
    using Type = LightManager::ExternalLightMutationType;
    for (const auto& mutation : mgr.m_pendingExternalLightMutations) {
      const auto handle = mutation.handle;
      switch (mutation.type) {
      case Type::AddAnalytic: {
        eraseDome(handle);
        const auto& incoming = mgr.m_pendingExternalAnalyticLights[mutation.payload];
        auto found = mgr.m_externalLights.find(handle);
        if (found != mgr.m_externalLights.end()) {
          auto& light = found->second;
          const uint16_t bufferIndex = light.getBufferIdx();
          const uint32_t stableIdentity = light.getStableIdentity();
          light = incoming;
          light.setBufferIdx(bufferIndex);
          light.setStableIdentity(stableIdentity);
          light.setFrameLastTouched(mgr.device()->getCurrentFrameId());
        } else {
          setExternalLightEmplace(mgr, handle, incoming);
        }
        break;
      }
      case Type::AddDome:
        eraseAnalytic(handle);
        mgr.m_externalDomeLights.insert_or_assign(handle, mgr.m_pendingExternalDomeLights[mutation.payload]);
        break;
      case Type::Remove:
        eraseAnalytic(handle);
        eraseDome(handle);
        mgr.m_persistentExternalLights.erase(handle);
        break;
      case Type::Activate:
        activate(handle);
        break;
      case Type::RegisterPersistent:
        if (handle) {
          mgr.m_persistentExternalLights.insert(handle);
        }
        break;
      case Type::UnregisterPersistent:
        mgr.m_persistentExternalLights.erase(handle);
        break;
      case Type::AutoInstance:
        for (auto persistent : mgr.m_persistentExternalLights) {
          activate(persistent);
        }
        break;
      }
    }
    mgr.m_pendingExternalLightMutations.clear();
    mgr.m_pendingExternalAnalyticLights.clear();
    mgr.m_pendingExternalDomeLights.clear();
    for (auto handle : mgr.m_persistentExternalLights) {
      activate(handle);
    }
  }

  // Applies static-light sleep to externally tracked lights.
  // A static light stops copying new data once isStaticCount reaches numFramesToPutLightsToSleep, which
  // preserves its temporal denoiser data. Dynamic lights, and every light while suppressLightKeeping() is set,
  // always update. An update keeps the buffer index and stable identity, and keeps externallyTrackedLightId
  // unless externalId is kInvalidExternallyTrackedLightId. frameLastTouched is always stamped.
  void updateLightStaticSleep(
      RtLight* light,
      const RtLight& newLight,
      DxvkDevice* device,
      uint64_t externalId) {

    const uint16_t bufferIdx = light->getBufferIdx();
    // Upstream's light-identity table (sparse rendering) keys reservoirs on a
    // per-light stable identity that must survive a definition update.
    const uint32_t stableIdentity = light->getStableIdentity();

    if (!newLight.isDynamic && !LightManager::suppressLightKeeping()) {
      const uint32_t isStaticCount = light->isStaticCount;

      // If this light hasn't moved for N frames, put it to sleep to preserve
      // temporal data
      if (isStaticCount < RtxOptions::getNumFramesToPutLightsToSleep()) {
        *light = newLight;
        light->setBufferIdx(bufferIdx);
        light->setStableIdentity(stableIdentity);
        if (externalId != kInvalidExternallyTrackedLightId) {
          light->setExternallyTrackedLightId(externalId);
        }
        light->isStaticCount = isStaticCount + 1;  // Preserve and increment counter
      } else {
        // The light is asleep, so only its counter advances.
        light->isStaticCount = isStaticCount + 1;
      }
    } else {
      // Dynamic lights always update
      *light = newLight;
      light->setBufferIdx(bufferIdx);
      light->setStableIdentity(stableIdentity);
      if (externalId != kInvalidExternallyTrackedLightId) {
        light->setExternallyTrackedLightId(externalId);
      }
    }

    light->setFrameLastTouched(device->getCurrentFrameId());
  }

  // Applies a new analytical definition during the ordered mutation flush and stamps
  // frameLastTouched.
  void setExternalLightEmplace(
      LightManager& mgr,
      remixapi_LightHandle handle,
      const RtLight& rtlight) {

    auto [it, inserted] = mgr.m_externalLights.emplace(handle, rtlight);
    if (inserted) {
      it->second.setFrameLastTouched(mgr.device()->getCurrentFrameId());
    }
  }

  // Defers erasure to the ordered mutation point without invalidating live iterators.
  void disableExternalLightQueue(LightManager& mgr, remixapi_LightHandle handle) {
    mgr.m_pendingExternalLightMutations.push_back({ LightManager::ExternalLightMutationType::Remove, handle });
  }

  // Called from LightManager::registerPersistentExternalLight.
  void registerPersistentLight(LightManager& mgr, remixapi_LightHandle handle) {
    mgr.m_pendingExternalLightMutations.push_back({ LightManager::ExternalLightMutationType::RegisterPersistent, handle });
  }

  // Called from LightManager::unregisterPersistentExternalLight.
  void unregisterPersistentLight(LightManager& mgr, remixapi_LightHandle handle) {
    mgr.m_pendingExternalLightMutations.push_back({ LightManager::ExternalLightMutationType::UnregisterPersistent, handle });
  }

  // Activates the persistent set after all commands preceding this request.
  void queueAutoInstancePersistent(LightManager& mgr) {
    mgr.m_pendingExternalLightMutations.push_back({ LightManager::ExternalLightMutationType::AutoInstance, nullptr });
  }

} // namespace fork_hooks
} // namespace dxvk
