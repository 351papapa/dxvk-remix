/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtx_types.h"
#include "../util/util_spatial_map.h"

namespace dxvk {

class DxvkDevice;
class RtCamera;
class RayPortalManager;
namespace translator {
  struct LifetimeFrame;
}

// Tracks draw calls across frames by mapping each draw call to a ReplacementInstance.
// Uses a two-level hash lookup (identity hash for L1, tracking hash + spatial proximity
// for L2) to match the current frame's draw calls against previous frames.
class DrawCallTracker {
public:
  DrawCallTracker(DxvkDevice* device);
  ~DrawCallTracker() = default;

  DrawCallTracker(const DrawCallTracker&) = delete;
  DrawCallTracker& operator=(const DrawCallTracker&) = delete;

  // Removes replacement instances whose spatialMapHash matches (same bucket key as
  // m_assetSpatialMaps). Used when a draw source tied to that bucket is invalidated.
  void removeReplacementInstancesWithSpatialMapHash(XXH64_hash_t spatialMapHash);

  // Find or create a ReplacementInstance from a SceneDraw.
  // L1 identity includes legacy material hash from SceneDraw plus the hash of
  // overrideMaterialData when non-null (terrain bake, particles, etc.); full merged
  // render MaterialData is not required for tracking.
  // Portal-aware matching for ViewModel draw calls is performed using the RayPortalManager.
  ReplacementInstance* findOrCreateReplacementInstance(
    const ReplacementInstance::LookupKey& key);

  // Wrapper for the above function that constructs the LookupKey from the SceneDraw.
  ReplacementInstance* findOrCreateReplacementInstance(
      const SceneDraw& drawCallState,
      const RayPortalManager& rayPortalManager,
      const MaterialData* overrideMaterialData);

  // Look up an existing ReplacementInstance by L1 identity hash only. Returns
  // nullptr if no RI exists for this key. Unlike findOrCreateReplacementInstance,
  // this never allocates and never runs the L2 spatial search -- intended
  // for "is there stale state to clean up?" probes that must not pollute the
  // tracker with empty RIs.
  ReplacementInstance* findReplacementInstanceByIdentity(XXH64_hash_t identityHash);

  // Garbage collect expired ReplacementInstances. Object anti-culling keeps mesh RIs alive
  // if their bounding box intersects the camera frustum. Light content uses the camera's
  // light anti-culling frustum (wider FOV) and extended lifetime.
  void garbageCollectReplacementInstances(
      RtCamera& camera,
      bool isAntiCullingSupported,
      bool parallelScan = false,   // rtx.parallelScenePrepScans
      uint32_t keptCutFrame = ~0u); // SceneManager::m_lastKeptCutFrame
  // The same once its inputs are read (lifetime.currentFrame and forceCollect are filled in here).
  void collectReplacementInstances(const translator::LifetimeFrame& lifetimeIn, bool parallelScan = false);

  size_t getReplacementInstanceCount() const { return m_replacementInstances.size(); }

  void clear();

  // Rebuild all spatial maps with a new cell size.
  // Call when uniqueObjectDistance changes.
  void rebuildSpatialMaps(float cellSize);

  const std::vector<std::unique_ptr<ReplacementInstance>>& getReplacementInstances() const {
    return m_replacementInstances;
  }

  // Compute a hash that uniquely identifies a draw call for L1 matching.
  // When overrideMaterialData is non-null, its hash is mixed in (e.g. terrain bake, particle material).
  static XXH64_hash_t computeIdentityHash(
      const SceneDraw& drawCallState,
      const MaterialData* overrideMaterialData = nullptr);

  // rtx.retainedSteadyFastPath: changes whenever a ReplacementInstance cached by a retained
  // entry (ReplacementInstance::retainedSteady) is destroyed or re-keyed, or on clear().
  uint64_t getRetainedSteadyEpoch() const { return m_retainedSteadyEpoch; }
  // rtx.retainedPreciseInvalidation: the same events, split into clear() (invalidates everything;
  // this epoch) and the individual replacement instances destroyed or re-keyed since the last
  // take (addresses only, never dereferenced: they may be freed).
  uint64_t getRetainedSteadyClearEpoch() const { return m_retainedSteadyClearEpoch; }
  size_t getRetainedSteadyInvalidationCount() const { return m_retainedSteadyInvalidations.size(); }
  const std::vector<const ReplacementInstance*>& getRetainedSteadyInvalidations() const { return m_retainedSteadyInvalidations; }
  void clearRetainedSteadyInvalidations() { m_retainedSteadyInvalidations.clear(); }

private:
  uint64_t m_retainedSteadyEpoch = 0;
  uint64_t m_retainedSteadyClearEpoch = 0;
  std::vector<const ReplacementInstance*> m_retainedSteadyInvalidations;
  std::vector<uint8_t> m_gcStaleScratch; // rtx.parallelScenePrepScans
  void noteRekeyOrDestroy(const ReplacementInstance* ri) {
    if (ri != nullptr && ri->retainedSteady) {
      ++m_retainedSteadyEpoch;
      if (m_retainedSteadyInvalidations.size() < 65536) {
        m_retainedSteadyInvalidations.push_back(ri);
      } else {
        // Nobody taking them: treat as a clear.
        m_retainedSteadyInvalidations.clear();
        ++m_retainedSteadyClearEpoch;
      }
    }
  }
  // Cleans up a ReplacementInstance's map/index entries and prims. Does NOT remove
  // the entry from m_replacementInstances -- callers must handle that separately.
  void destroyReplacementInstance(ReplacementInstance* replacementInstance);

  DxvkDevice* m_device;
  uint32_t frameId() const;

  std::unordered_map<XXH64_hash_t, ReplacementInstance*> m_identityHashMap;

  using ReplacementSpatialMap = SpatialMap<ReplacementInstance>;
  std::unordered_map<XXH64_hash_t, ReplacementSpatialMap> m_assetSpatialMaps;

  // Attempts to match a newly created ReplacementInstance through ray portals.
  // If a match is found, destroys the new RI and returns the reassociated match.
  // Returns nullptr if no portal match is found.
  ReplacementInstance* tryPortalMatch(
      ReplacementInstance* newInstance,
      const ReplacementInstance::LookupKey& key,
      const RayPortalManager& rayPortalManager);

  // Erases an entry from a spatial map bucket and removes the bucket if empty.
  void eraseFromSpatialMap(
      std::unordered_map<XXH64_hash_t, ReplacementSpatialMap>& mapOfMaps,
      XXH64_hash_t bucketKey,
      XXH64_hash_t transformHash,
      const ReplacementInstance* data);

  // Reassociates an existing ReplacementInstance with a new draw call identity.
  // Updates position fields and optionally moves within spatial maps.
  // moveInAssetMap: if non-null, the RI is moved within this asset spatial map.
  ReplacementInstance* reassociateMatch(
      ReplacementInstance* match,
      const ReplacementInstance::LookupKey& key,
      ReplacementSpatialMap* moveInAssetMap);

  std::vector<std::unique_ptr<ReplacementInstance>> m_replacementInstances;
  uint32_t m_nextReplacementInstanceId = 0;
};

}  // namespace dxvk
