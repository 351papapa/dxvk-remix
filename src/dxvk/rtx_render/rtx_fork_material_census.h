#pragma once

// rtx.coreMaterial.census.
//
// Logging only. Every 300 frames, while the option is on, SceneManager::prepareSceneData hands this the frame's
// ordered surface list (one entry per surface, the order the surface-material upload writes) and the material /
// extension caches, and [MaterialCensus] lines describe type and class, alpha state, material features,
// non-identity legacy state and index widths. Each count is surfaces / unique materials (cache index) / share of
// triangles (index count / 3 of the instance's BLAS entry).

#include <cstdint>
#include <vector>

namespace dxvk {

  class RtInstance;
  struct RtSurfaceMaterial;

  void runMaterialCensus(const std::vector<RtInstance*>& orderedSurfaces,
                         const std::vector<RtSurfaceMaterial>& materials,
                         const std::vector<RtSurfaceMaterial>& extensions,
                         uint32_t frameId);

} // namespace dxvk
