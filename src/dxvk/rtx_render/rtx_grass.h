#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "rtx_option.h"
#include "rtx_asset_replacer.h"
#include "rtx/pass/grass/grass_args.h"

namespace dxvk {
class SceneManager;
class GrassGpu;

class GrassSystem {
public:
  GrassSystem();
  ~GrassSystem();
  void update(const Rc<DxvkContext>& ctx, SceneManager& scene);
  void clear();
  bool matchesOriginalName(const std::string* name) const;
  GrassGpu* gpu() const { return m_gpu.get(); }

  RTX_OPTION("rtx.grass", bool, enabled, false, "Generate USD grass on surfaces selected by original or replacement texture names.");
  RTX_OPTION("rtx.grass", std::string, assetPath, "", "Grass USD path relative to the host executable directory, or an absolute path.");
  RTX_OPTION("rtx.grass", std::string, diffuseTextures, "", "Semicolon-separated replacement diffuse filenames. Exact, case-insensitive matching.");
  RTX_OPTION("rtx.grass", std::string, originalTextures, "", "Exact captured texture names for unreplaced geometry, resolved through logs/DffDrawCalls captures. Ambiguous names are skipped.");
  RTX_OPTION("rtx.grass", float, radius, 35.0f, "Horizontal camera radius in game units.");
  RTX_OPTION("rtx.grass", bool, frontRearRange, false, "Use camera-facing half discs: front radius R, rear radius R/2.");
  RTX_OPTION("rtx.grass", bool, frustumCull, false, "Cull grass outside the expanded camera frustum beyond 6 world units. Disable to preserve all in-range grass for offscreen reflections and shadows.");
  RTX_OPTION("rtx.grass", float, density, 0.4f, "Grass clumps per square game unit of surface.");
  RTX_OPTION("rtx.grass", float, height, 0.65f, "Scale only local Z, aligned with the ground normal. Does not change width.");
  RTX_OPTION("rtx.grass", float, width, 0.65f, "Independent local XY scale of the normalized USD grass.");
  RTX_OPTION("rtx.grass", float, maxSlopeDegrees, 45.0f, "Maximum ground slope in degrees.");
  RTX_OPTION("rtx.grass", float, fadeDistance, 5.0f, "Fade local Z inside the range boundary, including the front/rear transition.");
  RTX_OPTION("rtx.grass", float, groundOffset, -0.02f, "Root offset along the ground normal.");
  RTX_OPTION("rtx.grass", bool, keepInsideSurface, false, "Inset roots by grass bounds at open mesh edges. Shared triangle edges are welded.");
  RTX_OPTION("rtx.grass", int, maxInstances, 256, "Maximum submitted grass clumps (1-512); nearest clumps take priority.");
  RTX_OPTION("rtx.grass", int, reloadVersion, 0, "Increment to reload the grass USD and clear placement caches.");

private:
  friend class ImGUI;
  struct Patch { std::shared_ptr<const std::vector<GrassRoot>> points; uint32_t lastFrame = 0; };
  struct Job;
  struct Workers;
  struct RootJob;
  struct Source { std::vector<std::array<Vector3, 3>> triangles; uint32_t lastFrame = 0; };
  std::unordered_map<uint64_t, Source> m_sources;
  std::shared_ptr<ReplacementBucket> m_asset;
  std::unordered_map<uint64_t, Patch> m_patches;
  std::vector<std::shared_ptr<Job>> m_jobs;
  std::unique_ptr<Workers> m_workers;
  std::shared_ptr<RootJob> m_rootJob;
  std::unique_ptr<GrassGpu> m_gpu;
  std::vector<uint64_t> m_activePatches;
  uint64_t m_generation = 0;
  uint32_t m_budget = 0;
  size_t m_candidateCount = 0;
  std::string m_loadedPath;
  std::string m_resolvedPath;
  std::string m_rules;
  std::string m_originalRules;
  std::vector<std::string> m_names;
  std::string m_status = "Disabled";
  int m_reloadVersion = -1;
  float m_density = -1;
  float m_width = -1;
  float m_slope = -1;
  bool m_inset = false;
  bool m_zUp = true;
  bool m_wasEnabled = false;
  float m_boundWidth = 0;
  float m_boundHeight = 0;
  Vector3 m_horizontalForward = Vector3(0, 1, 0);
  uint32_t m_frame = 0;
  size_t m_matchedSurfaces = 0;
  size_t m_visibleClumps = 0;
  size_t m_createdClumps = 0;
  double m_updateMilliseconds = 0;
  size_t m_skippedSurfaces = 0;
  bool m_limited = false;
};
}
