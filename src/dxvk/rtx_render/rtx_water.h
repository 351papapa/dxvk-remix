#pragma once
#include "rtx_common_object.h"
#include "rtx_option.h"
#include "rtx_resources.h"
#include "rtx_texture.h"
#include "rtx_water_foam_geometry.h"
#include "rtx_water_surface.h"
#include "rtx_water_medium.h"
#include "rtx/pass/volumetrics/water_volume_caustics_args.h"
#include "../../util/util_vector.h"
#include "../../../public/include/remixapi/water_controls.h"

struct WetnessArgs;

namespace dxvk {
class RtCamera;
class SceneManager;
struct WaterLightSurface {
  bool valid = false;
  float height = 0.0f;
};

class WaterSystem : public CommonDeviceObject {
public:
  explicit WaterSystem(DxvkDevice* device) : CommonDeviceObject(device) { }
  ~WaterSystem();
  void publishControls(const RtCamera& camera);
  void prepareCaustics(RtxContext* ctx, const Resources::RaytracingOutput& output);
  void prepareVolumeCaustics(RtxContext* ctx);
  void bindVolumeCausticsResources(RtxContext* ctx);
  bool volumeCausticsActive() const { return m_volumeCausticsActive; }
  bool volumeCausticsChanged() const { return m_volumeCausticsChanged; }
  const WaterVolumeCausticsArgs& volumeCausticsArgs() const { return m_volumeCausticsArgs; }
  void compositeCaustics(RtxContext* ctx, const Resources::RaytracingOutput& output);
  static std::string resolveAssetPath(const std::string& path);
  void compositeFoam(RtxContext* ctx, const Resources::RaytracingOutput& output);
  void fillShoreWetnessArgs(WetnessArgs& args);
  void bindShoreWetnessResources(RtxContext* ctx);
  static WaterOpticalProperties opticalProperties();
  void updateCameraMedium(const SceneManager& scene);
  const WaterMediumState& cameraMedium() const { return m_cameraMedium; }
  const WaterLightSurface& cameraLightSurface() const { return m_cameraLightSurface; }

  // Updates GPU geometry before scene ingestion and acceleration-structure preparation.
  void submitFoamSurface(Rc<DxvkContext> ctx);
  bool ownsFoamGeometry(const RasterGeometry& geometry) const;
  bool ownsFoamVertexBuffer(const Rc<DxvkBuffer>& buffer) const;
  void submitGpuSurface(Rc<DxvkContext> ctx);

  static void showImguiSettings();
  static bool needsWetnessShader() { return enabled() && (shoreWetnessStrength() > 0.0f || detailStrength() > 0.0f); }
  RTX_OPTION("rtx.water", std::string, foamColorTexture, "", "DDS albedo and coverage texture for foam shaded through scene materials.");
  RTX_OPTION("rtx.water", std::string, foamNormalTexture, "", "DDS tangent-space normal texture for foam shaded through scene materials.");
  RTX_OPTION("rtx.water", std::string, foamCompositeTexture, "", "DDS pattern for the screen-space foam display.");
  RTX_OPTION_FLAG("rtx.water", bool, enabled, false, RtxOptionFlags::InvalidatesDrawcallTranslation, "Enable the new water system. Disabled: restore original water geometry and existing water texture/weather shading without new simulation or material overrides.");
  RTX_OPTION("rtx.water", bool, gpuSurface, false, "Generate shared-protocol water vertices, beach waves, wake arms and normals on the GPU. Requires the matching shared-protocol plugin; safely idles when no shared GPU-water frame is available.");
  RTX_OPTION("rtx.water", Vector2, waterMapOrigin, Vector2(0.0f), "World XY origin of the producer's 128 by 128 water-level map.");
  RTX_OPTION("rtx.water", float, waterMapCellSize, 1.0f, "World units per cell in the producer's water-level map.");
  RTX_OPTION("rtx.water", bool, gpuSimulation, false, "Run the interaction wave solver and wake-foam surface following on the GPU. Requires GPU water surface; safely idles when the matching shared-protocol simulation mapping is unavailable.");
  RTX_OPTION("rtx.water", bool, localFlow, false, "Experimental GPU local wake flow and foam transport. Requires both GPU water options. Switching resets wake foam history; shore foam material is unchanged.");
  RTX_OPTION("rtx.water", bool, localFlowDiagnostics, false, "Temporary asynchronous GPU flow and foam readback every two seconds. Disable after diagnosis.");
  RTX_OPTION("rtx.water", bool, hullWaves, false, "GPU boat pressure waves deform the actual water surface and normals. Requires both GPU water options, independent of foam flow.");
  RTX_OPTION("rtx.water", float, hullWaveStrength, 1.f, "Boat pressure-wave strength, range 0 to 3.");
  RTX_OPTION("rtx.water", float, hullWaveSpeed, 5.f, "Local water wave propagation speed, range 2 to 8 world units per second.");
  RTX_OPTION("rtx.water", float, hullWaveDamping, .35f, "Local water wave damping per second, range 0.05 to 2.");
  RTX_OPTION("rtx.water", float, localFlowStrength, 1.f, "Boat forward flow force multiplier.");
  RTX_OPTION("rtx.water", float, localFlowSwirl, 1.f, "Boat counter-rotating wake force multiplier.");
  RTX_OPTION("rtx.water", float, localFlowDrag, .35f, "Local wake velocity damping per second.");

  RTX_OPTION_FLAG("rtx.water", bool, overrideMaterial, false, RtxOptionFlags::InvalidatesDrawcallTranslation, "Apply water color, distance and non-thin-walled shading to translucent AnimatedWater surfaces.");
  RTX_OPTION_FLAG("rtx.water", Vector3, transmittanceColor, Vector3(0.51138175f, 0.8944833f, 0.98841697f), RtxOptionFlags::InvalidatesDrawcallTranslation, "Baseline linear RGB water transmittance at the authored measurement distance. Added particles contribute neutral extinction; the resulting total extinction is shared by the surface, caustics and underwater volume.");
  RTX_OPTION_FLAG("rtx.water", float, transmittanceDistance, 7.5f, RtxOptionFlags::InvalidatesDrawcallTranslation, "Water transmittance measurement distance in game world units.");
  RTX_OPTION_FLAG("rtx.water", float, particleExtinctionPerMeter, 0.0f, RtxOptionFlags::InvalidatesDrawcallTranslation, "Added neutral suspended-particle extinction per meter, 0 to 1. Zero preserves authored water transmittance. Increases the same total extinction for water materials, caustics and underwater lighting; respects scene scale.");
  RTX_OPTION("rtx.water", float, amplitude, 0.55f, "Local geometric swell amplitude in game world units. Zero disables swell.");
  RTX_OPTION("rtx.water", float, oceanWaveStrength, 1.25f, "Overall ocean swell and beach-wave height multiplier, applied on the GPU.");
  RTX_OPTION("rtx.water", float, rainWaveMultiplier, 1.25f, "Rainy ocean wave height relative to normal weather; blends with the weather transition.");
  RTX_OPTION("rtx.water", float, stormWaveMultiplier, 1.5f, "Stormy ocean wave height relative to normal weather; blends with the weather transition.");
  RTX_OPTION("rtx.water", float, wavelength, 12.0f, "Primary geometric swell wavelength in game world units.");
  RTX_OPTION("rtx.water", float, speed, 1.0f, "Geometric swell animation speed multiplier.");
  RTX_OPTION("rtx.water", float, steepness, 0.55f, "Gerstner horizontal displacement strength. Limited internally to prevent folded crests.");
  RTX_OPTION("rtx.water", bool, beachWaves, false, "GPU beach waves driven by measured terrain depth. Requires GPU water surface; independent of ocean swell amplitude.");
  RTX_OPTION("rtx.water", std::string, beachMaskPath, "", "Optional IVTX world-space beach eligibility mask. Empty or invalid uses measured shallow-coast qualification; masks are specific to their authored map.");
  RTX_OPTION("rtx.water", float, beachWaveHeight, 1.06f, "Beach wave height in world units, independent of the main ocean swell amplitude.");
  RTX_OPTION("rtx.water", float, beachWaveLength, 8.9f, "Shore depth-domain phase scale; local crest distance also depends on the beach slope.");
  RTX_OPTION("rtx.water", float, beachWavePeriod, 4.5f, "Seconds between beach wave crests.");
  RTX_OPTION("rtx.water", float, beachWaveDepth, 3.1f, "Water depth in world units over which terrain-driven beach waves develop.");
  RTX_OPTION("rtx.water", float, detailStrength, 1.5f, "Multiplier on existing weather-driven AnimatedWater micro-wave strength.");
  RTX_OPTION("rtx.water", float, detailScale, 1.0f, "Multiplier on existing weather-driven AnimatedWater micro-wave scale.");
  RTX_OPTION("rtx.water", float, interactionStrength, 2.0f, "Object and boat wake impulse strength multiplier.");
  RTX_OPTION("rtx.water", float, interactionRadius, 1.5f, "Object and boat wake impulse radius multiplier.");
  RTX_OPTION("rtx.water", float, propagationSpeed, 4.0f, "Interaction wave propagation speed in game world units per second.");
  RTX_OPTION("rtx.water", float, damping, 0.45f, "Interaction wave damping per second; smaller values persist longer.");
  RTX_OPTION("rtx.water", float, simulationRadius, 54.0f, "Circular geometric wave and interaction radius around the player, in world units (16-96). Fixed grid resolution: larger radii reduce spatial detail rather than increase the cell count.");
  RTX_OPTION("rtx.water", bool, reduceOffscreen, false, "Reduce local interaction solver to 30 Hz outside the expanded camera view; preserve state for camera turns and reflections.");
private:
  std::unique_ptr<WaterGpuSurface> m_gpuSurface;
  // The actual foam line fix. Intersection foam is displayed as a real surface that the
  // renderer lights, instead of being tinted into the finished image by a composite pass
  // that has to invent its own brightness. When this is on the screen-space composite is
  // skipped so the two cannot both draw; contact detection and density transport keep
  // running either way, because they are what feeds the surface.
  RTX_OPTION("rtx.water", bool, foamLineMaterialPath, false, "Display intersection foam as real geometry lit by the scene, replacing the screen-space composite that faked its own illumination. Foam keeps the same density, width and fade; only the lighting path changes. Off by default.");
  RTX_OPTION("rtx.water", bool, shoreFoamEnabled, false, "Continuous noise foam at rendered water/geometry intersections. No shoreline texture cards.");
  RTX_OPTION("rtx.water", bool, shoreFoamFrustumCulling, false, "Cull intersection foam outside the expanded main camera view. Saves GPU contact queries and geometry work; offscreen reflections may omit foam.");
  RTX_OPTION("rtx.water", bool, contactFoam, false, "Contact foam at water/geometry intersections: shoreline, rocks and piers. The player is excluded separately by surface class, so disabling this is a global off switch and not the way to remove only the player's foam.");
  RTX_OPTION("rtx.water", bool, contactFoamPlayerExclusion, false, "Exclude the third-person player model from contact foam. Turn this off to A/B test whether the player class is matching more surfaces than it should; hot reloadable, no rebuild needed.");
  RTX_OPTION("rtx.water", bool, shoreFoamOceanOnly, false, "Restrict intersection foam to the shared-protocol ocean material; exclude authored pools and fountains.");
  RTX_OPTION("rtx.water", float, shoreFoamWidth, 0.85f, "World-space contact distance for shoreline foam.");
  RTX_OPTION("rtx.water", float, shoreFoamStrength, 0.8f, "Coverage of procedural shoreline foam.");
  RTX_OPTION("rtx.water", float, shoreFoamRunupWidth, 2.4f, "Width of the GPU beach foam band along a sloping shore. Zero disables new beach foam.");
  RTX_OPTION("rtx.water", float, shoreFoamWaveResponse, 2.5f, "Foam generation response to the measured GPU water-surface rise and fall.");
  RTX_OPTION("rtx.water", float, shoreFoamLifetime, 7.0f, "Seconds for waterborne foam to decay to five percent without a new source.");
  RTX_OPTION("rtx.water", float, crestFoamStrength, 2.0f, "Foam generated at ocean wave crests from the same continuous GPU wave phase as the water surface. Zero disables crest foam.");
  RTX_OPTION("rtx.water", float, crestFoamThreshold, 0.06f, "Minimum convex water curvature for crest foam, in inverse world units.");
  RTX_OPTION("rtx.water", float, shoreWetnessLifetime, 20.0f, "Seconds for exposed beach wetness to decay to five percent. The beach carries wetness, never residual foam.");
  RTX_OPTION("rtx.water", float, shoreWetnessStrength, 1.0f, "Darkening and smoothness of terrain recently covered by the ocean. Independent of rain wetness.");
  RTX_OPTION("rtx.water", float, shoreFoamTextureScale, 3.0f, "World units per waterborne foam texture repeat.");
  RTX_OPTION("rtx.water", float, shoreFoamSkyAmbient, 1.0f, "Multiplier on intersection foam ambient. 1.0 keeps the foam's own brightness while still following the environment; lower darkens it, higher brightens it. 0 removes the ambient term entirely.");
  RTX_OPTION("rtx.water", bool, causticsEnabled, false, "Project animated light caustics onto submerged opaque receivers. Requires AnimatedWater geometry.");
  RTX_OPTION("rtx.water", int, causticsLocalLights, 4, "Maximum nearby RTX local lights contributing to caustics, 0 disables, up to 8. Candidates ranked by estimated camera irradiance.");
  RTX_OPTION("rtx.water", float, causticsLocalLightRange, 60.0f, "Local caustic light and receiver range from camera in world units; fades over the outer quarter.");
  RTX_OPTION("rtx.water", float, causticsFadeStart, 0.5f, "Fraction of maximum depth or reflected height where caustics begin fading out.");
  RTX_OPTION("rtx.water", float, causticsStrength, 2.0f, "Additional focused sunlight strength on submerged receivers.");
  RTX_OPTION("rtx.water", float, causticsScale, 4.0f, "World units per caustic texture repeat for the shared-protocol ocean material.");
  RTX_OPTION("rtx.water", bool, causticsScalePerDomain, false, "Use causticsPoolScale for authored pools and fountains instead of the ocean scale. The split is decided per receiver from isOceanWater on the water surface the refracted ray actually hit. Off: every water surface uses causticsScale.");
  RTX_OPTION("rtx.water", float, causticsPoolScale, 1.6f, "World units per caustic texture repeat for non-ocean animated water (pools, fountains). Small values pack more of the pattern into the same distance, so the web reads finer than the ocean tile at the same camera distance. Ignored when causticsScalePerDomain is off.");
  RTX_OPTION("rtx.water", float, causticsDispersion, 0.35f, "Chromatic dispersion on caustic receivers, shared above and below water. Evaluates three pattern taps; zero uses one tap.");
  RTX_OPTION("rtx.water", float, causticsSpeed, 0.6f, "Caustic texture deformation speed. Zero freezes the pattern.");
  RTX_OPTION("rtx.water", float, causticsMaxDepth, 12.0f, "Maximum receiver depth in world units.");
  RTX_OPTION("rtx.water", std::string, causticsTexture, "", "DDS pattern used by receiver caustics.");
  RTX_OPTION("rtx.water", bool, causticsRandomized, false, "Randomize caustic tiles and continuously deform their coordinates.");
  RTX_OPTION("rtx.water", bool, reflectedCausticsEnabled, false, "Approximate sunlight reflected from water onto nearby opaque receivers.");
  RTX_OPTION("rtx.water", float, reflectedCausticsStrength, 1.0f, "Reflected caustic strength, independent of underwater strength.");
  RTX_OPTION("rtx.water", float, reflectedCausticsHeight, 5.0f, "Maximum receiver height above water in world units.");
  RTX_OPTION("rtx.water", bool, underwaterEnabled, false, "Use native volumetric lighting for the configured camera water medium. Requires water material override and a neutral camera medium; shares the surface transmittance and distance.");
  RTX_OPTION("rtx.water", std::string, cameraMediumMaterialHash, "", "Hexadecimal source material hash supplied by SetCameraMediumMaterial for water. Empty disables water classification; unrelated glass media remain unchanged.");
  RTX_OPTION("rtx.water", float, underwaterScattering, 0.0f, "Scattering fraction of neutral water extinction, 0 to 1: the weakest authored RGB extinction plus added particle extinction. The remaining fraction absorbs light. Water color follows shared transmittance and scene lighting; zero gives absorption only.");
  RTX_OPTION("rtx.water", float, underwaterAnisotropy, 0.0f, "Forward scattering of the water medium, 0 to 0.8. Native volumetric lighting supplies sun shafts and local light scattering.");
  RTX_OPTION("rtx.water", float, volumetricCausticsStrength, 0.0f, "Contrast of caustic shafts from the strongest upward directional light in the underwater volume, 0 to 1. Any positive value selects one shared dynamic pattern for projected ocean receivers and that light in the underwater volume; zero keeps the standard receiver pattern. Uses the ocean caustic scale, speed and randomized animation. Pools and reflected caustic patterns retain their own paths. Bounded contrast recovery counters texture filtering; local volume gain is capped at eight. Source-mean normalization does not guarantee total energy conservation under animated warps or mip contrast recovery. Receiver brightness retains its existing pattern bounds, and a black source adds no surface caustics. Requires enabled receiver caustics and a registered GPU water surface. Animated shafts use a shorter lighting history and may show more noise.");
  WaterVolumeCausticsArgs m_volumeCausticsArgs {};
  bool m_volumeCausticsActive = false;
  bool m_volumeCausticsChanged = false;
  WaterLightSurface m_causticLightSurface;
  Rc<DxvkImageView> m_volumeCausticPattern;
  uint64_t m_volumeCausticSourceCookie = 0;
  Rc<DxvkBuffer> m_volumeCausticSurface;
  WaterMediumState m_cameraMedium;
  WaterLightSurface m_cameraLightSurface;
  void prepareCausticTexture(RtxContext* ctx);
  TextureRef m_causticTexture;
  TextureRef m_foamTexture;
  TextureRef m_foamCompositeTexture;
  std::string m_foamCompositePath;
  std::string m_foamColorPath;
  std::string m_foamNormalPath;
  TextureRef m_foamNormalTexture;
  // GPU foam geometry lifetime. Declared as unique_ptr so the header does not need the geometry
  // implementation; the instance outlives the registered mesh handle and is destroyed
  // only after the draw is retired.
  std::unique_ptr<class WaterFoamGpuGeometry> m_foamGeometry;
  remixapi_MeshHandle m_foamMeshHandle = nullptr;
  remixapi_MaterialHandle m_foamMaterial = nullptr;
  bool m_foamRegistered = false;
  // Low-frequency CPU-side diagnostic counter (frame index of the last log).
  uint64_t m_foamDiagFrame = 0;
  uint32_t m_foamArgsLogCounter = 0;
  uint32_t m_materialPathLogCounter = 0;
  // One-shot failure latch. Missing asset, an empty preload result and a kFailed upload
  // all land here, so each is reported once per enable instead of every frame, and the
  // geometry stops re-scanning for the asset on every frame.
  enum class FoamTextureState { NotRequested, Pending, Ready, Failed };
  FoamTextureState m_foamTextureState = FoamTextureState::NotRequested;
  Resources::Resource m_foamField[2];
  // Absolute water height (R32) and validity (G32), double buffered in lockstep with the
  // density field above. A separate 32-bit resource because the density field is half
  // precision, where a world height quantises to half or whole units - coarser than the
  // surface offset, which would make the foam step and its normals stair-step.
  Resources::Resource m_foamHeight[2];
  Vector2 m_foamOrigin = Vector2(0.0f, 0.0f);
  // Snapshot of the frame that PRODUCED the density the display surface will read.
  // The update pass runs late in the frame and the display is generated early in the
  // next one, so every value tied to that field - origin, camera height, clock - must
  // be saved together. Reading the current camera or clock instead would mix two frames:
  // after a camera jump the box would describe the new position while the vertices came
  // from the old one, and time drift would desynchronise erosion and UV from the density.
  float m_foamSnapshotCameraUp = 0.0f;
  float m_foamSnapshotTime = 0.0f;
  uint32_t m_foamIndex = 0;
  bool m_foamReady = false;
  uint32_t m_shoreWetnessIndex = 0;
  bool m_shoreWetnessReady = false;
  std::string m_causticTexturePath;
  Resources::Resource m_causticRadiance;
  bool m_causticsPrepared = false;
  void* m_mapping = nullptr;
  void* m_mutex = nullptr;
  remixapi::water::Controls* m_controls = nullptr;
};
}
