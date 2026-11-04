#include "rtx_water.h"
#include "rtx_water_visibility.h"
#include "rtx_numos_weather.h"
#include "rtx_camera.h"
#include "rtx_imgui.h"
#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_options.h"
#include "rtx_asset_replacer.h"
#include "rtx_water_foam_geometry.h"
#include "../../util/util_global_time.h"
#include "rtx_asset_data_manager.h"
#include "rtx_texture_manager.h"
#include "imgui/imgui.h"
#include <Windows.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <system_error>

namespace dxvk {
namespace {
struct CameraWaterParticleOptions {
  RTX_OPTION("rtx.water", float, underwaterRainParticleExtinctionPerMeter, 0.0f,
    "Additional neutral extinction per meter for a submerged camera in rain, from 0 to 1. Zero keeps the dry water clarity. The particle scattering fraction controls how much light is scattered.");
  RTX_OPTION("rtx.water", float, underwaterStormParticleExtinctionPerMeter, 0.0f,
    "Additional neutral extinction per meter for a submerged camera in a storm, from 0 to 1. The value blends with rain or clear weather. Zero keeps the dry water clarity.");
};
}

WaterMediumState getWeatherCameraWaterMedium(const WaterMediumState& medium, const NumosWeather& weather) {
  if (!medium.active) {
    return medium;
  }
  const float rain = CameraWaterParticleOptions::underwaterRainParticleExtinctionPerMeter();
  const float storm = CameraWaterParticleOptions::underwaterStormParticleExtinctionPerMeter();
  if (rain == 0.0f && storm == 0.0f) {
    return medium;
  }
  return applyWaterWeatherParticles(medium, weather.waterState(), rain, storm,
    RtxOptions::getMeterToWorldUnitScale(), WaterSystem::underwaterScattering());
}

WaterOpticalProperties WaterSystem::opticalProperties() {
  const float requestedParticles = particleExtinctionPerMeter();
  const float particlesPerMeter = std::isfinite(requestedParticles) ? std::clamp(requestedParticles, 0.0f, 1.0f) : 0.0f;
  const float worldUnitsPerMeter = RtxOptions::getMeterToWorldUnitScale();
  // Invalid scene scales must not turn clear water into an infinitely dense medium.
  const float particlesPerWorldUnit = std::isfinite(worldUnitsPerMeter) && worldUnitsPerMeter > 0.0f
    ? particlesPerMeter / (std::max)(worldUnitsPerMeter, 0.000001f) : 0.0f;
  return waterOpticalProperties(transmittanceColor(), transmittanceDistance(), particlesPerWorldUnit);
}

void WaterSystem::updateCameraMedium(const SceneManager& scene) {
  // Latched after scene preparation: no option writes and no glass-as-water heuristic.
  m_cameraMedium = makeWaterMediumState(enabled() && underwaterEnabled() && overrideMaterial(),
    parseWaterMediumHash(cameraMediumMaterialHash()), scene.getStartInMediumMaterialSourceHash(),
    opticalProperties(),
    underwaterScattering(), underwaterAnisotropy(), m_cameraMedium);
  WaterLightSurface surface;
  surface.valid = m_cameraMedium.active && RtxOptions::zUp() && gpuSurface() && m_gpuSurface != nullptr
    && m_gpuSurface->surfaceHeightAt(scene.getCamera().getPosition(), surface.height);
  m_cameraMedium.changed |= surface.valid != m_cameraLightSurface.valid
    || (surface.valid && surface.height != m_cameraLightSurface.height);
  m_cameraLightSurface = surface;
  if (m_cameraMedium.changed) {
    Logger::info(str::format("[Water medium] active=", m_cameraMedium.active,
      " extinction=", m_cameraMedium.extinction.x, ",", m_cameraMedium.extinction.y, ",", m_cameraMedium.extinction.z,
      " scattering=", m_cameraMedium.scattering.x, " g=", m_cameraMedium.anisotropy,
      " surfaceValid=", m_cameraLightSurface.valid, " surfaceHeight=", m_cameraLightSurface.height));
  }
}

std::string WaterSystem::resolveAssetPath(const std::string& path) {
  if (path.empty()) { return {}; }
  std::filesystem::path asset = std::filesystem::u8path(path);
  if (asset.is_absolute()) { return asset.lexically_normal().u8string(); }
  wchar_t executable[32768] = {};
  GetModuleFileNameW(nullptr, executable, 32768);
  auto folder = std::filesystem::path(executable).parent_path();
  if (_wcsicmp(folder.filename().c_str(), L".trex") == 0) { folder = folder.parent_path(); }
  return (folder / asset).lexically_normal().u8string();
}

WaterSystem::~WaterSystem() {
  if (m_controls) { UnmapViewOfFile(m_controls); }
  if (m_mapping) { CloseHandle(m_mapping); }
  if (m_mutex) { CloseHandle(m_mutex); }
}

bool WaterSystem::ownsFoamGeometry(const RasterGeometry& geometry) const {
  return m_foamGeometry != nullptr && m_foamRegistered && geometry.externalMesh == m_foamMeshHandle
    && m_foamGeometry->ownsGeometry(geometry);
}

bool WaterSystem::ownsFoamVertexBuffer(const Rc<DxvkBuffer>& buffer) const {
  return m_foamGeometry != nullptr && m_foamGeometry->ownsVertexBuffer(buffer);
}

void WaterSystem::publishControls(const RtCamera& camera) {
  if (!m_mutex) { m_mutex = CreateMutexW(nullptr, FALSE, remixapi::water::kMutexName); }
  if (!m_mapping) { m_mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(remixapi::water::Controls), remixapi::water::kMappingName); }
  if (m_mapping && !m_controls) { m_controls = static_cast<remixapi::water::Controls*>(MapViewOfFile(m_mapping, FILE_MAP_WRITE, 0, 0, sizeof(remixapi::water::Controls))); }
  if (!m_controls || !m_mutex) { return; }
  const DWORD wait = WaitForSingleObject(m_mutex, 0);
  if (wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED) { return; }
  remixapi::water::Controls controls;
  controls.tick = GetTickCount();
  controls.enabled = enabled() ? 1u : 0u;
  controls.amplitude = std::clamp(amplitude(), 0.0f, 2.0f);
  controls.wavelength = std::clamp(wavelength(), 6.0f, 40.0f);
  controls.speed = std::clamp(speed(), 0.0f, 3.0f);
  controls.steepness = std::clamp(steepness(), 0.0f, 0.9f);
  controls.interactionStrength = std::clamp(interactionStrength(), 0.0f, 5.0f);
  controls.interactionRadius = std::clamp(interactionRadius(), 0.5f, 3.0f);
  controls.propagationSpeed = std::clamp(propagationSpeed(), 1.0f, 8.0f);
  controls.damping = std::clamp(damping(), 0.05f, 3.0f);
  controls.reduceOffscreen = reduceOffscreen() ? 1u : 0u;
  controls.simulationRadius = std::clamp(simulationRadius(), 16.0f, 96.0f);
  const auto clip = camera.getViewToProjectionf() * camera.getWorldToViewf();
  std::memcpy(controls.worldToClip, &clip[0][0], sizeof(controls.worldToClip));
  controls.cameraValid = 1;
  *m_controls = controls;
  ReleaseMutex(m_mutex);
}

void WaterSystem::submitGpuSurface(Rc<DxvkContext> ctx) {
  if (!m_gpuSurface && gpuSurface()) { m_gpuSurface = std::make_unique<WaterGpuSurface>(m_device); }
  if (m_gpuSurface) { m_gpuSurface->submit(ctx, enabled() && gpuSurface() && RtxOptions::zUp(), gpuSimulation()); }
}

void WaterSystem::submitFoamSurface(Rc<DxvkContext> ctx) {
  const bool assetsChanged = m_foamColorPath != foamColorTexture() || m_foamNormalPath != foamNormalTexture();
  if (!enabled() || !foamLineMaterialPath() || assetsChanged) {
    if (m_foamRegistered) {
      // Destroy the mesh through SceneManager so the instance and its geometry cache
      // entry retire together; erasing only the AssetReplacer entry would leave the
      // instance behind.
      ctx->getCommonObjects()->getSceneManager().destroyExternalMesh(ctx, m_foamMeshHandle);
      m_foamRegistered = false;
    }
    // Retire the material we registered as well. Leaving it in the registry would make a
    // later re-enable hit the duplicate-handle path in makeMaterialWithTexturePreload and
    // silently keep the old configuration.
    if (m_foamMaterial != nullptr) {
      ctx->getCommonObjects()->getSceneManager().getAssetReplacer()->destroyExternalMaterial(m_foamMaterial);
      m_foamMaterial = nullptr;
    }
    m_foamTexture = TextureRef();
    m_foamNormalTexture = TextureRef();
    m_foamTextureState = FoamTextureState::NotRequested;
    // Reset the diagnostic clocks too, so a re-enable starts from a deterministic frame and
    // phase instead of resuming mid-cycle.
    m_foamDiagFrame = 0;
    m_foamGeometry.reset();
    m_foamMeshHandle = nullptr;
    m_foamColorPath = foamColorTexture();
    m_foamNormalPath = foamNormalTexture();
    if (!enabled() || !foamLineMaterialPath()) { return; }
  }

  // Frame parameters are taken ONCE here and reused for the dispatch, the bounding box
  // and the draw, so the vertices the GPU writes and the box the acceleration structure
  // gets cannot disagree.
  //
  // The placement is no longer a fixed test position. The display surface covers exactly
  // the density domain that the foam update pass owns, so the foam line is generated on
  // top of the density that produced it. The update pass runs late in the frame, after
  // the acceleration structure for this frame has been built, so this reads the buffer
  // it completed last frame together with the origin that buffer was built against -
  // a deliberate one-frame lag, and the reason the origin is carried alongside the field
  // rather than recomputed from the camera here.
  if (!m_foamReady) {
    // No completed field yet. Submitting now would describe a domain that does not
    // exist, so skip until the update pass has produced one.
    return;
  }
  WaterFoamGpuGeometry::FrameParams params {};
  params.zUp = RtxOptions::zUp();
  params.gridOriginX = m_foamOrigin.x;
  params.gridOriginY = m_foamOrigin.y;
  params.gridSpacing = kWaterFoamFieldSpacing;
  // User strength is a coverage control, not a brightness: it scales how much of the
  // foam line is drawn rather than how bright it is, which is what keeps the lighting
  // coming from the renderer instead of from a fudge factor.
  params.coverageScale = std::clamp(shoreFoamStrength(), 0.0f, 1.0f);
  params.textureScale = std::clamp(shoreFoamTextureScale(), 0.25f, 12.0f);
  params.surfaceOffset = kWaterFoamSurfaceOffset;
  // Snapshot values, not current ones. The density being read was produced last frame, so
  // the clock, the camera height it was probed from and its origin all come from the saved
  // snapshot. Reading the live camera or clock here would mix two frames: after a camera
  // jump the bounding box would describe the new position while the vertices came from the
  // old one, and the erosion/UV clock would drift away from the density it modulates.
  params.time = m_foamSnapshotTime;
  params.cameraUp = m_foamSnapshotCameraUp;
  params.frustumCulling = shoreFoamFrustumCulling();
  const auto& foamCamera = ctx->getCommonObjects()->getSceneManager().getCamera();
  params.worldToClip = foamCamera.getViewToProjectionf() * foamCamera.getWorldToViewf();
  // Height range from the probe's own geometry rather than an assumed sea level. The probe
  // fires from cameraUp + 64 downward with its ray clamped to 256, so every surface it can
  // record lies in [cameraUp - 192, cameraUp + 64]. The box must cover that bracket; the
  // camera height is the probe's reference, not a claim about where the water is.
  params.waterHeightCentre = m_foamSnapshotCameraUp;
  params.waterHeightHalfRange = (std::max)(kWaterFoamProbeUpReach, kWaterFoamProbeDownReach);
  if (!std::isfinite(params.gridOriginX) || !std::isfinite(params.gridOriginY)
   || !std::isfinite(params.gridSpacing) || params.gridSpacing <= 0.0f) {
    return;
  }

  if (m_foamGeometry == nullptr) {
    m_foamGeometry = std::make_unique<WaterFoamGpuGeometry>(m_device);
  }
  if (!m_foamGeometry->ensureTopology(ctx)) {
    // Topology initialisation failed: do not submit a draw with no valid geometry.
    return;
  }

  // Diagnostic state is reported BEFORE the early returns below: otherwise a pending or
  // failed texture returns first and the log never shows why nothing is drawn.

  // Texture state machine. Three outcomes are distinguished: not requested yet, pending
  // (requested and the upload has not landed), and failed. Only transitions and failures
  // are logged, so there is no per-frame scan and no repeated warning.
  if (m_foamTextureState == FoamTextureState::NotRequested) {
    // Resolve an absolute candidate path for the log before requesting it. This is only
    // the loose-DDS resolution candidate, not proof that any packaged asset resolves this
    // way. Absolute paths are computed rather than hardcoded so this works on any machine.
    const std::string kFoamTextureRelative = resolveAssetPath(m_foamColorPath);
    std::error_code fsError;
    const std::filesystem::path foamTextureAbsolute = std::filesystem::absolute(kFoamTextureRelative, fsError);
    const std::string foamTextureAbsoluteStr = fsError ? std::string("<path resolution failed: ") + fsError.message() + ">" : foamTextureAbsolute.string();
    auto asset = AssetDataManager::get().findAsset(kFoamTextureRelative);
    if (asset == nullptr) {
      m_foamTextureState = FoamTextureState::Failed;
      Logger::warn(str::format("[WaterFoam] foam texture asset not found. Candidate absolute path: ",
                   foamTextureAbsoluteStr, " (loose-DDS resolution candidate relative to the host executable directory). "
                   "Foam inactive until the switch is toggled off and on again."));
    } else {
      m_foamTexture = TextureRef(ctx->getCommonObjects()->getTextureManager().preloadTextureAsset(asset, ColorSpace::AUTO, true));
      auto normalAsset = AssetDataManager::get().findAsset(resolveAssetPath(m_foamNormalPath));
      if (normalAsset != nullptr) {
        m_foamNormalTexture = TextureRef(ctx->getCommonObjects()->getTextureManager().preloadTextureAsset(normalAsset, ColorSpace::AUTO, true));
      } else {
        Logger::warn("[WaterFoam] missing foam normal texture; set rtx.water.foamNormalTexture");
      }
      const Rc<ManagedTexture>& managed = m_foamTexture.getManagedTexture();
      if (managed.ptr() == nullptr) {
        m_foamTextureState = FoamTextureState::Failed;
        Logger::warn("[WaterFoam] foam texture preload returned no managed texture; "
                     "foam inactive until the switch is toggled off and on again.");
      } else {
        m_foamTextureState = FoamTextureState::Pending;
        Logger::info("[WaterFoam][CPU] foam texture requested; waiting for upload");
      }
    }
  }

  if (m_foamTextureState == FoamTextureState::Pending) {
    const Rc<ManagedTexture>& managed = m_foamTexture.getManagedTexture();
    const Rc<ManagedTexture>& normalManaged = m_foamNormalTexture.getManagedTexture();
    if (normalManaged.ptr() == nullptr || normalManaged->m_state.load() == ManagedTexture::State::kFailed) {
      m_foamTextureState = FoamTextureState::Failed;
      Logger::warn("[WaterFoam] normal texture missing or failed; set rtx.water.foamNormalTexture and toggle foam to retry");
    } else if (managed.ptr() != nullptr && managed->m_state.load() == ManagedTexture::State::kFailed) {
      m_foamTextureState = FoamTextureState::Failed;
      Logger::warn("[WaterFoam] foam texture upload failed; "
                   "foam inactive until the switch is toggled off and on again.");
    } else if (!m_foamTexture.isImageEmpty() && !m_foamNormalTexture.isImageEmpty()) {
      m_foamTextureState = FoamTextureState::Ready;
      Logger::info("[WaterFoam][CPU] surf foam color and tangent normal textures ready; roughness=0.90");
    }
  }

  if (m_foamTextureState != FoamTextureState::Ready) {
    // Pending and Failed both land here: no re-request, no repeated warning, no blocking.
    return;
  }


  // Collect any completed GPU samples. Non-blocking: only slots whose fence value has
  // been reached are read, and nothing here waits on the GPU.
  if (m_foamGeometry != nullptr) {
    WaterFoamGpuGeometry::ReadbackSample sample {};
    while (m_foamGeometry->tryConsumeReadback(sample)) {
      const float tx = sample.translation[0];
      const float ty = sample.translation[1];
      const float tz = sample.translation[2];
      bool attributesValid = true;
      for (uint32_t v = 0; v < 3; ++v) {
        WaterFoamGpuVertex vertex {};
        std::memcpy(vertex.position, sample.position[v], sizeof(vertex.position));
        std::memcpy(vertex.normal, sample.normal[v], sizeof(vertex.normal));
        std::memcpy(vertex.texcoord, sample.texcoord[v], sizeof(vertex.texcoord));
        vertex.color = sample.color[v];
        attributesValid &= waterFoamVertexAttributesValid(vertex);
      }
      if (!attributesValid) { Logger::err("[WaterFoam] invalid sampled GPU vertex attributes"); }

      Logger::info(str::format(
        "[WaterFoam][GPU readback] issuedFrame=", sample.frame,
        " generation=", sample.generation,
        " zUp=", sample.zUp ? 1 : 0,
        " issuedPlacement=(", tx, ",", ty, ",", tz, ")",
        " v0=(", sample.position[0][0], ",", sample.position[0][1], ",", sample.position[0][2], ")",
        " vMid=(", sample.position[1][0], ",", sample.position[1][1], ",", sample.position[1][2], ")",
        " vLast=(", sample.position[2][0], ",", sample.position[2][1], ",", sample.position[2][2], ")",
        " n0=(", sample.normal[0][0], ",", sample.normal[0][1], ",", sample.normal[0][2], ")",
        " uv0=(", sample.texcoord[0][0], ",", sample.texcoord[0][1], ")",
        " colorA0=", (sample.color[0] >> 24) & 0xFFu,
        " sampledVertices=3 attributesValid=", attributesValid,
        " pendingSlots=", m_foamGeometry->pendingReadbackCount(),
        "  (source vertices only: does not prove internal cache/BVH or shading consumed them)"));
    }
  }

  // A rough dielectric with textured coverage: lighting comes from the scene, not emission.
  if (m_foamMaterial == nullptr) {

    OpaqueMaterialData opaque = LegacyMaterialData::createDefault();
    opaque.setAlbedoOpacityTexture(m_foamTexture);
    opaque.setNormalTexture(m_foamNormalTexture);
    // The texture replaces this constant when resident; keep the fallback foam-white too.
    opaque.setAlbedoConstant(Vector3(0.94f));
    opaque.setOpacityConstant(1.0f);
    opaque.setRoughnessConstant(0.90f);
    opaque.setMetallicConstant(0.0f);
    opaque.setEnableEmission(false);
    opaque.setEmissiveIntensity(0.0f);
    opaque.setEmissiveColorConstant(Vector3(0.0f));
    opaque.setEnableThinFilm(false);
    opaque.setAlphaIsThinFilmThickness(false);
    opaque.setSpriteSheetRows(1);
    opaque.setSpriteSheetCols(1);
    opaque.setSpriteSheetFPS(0);
    // Alpha-blend classification also keeps foam out of its own opaque shoreline probes.
    opaque.setUseLegacyAlphaState(false);
    opaque.setBlendEnabled(true);
    opaque.setBlendType(BlendType::kAlpha);
    opaque.setInvertedBlend(false);
    opaque.setAlphaTestType(AlphaTestType::kGreater);
    opaque.setAlphaTestReferenceValue(1);
    opaque.setFilterMode(lss::Mdl::Filter::Linear);
    opaque.setWrapModeU(lss::Mdl::WrapMode::Repeat);
    opaque.setWrapModeV(lss::Mdl::WrapMode::Repeat);

    const remixapi_MaterialHandle materialHandle = reinterpret_cast<remixapi_MaterialHandle>(kWaterFoamPrototypeMaterialHandleValue);
    AssetReplacer* replacer = ctx->getCommonObjects()->getSceneManager().getAssetReplacer().get();
    replacer->makeMaterialWithTexturePreload(*ctx, materialHandle, MaterialData { opaque });
    // Confirm registration before treating it as usable; the call itself returns void and
    // silently does nothing on a repeated handle.
    if (replacer->accessExternalMaterial(materialHandle) == nullptr) {
      Logger::warn("[WaterFoam] material registration failed; prototype will not draw");
      return;
    }
    m_foamMaterial = materialHandle;
  }

  // Low-frequency CPU-side diagnostic. Every value here is a CPU observation: the
  // dispatch and submit results are RECORDING INTENT, not proof that the GPU wrote
  // anything. Named accordingly so it cannot be mistaken for GPU evidence.
  if ((m_foamDiagFrame++ % 600ull) == 0ull) {
    Logger::info(str::format(
      "[WaterFoam][CPU] zUp=", params.zUp ? 1 : 0,
      " origin=(", params.gridOriginX, ",", params.gridOriginY, ")",
      " extent=", kWaterFoamSheetExtent,
      " textureState=", static_cast<int>(m_foamTextureState),
      " textureReady=", m_foamTexture.isImageEmpty() ? 0 : 1,
      " materialRegistered=", m_foamMaterial != nullptr ? 1 : 0,
      " generation=", m_foamGeometry ? m_foamGeometry->contentGeneration() : 0ull,
      " registered=", m_foamRegistered ? 1 : 0,
      " vertices=", m_foamGeometry ? m_foamGeometry->vertexCount() : 0u,
      " indices=", m_foamGeometry ? m_foamGeometry->indexCount() : 0u,
      " (CPU recording intent only; not GPU write confirmation)"));
  }

  // Generate before submission so the snapshot references this frame's buffer.
  m_foamGeometry->dispatchSheetUpdate(ctx, params, m_foamField[m_foamIndex], m_foamHeight[m_foamIndex],
    m_gpuSurface ? m_gpuSurface->waveInputBuffer() : nullptr,
    m_gpuSurface ? m_gpuSurface->positionsBuffer() : nullptr,
    m_gpuSurface ? m_gpuSurface->shoreDepthBuffer() : nullptr,
    m_gpuSurface ? m_gpuSurface->renderVerticesBuffer() : nullptr);
  // Sample the generated vertices occasionally, and only when a slot is free. The fence
  // is polled on later frames; nothing here waits for the GPU.
  if (localFlowDiagnostics() && (m_foamDiagFrame % 120ull) == 0ull) {
    m_foamGeometry->tryIssueReadback(ctx, params, m_foamDiagFrame);
  }
  auto snapshot = m_foamGeometry->buildGeometrySnapshot(m_foamMaterial, params,
    params.zUp && m_gpuSurface ? m_gpuSurface->topologyHash() : 0);

  AssetReplacer* assets = ctx->getCommonObjects()->getSceneManager().getAssetReplacer().get();
  if (!m_foamRegistered) {
    m_foamMeshHandle = reinterpret_cast<remixapi_MeshHandle>(kWaterFoamPrototypeMeshHandleValue);
    assets->registerExternalMesh(m_foamMeshHandle, std::vector<RasterGeometry>{ *snapshot });
    m_foamRegistered = true;
  } else {
    // Stable handle, new metadata: the registry entry is replaced so the changed content
    // version reaches the cache/BVH update path. If the handle is somehow no longer
    // registered, re-register instead of submitting a draw with no geometry.
    if (!assets->registerGeneratedMesh(m_foamMeshHandle, std::vector<RasterGeometry>{ *snapshot })) {
      assets->registerExternalMesh(m_foamMeshHandle, std::vector<RasterGeometry>{ *snapshot });
    }
  }

  // The sheet's vertices are already in world space, so the instance transform is
  // IDENTITY. Applying the translation again here would move it twice.
  auto state = std::make_unique<ExternalDrawState>();
  state->drawCall.cameraType = CameraType::Main;
  state->drawCall.modifyTransformData().objectToWorld = Matrix4();
  state->drawCall.modifyTransformData().texgenMode = TexGenMode::None;
  state->drawCall.modifyCategoryFlags().set(InstanceCategories::IgnoreAntiCulling);
  // OMM cache keys do not include this sheet's changing vertex opacity and UVs.
  state->drawCall.modifyCategoryFlags().set(InstanceCategories::IgnoreOpacityMicromap);
  state->drawCall.modifyMaterialData().alphaTestEnabled = true;
  state->drawCall.modifyMaterialData().alphaTestReferenceValue = 1;
  state->drawCall.modifyMaterialData().alphaTestCompareOp = VK_COMPARE_OP_GREATER;
  // Self-exclusion chain, verified in source rather than inferred from parameter names:
  // this draw sets useLegacyAlphaState = false, so rtx_instance_manager.cpp:751-752 takes
  // blendEnabled from the MATERIAL (getBlendEnabled, which we set true in the material
  // above) rather than from this draw call's enableBlending; :892 derives
  // isBlendingDisabled = !blendEnabled; :1305 then assigns OBJECT_MASK_ALPHA_BLEND instead
  // of OBJECT_MASK_OPAQUE; and foamContact() only commits candidates from
  // OBJECT_MASK_OPAQUE (water_foam_common.slangh:32). The foam mesh is therefore invisible
  // to its own contact probe and cannot re-seed itself.
  // forceAlphaTest, which would force blendEnabled = false, requires the instance to carry
  // AlphaBlendToCutout or HairCards; this draw sets neither.
  state->drawCall.modifyMaterialData().blendMode.enableBlending = true;
  state->drawCall.modifyMaterialData().blendMode.colorSrcFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  state->drawCall.modifyMaterialData().blendMode.colorDstFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  state->drawCall.modifyMaterialData().blendMode.colorBlendOp = VK_BLEND_OP_ADD;
  state->drawCall.modifyMaterialData().blendMode.alphaSrcFactor = VK_BLEND_FACTOR_ONE;
  state->drawCall.modifyMaterialData().blendMode.alphaDstFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  state->drawCall.modifyMaterialData().blendMode.alphaBlendOp = VK_BLEND_OP_ADD;
  state->drawCall.modifyMaterialData().blendMode.writeMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  state->drawCall.modifyMaterialData().textureColorArg1Source = RtTextureArgSource::Texture;
  state->drawCall.modifyMaterialData().textureColorOperation = DxvkRtTextureOperation::SelectArg1;
  state->drawCall.modifyMaterialData().textureAlphaArg1Source = RtTextureArgSource::Texture;
  state->drawCall.modifyMaterialData().textureAlphaArg2Source = RtTextureArgSource::VertexColor0;
  state->drawCall.modifyMaterialData().textureAlphaOperation = DxvkRtTextureOperation::Modulate;
  state->drawCall.modifyMaterialData().tFactor = 0xffffffff;
  // Must stay false: the vertex alpha carries foam coverage, and marking it as baked
  // lighting would make the renderer treat coverage as illumination.
  state->drawCall.modifyMaterialData().isVertexColorBakedLighting = false;
  state->mesh = m_foamMeshHandle;
  state->cameraType = CameraType::Main;
  state->categories.set(InstanceCategories::IgnoreAntiCulling);
  state->categories.set(InstanceCategories::IgnoreOpacityMicromap);
  state->doubleSided = true;

  ctx->getCommonObjects()->getSceneManager().submitExternalDraw(ctx, std::move(state));
}

void WaterSystem::showImguiSettings() {
  constexpr auto flags = ImGuiSliderFlags_AlwaysClamp;
  RemixGui::Checkbox("Enable new water system", &enabledObject());
  ImGui::BeginDisabled(!enabled());
  if (ImGui::CollapsingHeader("Water appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
    RemixGui::Checkbox("Override water material", &overrideMaterialObject());
    RemixGui::ColorEdit3("Transmittance color", &transmittanceColorObject());
    RemixGui::SliderFloat("Transmittance distance", &transmittanceDistanceObject(), 0.1f, 100.0f, "%.2f", flags);
    RemixGui::SliderFloat("Detail wave strength", &detailStrengthObject(), 0.0f, 3.0f, "%.2f", flags);
    RemixGui::SliderFloat("Detail wave scale", &detailScaleObject(), 0.25f, 4.0f, "%.2f", flags);
    RemixGui::SliderFloat("Added particle extinction / m", &particleExtinctionPerMeterObject(), 0.0f, 1.0f, "%.4f", flags);
    ImGui::TextWrapped("Detail waves control surface distortion and reflections. Refraction also follows the water material index of refraction.");
    ImGui::TextWrapped("Transmittance and suspended particles are shared by the water surface, underwater lighting and caustics.");
  }
  if (ImGui::CollapsingHeader("Waves and interactions")) {
    RemixGui::SliderFloat("Wave amplitude", &amplitudeObject(), 0.0f, 2.0f, "%.2f", flags);
    RemixGui::SliderFloat("Ocean wave size", &oceanWaveStrengthObject(), 0.0f, 3.0f, "%.2fx", flags);
    RemixGui::SliderFloat("Wavelength", &wavelengthObject(), 6.0f, 40.0f, "%.1f", flags);
    RemixGui::SliderFloat("Wave speed", &speedObject(), 0.0f, 3.0f, "%.2f", flags);
    RemixGui::SliderFloat("Wave steepness", &steepnessObject(), 0.0f, 0.9f, "%.2f", flags);
    RemixGui::SliderFloat("Rainy wave multiplier", &rainWaveMultiplierObject(), 1.0f, 3.0f, "%.2fx", flags);
    RemixGui::SliderFloat("Stormy wave multiplier", &stormWaveMultiplierObject(), 1.0f, 3.0f, "%.2fx", flags);
    RemixGui::Checkbox("GPU beach waves", &beachWavesObject());
    RemixGui::SliderFloat("Beach wave height", &beachWaveHeightObject(), 0.0f, 1.5f, "%.2f", flags);
    RemixGui::SliderFloat("Beach wavelength", &beachWaveLengthObject(), 4.0f, 30.0f, "%.1f", flags);
    RemixGui::SliderFloat("Beach wave period", &beachWavePeriodObject(), 2.0f, 15.0f, "%.1f sec", flags);
    RemixGui::SliderFloat("Beach wave depth", &beachWaveDepthObject(), 0.5f, 8.0f, "%.2f", flags);
    RemixGui::SliderFloat("Interaction strength", &interactionStrengthObject(), 0.0f, 5.0f, "%.2f", flags);
    RemixGui::SliderFloat("Interaction radius", &interactionRadiusObject(), 0.5f, 3.0f, "%.2f", flags);
    RemixGui::Checkbox("GPU boat waves: surface height and normals", &hullWavesObject());
    RemixGui::SliderFloat("Boat wave strength", &hullWaveStrengthObject(), 0.f, 3.f);
    RemixGui::Checkbox("GPU local wake flow", &localFlowObject());
    RemixGui::SliderFloat("Boat flow strength", &localFlowStrengthObject(), 0.f, 4.f);
    RemixGui::SliderFloat("Boat wake swirl", &localFlowSwirlObject(), 0.f, 4.f);
    ImGui::TextWrapped("Beach waves follow terrain depth. Their height is independent of the local wave amplitude.");
  }
  if (ImGui::CollapsingHeader("Foam and wet shore")) {
    RemixGui::Checkbox("GPU crest and contact foam", &shoreFoamEnabledObject());
    RemixGui::Checkbox("Contact foam", &contactFoamObject());
    RemixGui::SliderFloat("Shore foam width", &shoreFoamWidthObject(), 0.1f, 2.0f, "%.2f", flags);
    RemixGui::SliderFloat("Shore foam coverage", &shoreFoamStrengthObject(), 0.0f, 1.0f, "%.2f", flags);
    RemixGui::SliderFloat("Beach foam width", &shoreFoamRunupWidthObject(), 0.0f, 6.0f, "%.2f", flags);
    RemixGui::SliderFloat("Foam wave response", &shoreFoamWaveResponseObject(), 0.0f, 8.0f, "%.2f", flags);
    RemixGui::SliderFloat("Water foam lifetime", &shoreFoamLifetimeObject(), 0.5f, 15.0f, "%.1f sec", flags);
    RemixGui::SliderFloat("Crest foam strength", &crestFoamStrengthObject(), 0.0f, 6.0f, "%.2f", flags);
    RemixGui::SliderFloat("Foam texture scale", &shoreFoamTextureScaleObject(), 0.25f, 12.0f, "%.2f", flags);
    RemixGui::SliderFloat("Beach wetness", &shoreWetnessStrengthObject(), 0.0f, 1.0f, "%.2f", flags);
    RemixGui::SliderFloat("Beach drying time", &shoreWetnessLifetimeObject(), 1.0f, 120.0f, "%.1f sec", flags);
    ImGui::TextWrapped("Whitecaps follow ocean wave crests. Foam stays on water; exposed beach terrain retains a drying wet sheen.");
  }
  if (ImGui::CollapsingHeader("Caustics and underwater lighting")) {
    RemixGui::Checkbox("Underwater receiver caustics", &causticsEnabledObject());
    RemixGui::SliderFloat("Caustic strength", &causticsStrengthObject(), 0.0f, 8.0f, "%.2f", flags);
    RemixGui::SliderFloat("Caustic scale", &causticsScaleObject(), 0.5f, 16.0f, "%.2f", flags);
    RemixGui::SliderFloat("Caustic speed", &causticsSpeedObject(), 0.0f, 3.0f, "%.2f", flags);
    RemixGui::Checkbox("Underwater volumetric lighting", &underwaterEnabledObject());
    RemixGui::SliderFloat("Underwater caustic shaft contrast", &volumetricCausticsStrengthObject(), 0.0f, 1.0f, "%.2f", flags);
    RemixGui::SliderFloat("Particle scattering fraction", &underwaterScatteringObject(), 0.0f, 1.0f, "%.2f", flags);
    RemixGui::SliderFloat("Additional rain extinction / m", &CameraWaterParticleOptions::underwaterRainParticleExtinctionPerMeterObject(), 0.0f, 1.0f, "%.3f", flags);
    RemixGui::SliderFloat("Additional storm extinction / m", &CameraWaterParticleOptions::underwaterStormParticleExtinctionPerMeterObject(), 0.0f, 1.0f, "%.3f", flags);
    RemixGui::SliderFloat("Water forward scattering", &underwaterAnisotropyObject(), 0.0f, 0.8f, "%.2f", flags);
    RemixGui::Checkbox("Reflected caustics above water", &reflectedCausticsEnabledObject());
    RemixGui::SliderFloat("Reflected caustic strength", &reflectedCausticsStrengthObject(), 0.0f, 8.0f, "%.2f", flags);
    ImGui::TextWrapped("Caustic scale and speed also control underwater shafts and reflected caustics.");
    ImGui::TextWrapped("Underwater lighting follows the camera water medium. It uses the same water transmittance; overall volumetric quality controls also apply.");
  }
  if (ImGui::CollapsingHeader("Advanced water settings")) {
    if (ImGui::TreeNode("Integration and resources")) {
      RemixGui::Checkbox("GPU water surface", &gpuSurfaceObject());
      RemixGui::Checkbox("GPU interaction waves and foam following (experimental)", &gpuSimulationObject());
      RemixGui::Checkbox("Foam through scene materials", &foamLineMaterialPathObject());
      RemixGui::DragFloat2("Water map world origin", &waterMapOriginObject(), 1.0f);
      RemixGui::SliderFloat("Water map cell size", &waterMapCellSizeObject(), 0.01f, 1024.0f, "%.2f", flags);
      RemixGui::InputText("Camera water material hash", &cameraMediumMaterialHashObject());
      RemixGui::InputText("Beach eligibility mask", &beachMaskPathObject());
      RemixGui::InputText("Foam material color texture", &foamColorTextureObject());
      RemixGui::InputText("Foam material normal texture", &foamNormalTextureObject());
      RemixGui::InputText("Foam composite pattern", &foamCompositeTextureObject());
      RemixGui::InputText("Caustic pattern", &causticsTextureObject());
      ImGui::TextWrapped("These values describe the game integration and its authored resources. Keep them unchanged for an installed water setup.");
      ImGui::TreePop();
    }
    if (ImGui::TreeNode("Simulation and foam tuning")) {
      RemixGui::SliderFloat("Simulation radius", &simulationRadiusObject(), 16.0f, 96.0f, "%.0f world units", flags);
      RemixGui::Checkbox("Reduce offscreen simulation", &reduceOffscreenObject());
      RemixGui::SliderFloat("Wave propagation speed", &propagationSpeedObject(), 1.0f, 8.0f, "%.2f", flags);
      RemixGui::SliderFloat("Wave damping", &dampingObject(), 0.05f, 3.0f, "%.2f", flags);
      RemixGui::SliderFloat("Boat wave propagation", &hullWaveSpeedObject(), 2.f, 8.f);
      RemixGui::SliderFloat("Boat wave damping", &hullWaveDampingObject(), .05f, 2.f);
      RemixGui::SliderFloat("Wake flow drag", &localFlowDragObject(), .05f, 2.f);
      RemixGui::Checkbox("Cull offscreen shore foam", &shoreFoamFrustumCullingObject());
      RemixGui::Checkbox("Ocean foam only", &shoreFoamOceanOnlyObject());
      RemixGui::Checkbox("Exclude player from contact foam", &contactFoamPlayerExclusionObject());
      RemixGui::SliderFloat("Crest foam curvature threshold", &crestFoamThresholdObject(), 0.01f, 0.4f, "%.3f", flags);
      RemixGui::SliderFloat("Shore foam sky ambient", &shoreFoamSkyAmbientObject(), 0.0f, 2.0f, "%.2f", flags);
      ImGui::TextWrapped("The simulation radius follows the player. A larger radius spreads the fixed grid over a larger area and reduces wave detail.");
      ImGui::TreePop();
    }
    if (ImGui::TreeNode("Caustic tuning")) {
      RemixGui::Checkbox("Randomized caustic pattern", &causticsRandomizedObject());
      RemixGui::Checkbox("Separate scale for pools and fountains", &causticsScalePerDomainObject());
      ImGui::BeginDisabled(!causticsScalePerDomain());
      RemixGui::SliderFloat("Caustic scale (non-ocean water)", &causticsPoolScaleObject(), 0.2f, 16.0f, "%.2f", flags);
      ImGui::EndDisabled();
      RemixGui::SliderFloat("Caustic dispersion", &causticsDispersionObject(), 0.0f, 2.0f, "%.2f", flags);
      RemixGui::SliderFloat("Caustic depth", &causticsMaxDepthObject(), 0.5f, 30.0f, "%.2f", flags);
      RemixGui::SliderInt("Caustic local lights", &causticsLocalLightsObject(), 0, 8);
      RemixGui::SliderFloat("Caustic local light range", &causticsLocalLightRangeObject(), 5.0f, 200.0f, "%.1f", flags);
      RemixGui::SliderFloat("Caustic fade start", &causticsFadeStartObject(), 0.0f, 0.9f, "%.2f", flags);
      RemixGui::SliderFloat("Reflected caustic height", &reflectedCausticsHeightObject(), 0.5f, 15.0f, "%.2f", flags);
      ImGui::TextWrapped("Pools and fountains can use a separate authored scale. Zero dispersion uses one pattern sample; positive values use three.");
      ImGui::TreePop();
    }
    if (ImGui::TreeNode("Diagnostics")) {
      RemixGui::Checkbox("GPU flow diagnostics", &localFlowDiagnosticsObject());
      WaterFoamGpuGeometry::showImguiSettings();
      ImGui::TreePop();
    }
  }
  ImGui::EndDisabled();
}
}
