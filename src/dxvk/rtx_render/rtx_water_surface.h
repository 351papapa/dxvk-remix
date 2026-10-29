#pragma once
#include "rtx_common_object.h"
#include <string>
#include "rtx_types.h"
#include "rtx_water_simulation.h"
#include "../../../public/include/remixapi/water_gpu_surface.h"

namespace dxvk {
class WaterGpuSurface : public CommonDeviceObject {
public:
  explicit WaterGpuSurface(DxvkDevice* device) : CommonDeviceObject(device) { }
  ~WaterGpuSurface();
  void submit(const Rc<DxvkContext>& ctx, bool enabled, bool simulationEnabled);
  float simulationTime() const { return m_registered ? m_submittedTime : -1.0f; }
  // Render-thread snapshot of the inputs used for the currently submitted mesh.
  // The Rc keeps the GPU input alive for downstream foam dispatches.
  Rc<DxvkBuffer> waveInputBuffer() const { return m_registered ? m_input : Rc<DxvkBuffer>(); }
  Rc<DxvkBuffer> renderVerticesBuffer() const { return m_registered ? m_vertices : Rc<DxvkBuffer>(); }
  Rc<DxvkBuffer> positionsBuffer() const { return m_registered ? m_positions : Rc<DxvkBuffer>(); }
  Rc<DxvkBuffer> shoreDepthBuffer() const { return m_registered ? m_shoreDepth : Rc<DxvkBuffer>(); }
  Rc<DxvkBuffer> beachMaskBuffer() const { return m_registered ? m_beachMask : Rc<DxvkBuffer>(); }
  uint64_t topologyHash() const { return m_registered ? m_topologyHash : 0; }
  float oceanWaveStrength() const { return m_registered ? m_oceanWaveStrength : 0.0f; }
private:
  void ensureResources(const Rc<DxvkContext>& ctx);
  void updateBeachMask(const Rc<DxvkContext>& ctx);
  void* m_mapping = nullptr;
  void* m_mutex = nullptr;
  remixapi::water::gpu::Frame* m_shared = nullptr;
  remixapi::water::gpu::Frame m_frame {};
  Rc<DxvkBuffer> m_input, m_positions, m_vertices, m_indices, m_shoreDepth;
  Rc<DxvkBuffer> m_shoreTerrain, m_shoreSeeds[2], m_beachMask;
  std::string m_beachMaskPath;
  uint64_t m_generation = 0, m_topologyHash = 0;
  uint64_t m_settingsHash = 0;
  uint32_t m_sequence = 0;
  uint32_t m_shoreLogFrames = 0;
  bool m_registered = false;
  float m_submittedTime = 0.0f;
  float m_oceanWaveStrength = 0.0f;
  std::unique_ptr<WaterGpuSimulation> m_simulation;
};
}
