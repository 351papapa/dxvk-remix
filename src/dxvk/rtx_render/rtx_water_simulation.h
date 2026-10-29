#pragma once
#include "rtx_common_object.h"
#include "rtx_types.h"
#include "../../util/sync/sync_signal.h"
#include "../../../public/include/remixapi/water_gpu_simulation.h"

namespace dxvk {
class WaterGpuSimulation : public CommonDeviceObject {
public:
  explicit WaterGpuSimulation(DxvkDevice* device) : CommonDeviceObject(device) { }
  ~WaterGpuSimulation();
  bool prepare(const Rc<DxvkContext>& ctx, bool enabled, remixapi::water::gpu::Frame& frame);
  const Rc<DxvkBuffer>& heights() const { return m_state[m_parity]; }
  bool hullActive() const { return m_hullActive; }
  float simulationTime() const { return m_valid ? m_simulationTime : -1.0f; }
  const Rc<DxvkBuffer>& hullHeights() const { return m_hullState[m_hullParity]; }
  const Rc<DxvkBuffer>& hullDomain() const { return m_hullDomain; }
  void submitFoam(const Rc<DxvkContext>& ctx, const Rc<DxvkBuffer>& input, const Rc<DxvkBuffer>& positions, bool enabled,
                  float verticalDisplacementPadding = 0.0f);
private:
  void ensureResources();
  void prepareHull(const Rc<DxvkContext>& ctx, const remixapi::water::simulation::FlowFrame& frame, bool reset);
  Rc<DxvkBuffer> m_hullState[2], m_hullPressure, m_hullForces, m_hullDomain;
  uint32_t m_hullParity=0;
  float m_hullX=0,m_hullY=0,m_hullBudget=0;
  bool m_hullActive=false;
  void diagnoseFlow(const Rc<DxvkContext>& ctx);
  Rc<DxvkBuffer> m_flowReadback;
  Rc<sync::Fence> m_flowFence = new sync::Fence {};
  uint64_t m_flowSignal = 0;
  uint32_t m_flowReadTick = 0, m_flowSourceEvents = 0, m_flowForceEvents = 0;
  bool m_flowReadPending = false;
  void prepareFlow(const Rc<DxvkContext>& ctx, const remixapi::water::simulation::FlowFrame& frame, bool reset);
  Rc<DxvkBuffer> m_flowState[2], m_flowPressure[2], m_flowDivergence, m_flowSources, m_flowForces, m_waterMap;
  uint32_t m_flowParity = 0, m_pressureParity = 0, m_flowReset = 0;
  float m_flowX = 0, m_flowY = 0, m_flowBudget = 0, m_waterMin = 0, m_waterMax = 0;
  bool m_flowActive = false, m_flowMesh = false, m_mapUploaded = false;
  void* m_mapping = nullptr;
  void* m_mutex = nullptr;
  remixapi::water::simulation::Shared* m_shared = nullptr;
  remixapi::water::gpu::Frame m_surface {};
  Rc<DxvkBuffer> m_state[2], m_mask, m_impulses, m_foamSource, m_foamTarget, m_foamIndices;
  uint32_t m_parity = 0, m_epoch = 0, m_resetSerial = 0, m_foamSequence = 0;
  uint32_t m_vertexCount = 0, m_indexCount = 0;
  float m_originX = 0, m_originY = 0, m_spacing = 0;
  float m_simulationTime = 0.0f;
  bool m_valid = false, m_registered = false;
  uint64_t m_foamGeneration = 0, m_topologyHash = 0;
  Vector3 m_min {}, m_max {};
};
}
