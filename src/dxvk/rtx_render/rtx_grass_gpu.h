#pragma once

#include <unordered_map>
#include <vector>
#include "rtx_resources.h"
#include "rtx/pass/grass/grass_args.h"

namespace dxvk {
class DxvkContext;
struct PointInstancerBatch;

class GrassGpu : public CommonDeviceObject {
public:
  explicit GrassGpu(DxvkDevice* device) : CommonDeviceObject(device) { }
  void setRoots(std::vector<GrassRoot>&& roots);
  void clear();
  void setFrame(const GrassConstants& constants) { m_constants = constants; }
  void dispatch(const Rc<DxvkContext>& ctx, const Rc<DxvkBuffer>& instances,
    const Rc<DxvkBuffer>& surfaces, const Rc<DxvkBuffer>& materials,
    const Rc<DxvkBuffer>& mapping, const std::vector<PointInstancerBatch>& batches);

private:
  struct PreviousBatch { uint32_t frame; uint32_t surface; uint32_t count; };
  std::unordered_map<uint64_t, PreviousBatch> m_previousBatches;
  std::vector<GrassRoot> m_roots;
  GrassConstants m_constants = {};
  bool m_rootsDirty = false;
  bool m_hasHistory = false;
  uint32_t m_lastFrame = 0;
  uint32_t m_previousBudget = 0;
  Rc<DxvkBuffer> m_constantsBuffer;
  Rc<DxvkBuffer> m_rootBuffer;
  Rc<DxvkBuffer> m_candidates;
  Rc<DxvkBuffer> m_state;
  Rc<DxvkBuffer> m_groups;
  Rc<DxvkBuffer> m_selected;
  Rc<DxvkBuffer> m_previous;
};
}
