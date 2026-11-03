// Adapted from xoxor4d/dxvk-remix, commit 7f5210e1a7cd1356bf0271230f82e6132b08ad8c.
// Modified for device-owned resources, frame validity and configurable world orientation and units.
// The original zlib/libpng license is retained in ThirdPartyLicenses.txt.
#pragma once

#include "rtx_common_object.h"
#include "rtx_resources.h"

struct RaytraceArgs;

namespace dxvk {

  class RtxContext;

  class RtxRainMap : public CommonDeviceObject {
  public:
    explicit RtxRainMap(DxvkDevice* device) : CommonDeviceObject(device) { }

    // Snapshots configuration and the camera window before frame constants are uploaded.
    void prepare(RtxContext& ctx, RaytraceArgs& constants);

    // Requires the current TLAS and constants. Runs before wet surface and contact-particle consumers.
    void dispatch(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);

    // Returns the map matching the most recently prepared constants, or a defined miss texture.
    Rc<DxvkImageView> view(RtxContext& ctx);

  private:
    void ensureResources(RtxContext& ctx, uint32_t resolution);
    Rc<DxvkImageView> placeholderView(RtxContext& ctx);

    Resources::Resource m_height;
    Resources::Resource m_map;
    Resources::Resource m_placeholder;
    uint32_t m_resolution = 0;
    uint32_t m_preparedFrame = 0;
    uint32_t m_dispatchedFrame = 0;
    bool m_enabled = false;
    bool m_traced = false;
  };

} // namespace dxvk
