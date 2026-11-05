#pragma once

// Photometer. Reads the screen center back
// (never waiting on the GPU) and shows, in physical units, the luminance there (cd/m^2), the illuminance on
// the surface (lux, assuming a diffuse surface: pi x luminance / albedo) and the scene EV100. The renderer's
// pre-exposure is divided out; the units come from the sky provider's calibration (NUMOS's
// FrameOutputs::luxPerUnit), so a lamp, a sign and the moon are measured on the same scale.

#include "dxvk_include.h"
#include "rtx_option.h"
#include "rtx_resources.h"

namespace dxvk {

  class DxvkContext;

  class Photometer {
  public:
    RTX_OPTION_FLAG("rtx.photometer", bool, enable, false, RtxOptionFlags::NoSave,
                    "Photometer (dev menu): reads the centre of the screen every frame and shows its luminance "
                    "(cd/m^2), the illuminance on the surface (lux) and the scene EV100, in physical units. Not saved.");

    // After the auto exposure (the final output still holds the pre-tonemap image): queue this frame's readback.
    static void queueReadback(DxvkContext* ctx, const Resources::RaytracingOutput& rtOutput);
    // The readout, plus the host calibration controls. `luxPerUnit`: the sky provider's calibration (0 if none).
    static void showImguiSettings(float luxPerUnit);
  };

}
