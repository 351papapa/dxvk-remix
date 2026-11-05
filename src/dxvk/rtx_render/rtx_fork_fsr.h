/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_resources.h"
#include "rtx_options.h"
#include "rtx_common_object.h"

#include "../dxvk_format.h"
#include "../dxvk_include.h"
#include "../util/rc/util_rc.h"
#include "../util/rc/util_rc_ptr.h"

#include <Windows.h>

// FFX SDK header defines FFX_API_ENTRY as dllexport, but we need dllimport
// since we're linking against the prebuilt DLL
#undef FFX_API_ENTRY
#define FFX_API_ENTRY __declspec(dllimport)

// Suppress macro redefinition warnings from FFX headers
#pragma warning(push)
#pragma warning(disable: 4005)

#include <ffx_api/ffx_api.hpp>
#include <ffx_api/ffx_upscale.hpp>
#include <ffx_api/vk/ffx_api_vk.hpp>

#pragma warning(pop)
#undef FFX_API_ENTRY
#define FFX_API_ENTRY

namespace dxvk {

  class DxvkDevice;
  class RtCamera;

  // How FFX should read Remix's primary depth, the device depth of the camera's jittered projection.
  struct FfxDepthConvention {
    bool inverted = false;
    bool infinite = false;
    float cameraNear = 0.0f;
    float cameraFar = 0.0f;
    float viewSpaceToMeters = 1.0f;
  };

  FfxDepthConvention getFfxDepthConvention(const RtCamera& camera);

  // FSR quality presets matching DLSS/XeSS pattern
  enum class FSRPreset : uint32_t {
    UltraPerformance = 0,  // 3.0x upscale
    Performance,           // 2.0x upscale  
    Balanced,              // 1.7x upscale
    Quality,               // 1.5x upscale
    NativeAA,              // 1.0x upscale (FSRAA mode)
    Custom,                // Use resolutionScale setting
    Invalid
  };

  const char* fsrPresetToString(FSRPreset preset);

  /**
   * \brief FSR3 upscaler
   * 
   * AMD FidelityFX Super Resolution 3 implementation following RtxPass architecture
   */
  class DxvkFSR : public RtxPass, public CommonDeviceObject {

  public:

    struct FSROptions {
      friend class DxvkFSR;
      friend class ImGUI;

      RTX_OPTION("rtx.fsr", FSRPreset, preset, FSRPreset::Balanced,
                 "The FSR upscaling preset, which trades quality for performance. 0: Ultra Performance (3x), 1: "
                 "Performance (2x), 2: Balanced (1.7x), 3: Quality (1.5x), 4: Native AA (1x), 5: Custom (uses "
                 "rtx.resolutionScale).");
      // Note: sharpening amount is shared with the standalone RCAS pass and lives
      // at `rtx.sharpening.sharpness` (see DxvkRCAS::Options).
    };

    // Constructor with device
    DxvkFSR(DxvkDevice* device);

    ~DxvkFSR();

    VkExtent3D getInputSize(const VkExtent3D& targetExtent) const;
    
    void dispatch(
      Rc<DxvkContext> renderContext,
      DxvkBarrierSet& barriers,
      const Resources::RaytracingOutput& rtOutput,
      RtCamera& camera,
      bool resetHistory,
      float deltaTimeMs);

    // Public methods needed by RTX context
    void setSetting(const uint32_t displaySize[2], const FSRPreset preset, uint32_t outRenderSize[2]);
    void getInputSize(uint32_t& width, uint32_t& height) const;
    void getOutputSize(uint32_t& width, uint32_t& height) const;
        

  protected:

    // RtxPass interface
    virtual bool isEnabled() const override;
    virtual bool onActivation(Rc<DxvkContext>& ctx) override;
    virtual void onDeactivation() override;

  private:
    float getUpscaleFactor(FSRPreset preset) const;
    float calcUpscaleFactor() const;
    void createFSRContext(const VkExtent3D& targetExtent, const FfxDepthConvention& depthConvention);
    void destroyFSRContext();
    
    // Member variables
    bool m_initialized;
    ffx::Context m_upscalingContext;
    VkExtent3D m_targetExtent;
    FSRPreset m_currentPreset;

    // Resolution tracking
    FSRPreset m_preset;
    FSRPreset m_actualPreset;
    VkExtent2D m_inputSize;
    VkExtent2D m_fsrOutputSize;
    bool m_recreate;
    float m_lastResolutionScale;
    
    // The depth convention the context was created with.
    FfxDepthConvention m_contextDepth;

    // Interval counters for the [FSR] log line.
    uint64_t m_dispatches = 0;
    uint64_t m_resets = 0;
    
    // Manual DLL handle
    HMODULE m_hFFX;
  };

}
