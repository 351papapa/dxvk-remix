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

namespace dxvk {
  class DxvkDevice;
  class DxvkSampler;

  class DxvkRCAS {
  public:
    struct Options {
      friend class DxvkRCAS;
      friend class ImGUI;

      // This key is not under `rtx.fsr` because it drives the standalone RCAS pass for every
      // non-FSR upscaler and is only forwarded to FSR's built-in RCAS stage.
      RTX_OPTION("rtx.sharpening", float, sharpness, 0.0f,
                 "The amount of RCAS sharpening after upscaling, from 0 (off, no sharpening pass runs) to 1 (maximum). "
                 "Applies to DLSS, DLSS Ray Reconstruction, XeSS and TAA-U through a separate RCAS pass, and is passed "
                 "to FSR's own sharpener when FSR is the upscaler.");
    };

    explicit DxvkRCAS(DxvkDevice* device);
    ~DxvkRCAS();

    void dispatch(
      Rc<RtxContext> ctx,
      const Resources::Resource& inputColor,
      const Resources::Resource& outputColor,
      const Rc<DxvkSampler>& linearSampler,
      float sharpness) const;

  private:
    Rc<vk::DeviceFn> m_vkd;
  };
}
