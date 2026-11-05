/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include <string>
#include <vulkan/vulkan.h>

#include "rtx_option.h"

// [Session] lines for bug reports, all cheap:
//   - the key settings once at the first frame, and each one again when it changes (menu, rtx.conf, the host,
//     or the runtime turning a feature off);
//   - the output and render resolution when they change;
//   - one status line a minute: frame rate, average and worst frame time, VRAM use and budget, instance and
//     light counts, and the warnings, errors and held-back repeated lines since the previous status line.
// It also applies rtx.log.writeAllRepeats to the logger, names the screenshots rtx.diagnostics.screenshotRequest
// asks for (RtxScreenshot saves them), and writes the session end line.
namespace dxvk {
  class RtxContext;

  class RtxSessionLog {
  public:
    RTX_OPTION("rtx.log", bool, writeAllRepeats, false,
               "Writes every repeated diagnostic line to the log. Off: a kind of tagged line (\"[Tag] word\") is "
               "written for its first 10 lines, then at most once every 5 minutes with a count of the lines held "
               "back. Also on while rtx.profile.detailedLogging is on.");
    RTX_OPTION_FLAG("rtx.diagnostics", std::string, screenshotRequest, std::string(""), RtxOptionFlags::NoSave,
                    "Set by the game integration (remix API SetConfigVariable) to save a screenshot for a bug report. "
                    "At the next present the frame is saved as a PNG named after the value, in the screenshots folder "
                    "of the Remix log folder, a [Mark] line in the log names the file, and the option is cleared. "
                    "Characters other than letters, digits, '-', '_' and '.' become '_' in the file name. The image is the game's back buffer at output resolution, "
                    "after tone mapping and with the game's own interface, without the Remix menu and before frame "
                    "generation. With HDR output active, the image is converted to SDR.");

    // CS thread, once per injected frame.
    static void onFrame(RtxContext& ctx);
    // CS thread, on a ray traced frame.
    static void noteExtents(const VkExtent3D& output, const VkExtent3D& render);
    // Called at present, when the screenshot (rtx_screenshot.h) reads the back buffer.
    // Returns the full path of the PNG that rtx.diagnostics.screenshotRequest asks for, and clears the request.
    // Returns an empty string when there is no new request.
    static std::string takeScreenshotRequest();
    // Writes the "[Session] clean shutdown" line once, which tells a bug report that the renderer did not crash.
    static void logShutdown(const char* reason);
  };
}
