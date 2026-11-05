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
#include "rtx_frontend_ui.h"

#include <algorithm>
#include <mutex>
#include <vector>

namespace dxvk {
  namespace frontend_ui {

    namespace {
      std::mutex s_mutex;
      std::vector<DrawFn> s_panels[unsigned(Group::Count)];
    }

    void registerPanel(Group group, DrawFn fn) {
      if (fn == nullptr || unsigned(group) >= unsigned(Group::Count)) {
        return;
      }
      std::lock_guard<std::mutex> lock(s_mutex);
      std::vector<DrawFn>& panels = s_panels[unsigned(group)];
      if (std::find(panels.begin(), panels.end(), fn) == panels.end()) {
        panels.push_back(fn);
      }
    }

    void drawPanels(Group group) {
      if (unsigned(group) >= unsigned(Group::Count)) {
        return;
      }
      std::vector<DrawFn> panels;
      {
        std::lock_guard<std::mutex> lock(s_mutex);
        panels = s_panels[unsigned(group)];
      }
      for (DrawFn fn : panels) {
        fn();
      }
    }

  } // namespace frontend_ui
} // namespace dxvk
