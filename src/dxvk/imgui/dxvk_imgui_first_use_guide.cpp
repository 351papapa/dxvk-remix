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
#include "dxvk_imgui_first_use_guide.h"

#include "imgui.h"
#include "dxvk_imgui.h"
#include "../rtx_render/rtx_options.h"
#include "../rtx_render/rtx_utils.h"
#include "../../util/util_env.h"
#include "../../util/util_local_data.h"

namespace dxvk {

  static constexpr const char* kLicenseUrl = "https://github.com/NVIDIAGameWorks/dxvk-remix/blob/main/LICENSE-MIT";
  static constexpr const char* kNeverShowFirstUseGuideKey = "neverShowFirstUseGuide";

  bool ImGuiFirstUseGuide::shouldShow() {
    LocalData::app().load();
    return !LocalData::app().get<bool>(kNeverShowFirstUseGuideKey, false)
           && !RtxOptions::Automation::disableBlockingDialogBoxes();
  }

  bool ImGuiFirstUseGuide::show(ImFont* boldFont) {
    bool wantClose = false;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImVec2 windowSize(520.0f, 0.0f);
    ImGui::SetNextWindowPos(
      ImVec2(viewport->Size.x * 0.5f, viewport->Size.y * 0.5f),
      ImGuiCond_Always,
      ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);

    constexpr ImGuiWindowFlags kWindowFlags =
      ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_AlwaysAutoResize;

    const std::string& productName = ImGUI::productName();

    // The window ID after "###" stays fixed, so the title can follow the product name.
    const std::string windowTitle = str::format("Welcome to ", ImGUI::productNameWithVersion(), "###Welcome to RTX Remix");
    if (!ImGui::Begin(windowTitle.c_str(), nullptr, kWindowFlags)) {
      ImGui::End();
      return false;
    }

    ImGui::Spacing();

    {
      const auto keyBindDescriptor = buildKeyBindDescriptorStringForDisplay(RtxOptions::remixMenuKeyBinds());
      ImGui::TextWrapped("%s is now active. To open or close the Remix menus press:", productName.c_str());
      ImGui::Spacing();

      if (boldFont) {
        ImGui::PushFont(boldFont);
      }
      ImGui::Indent(20.0f);
      ImGui::Text("%s", keyBindDescriptor.c_str());
      ImGui::Unindent(20.0f);
      if (boldFont) {
        ImGui::PopFont();
      }
      ImGui::TextWrapped(
        "%s Menus can be navigated using mouse, keyboard, or gamepad. On keyboard, use the arrow keys to navigate and the spacebar to select.",
        productName.c_str());
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextWrapped("By using this software, you agree to its license:");
    ImGui::Spacing();

    if (ImGui::SmallButton("License")) {
      env::openUrlInBrowser(kLicenseUrl);
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("%s", kLicenseUrl);
    }

    if (!ImGUI::donateUrl().empty()) {
      ImGui::Spacing();
      ImGUI::showDonateButton();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    {
      const float buttonWidth = 180.0f;
      const float availableWidth = ImGui::GetContentRegionAvail().x;
      const float spacing = ImGui::GetStyle().ItemSpacing.x;
      const float totalButtonsWidth = buttonWidth * 2.0f + spacing;
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (availableWidth - totalButtonsWidth) * 0.5f);

      if (ImGui::Button("Close", ImVec2(buttonWidth, 0))) {
        wantClose = true;
      }

      ImGui::SameLine();

      if (ImGui::Button("Never Show This Again", ImVec2(buttonWidth, 0))) {
        LocalData::app().set(kNeverShowFirstUseGuideKey, true);
        LocalData::app().save();
        wantClose = true;
      }
    }

    ImGui::Spacing();
    ImGui::End();

    return wantClose;
  }

}
