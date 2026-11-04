#include "rtx_host_settings.h"

#include "rtx_imgui.h"
#include <remix/remix_c.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace dxvk::host_settings {

  namespace {
    struct Entry {
      std::string key;
      std::string label;
      std::string section;
      std::string tooltip;
      remixapi_HostSettingType type = REMIXAPI_HOST_SETTING_TYPE_BOOL;
      uint32_t flags = 0;
      double minValue = 0.0;
      double maxValue = 0.0;
      double step = 0.0;
      double defaultValue = 0.0;
      double value = 0.0;
      // Value in effect when the host started (registration time). A
      // requires-restart setting that differs from it is pending a restart.
      double startupValue = 0.0;
      std::vector<std::string> enumLabels;
    };

    std::mutex s_mutex;
    std::vector<Entry> s_entries;
    std::unordered_map<std::string, uint32_t> s_byKey;
    std::atomic<uint64_t> s_generation { 0 };
    std::atomic<bool> s_empty { true };

    std::string safeString(const char* s) {
      return s ? std::string(s) : std::string();
    }

    double sanitize(const Entry& e, double v) {
      if (!std::isfinite(v)) {
        v = e.defaultValue;
      }
      switch (e.type) {
      case REMIXAPI_HOST_SETTING_TYPE_BOOL:
        return v != 0.0 ? 1.0 : 0.0;
      case REMIXAPI_HOST_SETTING_TYPE_ENUM: {
        const double top = e.enumLabels.empty() ? 0.0 : double(e.enumLabels.size() - 1);
        return std::clamp(std::round(v), 0.0, top);
      }
      case REMIXAPI_HOST_SETTING_TYPE_INT:
        v = std::round(v);
        break;
      default:
        break;
      }
      if ((e.type == REMIXAPI_HOST_SETTING_TYPE_INT || e.type == REMIXAPI_HOST_SETTING_TYPE_FLOAT) &&
          e.minValue < e.maxValue) {
        v = std::clamp(v, e.minValue, e.maxValue);
      }
      return v;
    }

    // Menu-side write; bumps the generation so the host re-reads.
    void setFromMenu(Entry& e, double v) {
      v = sanitize(e, v);
      if (v == e.value && e.type != REMIXAPI_HOST_SETTING_TYPE_ACTION) {
        return;
      }
      e.value = v;
      s_generation.fetch_add(1, std::memory_order_release);
    }
  }

  uint32_t registerSetting(const remixapi_HostSettingInfo& info) {
    if (!info.key || !info.key[0] || info.type > REMIXAPI_HOST_SETTING_TYPE_ACTION) {
      return 0;
    }
    Entry e;
    e.key = info.key;
    e.label = info.label && info.label[0] ? info.label : info.key;
    e.section = info.section && info.section[0] ? info.section : "General";
    e.tooltip = safeString(info.tooltip);
    e.type = info.type;
    e.flags = info.flags;
    e.minValue = info.minValue;
    e.maxValue = info.maxValue;
    e.step = info.step;
    if (info.type == REMIXAPI_HOST_SETTING_TYPE_ENUM) {
      if (!info.enumLabels || info.enumCount == 0) {
        return 0;
      }
      for (uint32_t i = 0; i < info.enumCount; ++i) {
        e.enumLabels.emplace_back(info.enumLabels[i] ? info.enumLabels[i] : "?");
      }
    }
    e.defaultValue = info.defaultValue;
    e.defaultValue = sanitize(e, info.defaultValue);
    e.value = sanitize(e, info.currentValue);
    e.startupValue = e.value;

    std::lock_guard lock { s_mutex };
    auto it = s_byKey.find(e.key);
    if (it != s_byKey.end()) {
      s_entries[it->second - 1] = std::move(e);
      return it->second;
    }
    s_entries.push_back(std::move(e));
    const uint32_t handle = uint32_t(s_entries.size());
    s_byKey.emplace(s_entries.back().key, handle);
    s_empty.store(false, std::memory_order_release);
    return handle;
  }

  bool getValue(uint32_t handle, double& out) {
    std::lock_guard lock { s_mutex };
    if (handle == 0 || handle > s_entries.size()) {
      return false;
    }
    out = s_entries[handle - 1].value;
    return true;
  }

  bool getValueByKey(const char* key, double& out) {
    if (!key) {
      return false;
    }
    std::lock_guard lock { s_mutex };
    auto it = s_byKey.find(key);
    if (it == s_byKey.end()) {
      return false;
    }
    out = s_entries[it->second - 1].value;
    return true;
  }

  bool setValue(uint32_t handle, double value) {
    std::lock_guard lock { s_mutex };
    if (handle == 0 || handle > s_entries.size()) {
      return false;
    }
    Entry& e = s_entries[handle - 1];
    e.value = sanitize(e, value);
    return true;
  }

  void clear() {
    std::lock_guard lock { s_mutex };
    s_entries.clear();
    s_byKey.clear();
    s_empty.store(true, std::memory_order_release);
    s_generation.fetch_add(1, std::memory_order_release);
  }

  uint64_t generation() {
    return s_generation.load(std::memory_order_acquire);
  }

  bool empty() {
    return s_empty.load(std::memory_order_acquire);
  }

  namespace {
    // Row chrome matching RemixGui::RtxOptionUxWrapper: a right-side lane
    // with the reset-to-default dot, and a tinted background on rows that
    // differ from their default.
    struct RowWrapper {
      Entry& entry;
      ImGuiWindow* window = nullptr;
      const ImGuiStyle* style = nullptr;
      ImVec2 startCursorPos {};
      float lineHeight = 0.0f;
      float circleRadius = 0.0f;
      float reservedRightW = 0.0f;
      float prevWorkRectMaxX = 0.0f;
      bool isNonDefault = false;
      bool active = false;

      explicit RowWrapper(Entry& e) : entry(e) {
        window = ImGui::GetCurrentWindow();
        if (!window || window->SkipItems) {
          return;
        }
        active = true;
        ImGuiContext& g = *ImGui::GetCurrentContext();
        style = &g.Style;
        const float fontSize = g.FontSize;
        const ImVec2& padding = style->FramePadding;
        startCursorPos = window->DC.CursorPos;
        lineHeight = fontSize + padding.y * 2.0f;
        circleRadius = fontSize * 0.40f;
        reservedRightW = padding.x + fontSize + circleRadius + style->TouchExtraPadding.x + 1.0f;
        prevWorkRectMaxX = window->WorkRect.Max.x;
        window->WorkRect.Max.x = ImMax(window->WorkRect.Min.x, prevWorkRectMaxX - reservedRightW);
        window->DrawList->PushClipRect(window->ClipRect.Min, ImVec2(window->WorkRect.Max.x, window->ClipRect.Max.y), true);
        isNonDefault = entry.type != REMIXAPI_HOST_SETTING_TYPE_ACTION && entry.value != entry.defaultValue;
        window->DrawList->ChannelsSplit(2);
        window->DrawList->ChannelsSetCurrent(1);
        ImGui::BeginGroup();
      }

      ~RowWrapper() {
        if (!active) {
          return;
        }
        ImGui::EndGroup();
        const ImVec2 groupMin = ImGui::GetItemRectMin();
        window->DrawList->PopClipRect();
        window->WorkRect.Max.x = prevWorkRectMaxX;
        window->DrawList->ChannelsSetCurrent(0);
        const float y0 = startCursorPos.y;
        const float y1 = startCursorPos.y + lineHeight;
        const float x0 = ImMax(window->WorkRect.Min.x, groupMin.x - style->ItemInnerSpacing.x * 0.5f);
        const float x1 = prevWorkRectMaxX;
        if (ImGui::IsMouseHoveringRect(ImVec2(x0, y0), ImVec2(x1, y1))) {
          window->DrawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), ImGui::GetColorU32(ImGuiCol_HeaderHovered));
        } else if (isNonDefault) {
          window->DrawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), ImGui::GetColorU32(ImGuiCol_ChildBg));
        }
        window->DrawList->ChannelsMerge();

        if (entry.type == REMIXAPI_HOST_SETTING_TYPE_ACTION) {
          return;
        }
        ImGuiContext& g = *ImGui::GetCurrentContext();
        const ImVec2 circleCenter(prevWorkRectMaxX - style->FramePadding.x - g.FontSize, startCursorPos.y + lineHeight * 0.5f);
        ImRect hitBb(ImVec2(circleCenter.x - circleRadius, circleCenter.y - circleRadius),
                     ImVec2(circleCenter.x + circleRadius, circleCenter.y + circleRadius));
        hitBb.Expand(style->TouchExtraPadding);
        const ImGuiID id = window->GetID("##hostSettingReset");
        bool hovered = false;
        if (ImGui::ItemAdd(hitBb, id)) {
          bool held = false;
          if (ImGui::ButtonBehavior(hitBb, id, &hovered, &held, ImGuiButtonFlags_PressedOnClick)) {
            setFromMenu(entry, entry.defaultValue);
          }
          const ImU32 fill = isNonDefault ? ImGui::GetColorU32((ImU32) 0xFFffc734) : ImGui::GetColorU32((ImU32) 0xFF464646);
          const ImU32 outline = ImGui::GetColorU32(hovered ? ImGuiCol_Text : ImGuiCol_Border);
          window->DrawList->AddCircleFilled(circleCenter, circleRadius, fill);
          window->DrawList->AddCircle(circleCenter, circleRadius, outline, 0, 1.0f);
          if (hovered) {
            ImGui::SetTooltip("Reset to default (%s)", entry.key.c_str());
          }
        }
      }

      RowWrapper(const RowWrapper&) = delete;
      RowWrapper& operator=(const RowWrapper&) = delete;
    };

    std::string formatValue(const Entry& e, double v) {
      switch (e.type) {
      case REMIXAPI_HOST_SETTING_TYPE_BOOL:
        return v != 0.0 ? "on" : "off";
      case REMIXAPI_HOST_SETTING_TYPE_ENUM: {
        const size_t i = size_t(v);
        return i < e.enumLabels.size() ? e.enumLabels[i] : std::to_string(i);
      }
      case REMIXAPI_HOST_SETTING_TYPE_INT:
        return std::to_string(int64_t(v));
      default: {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.4g", v);
        return buf;
      }
      }
    }

    std::string buildTooltip(const Entry& e) {
      std::string t;
      if (!e.tooltip.empty()) {
        t = e.tooltip;
        t += "\n\n";
      }
      t += "Key: " + e.key;
      if (e.type != REMIXAPI_HOST_SETTING_TYPE_ACTION) {
        t += "\nDefault: " + formatValue(e, e.defaultValue);
      }
      if (e.flags & REMIXAPI_HOST_SETTING_FLAG_REQUIRES_RESTART) {
        t += "\nRequires a game restart to take effect.";
      }
      return t;
    }

    bool matchesFilter(const Entry& e, const char* filter) {
      if (!filter[0]) {
        return true;
      }
      auto contains = [filter](const std::string& s) {
        const size_t n = strlen(filter);
        if (n > s.size()) {
          return false;
        }
        for (size_t i = 0; i + n <= s.size(); ++i) {
          if (_strnicmp(s.c_str() + i, filter, n) == 0) {
            return true;
          }
        }
        return false;
      };
      return contains(e.label) || contains(e.key) || contains(e.section);
    }

    void drawEntry(Entry& e) {
      const bool readOnly = (e.flags & REMIXAPI_HOST_SETTING_FLAG_READ_ONLY) != 0;
      const bool needsRestart = (e.flags & REMIXAPI_HOST_SETTING_FLAG_REQUIRES_RESTART) != 0;
      const bool pendingRestart = needsRestart && e.value != e.startupValue;

      std::string label = e.label;
      if (needsRestart) {
        label += pendingRestart ? " (restart pending)" : " (restart)";
      }
      label += "##" + e.key;

      ImGui::PushID(e.key.c_str());
      {
        RowWrapper row(e);
        if (readOnly) {
          ImGui::BeginDisabled();
        }
        if (pendingRestart) {
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.78f, 0.2f, 1.0f));
        }
        switch (e.type) {
        case REMIXAPI_HOST_SETTING_TYPE_BOOL: {
          bool v = e.value != 0.0;
          if (RemixGui::Checkbox(label.c_str(), &v)) {
            setFromMenu(e, v ? 1.0 : 0.0);
          }
          break;
        }
        case REMIXAPI_HOST_SETTING_TYPE_INT: {
          int v = int(e.value);
          bool changed;
          if (e.minValue < e.maxValue) {
            const float speed = e.step > 0.0 ? float(e.step) : float(std::max(1.0, (e.maxValue - e.minValue) / 200.0));
            changed = RemixGui::DragInt(label.c_str(), &v, speed, int(e.minValue), int(e.maxValue), "%d", ImGuiSliderFlags_AlwaysClamp);
          } else {
            changed = RemixGui::DragInt(label.c_str(), &v, e.step > 0.0 ? float(e.step) : 1.0f);
          }
          if (changed) {
            setFromMenu(e, double(v));
          }
          break;
        }
        case REMIXAPI_HOST_SETTING_TYPE_FLOAT: {
          float v = float(e.value);
          bool changed;
          if (e.minValue < e.maxValue) {
            const float speed = e.step > 0.0 ? float(e.step) : float((e.maxValue - e.minValue) / 500.0);
            changed = RemixGui::DragFloat(label.c_str(), &v, speed, float(e.minValue), float(e.maxValue), "%.3f", ImGuiSliderFlags_AlwaysClamp);
          } else {
            changed = RemixGui::DragFloat(label.c_str(), &v, e.step > 0.0 ? float(e.step) : 0.01f);
          }
          if (changed) {
            setFromMenu(e, double(v));
          }
          break;
        }
        case REMIXAPI_HOST_SETTING_TYPE_ENUM: {
          int v = int(e.value);
          std::vector<const char*> items;
          items.reserve(e.enumLabels.size());
          for (const auto& s : e.enumLabels) {
            items.push_back(s.c_str());
          }
          if (RemixGui::Combo(label.c_str(), &v, items.data(), int(items.size()))) {
            setFromMenu(e, double(v));
          }
          break;
        }
        case REMIXAPI_HOST_SETTING_TYPE_ACTION: {
          if (ImGui::Button(label.c_str())) {
            setFromMenu(e, e.value + 1.0);
          }
          break;
        }
        }
        if (pendingRestart) {
          ImGui::PopStyleColor();
        }
        if (readOnly) {
          ImGui::EndDisabled();
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          const std::string tip = buildTooltip(e);
          RemixGui::SetTooltipUnformatted(tip.c_str());
        }
      }
      ImGui::PopID();
    }
  }

  void showImguiTab() {
    static char s_filter[64] = {};

    std::lock_guard lock { s_mutex };
    if (s_entries.empty()) {
      ImGui::TextWrapped("No host settings registered.");
      return;
    }

    ImGui::TextWrapped("Settings exposed by the game integration. Changes apply immediately where the host "
                       "supports it; the host also saves them to its own config file.");
    ImGui::InputTextWithHint("##hostSettingsFilter", "Filter...", s_filter, sizeof(s_filter));
    ImGui::SameLine();
    if (ImGui::Button("Reset All")) {
      for (auto& e : s_entries) {
        if (e.type != REMIXAPI_HOST_SETTING_TYPE_ACTION && !(e.flags & REMIXAPI_HOST_SETTING_FLAG_READ_ONLY)) {
          setFromMenu(e, e.defaultValue);
        }
      }
    }
    RemixGui::SetTooltipToLastWidgetOnHover("Reset every host setting to its default.");
    RemixGui::Separator();

    // Sections in first-registration order.
    std::vector<const std::string*> sections;
    for (const auto& e : s_entries) {
      bool seen = false;
      for (auto* s : sections) {
        if (*s == e.section) {
          seen = true;
          break;
        }
      }
      if (!seen) {
        sections.push_back(&e.section);
      }
    }

    for (const std::string* section : sections) {
      bool any = false;
      for (const auto& e : s_entries) {
        if (e.section == *section && matchesFilter(e, s_filter)) {
          any = true;
          break;
        }
      }
      if (!any) {
        continue;
      }
      const std::string header = *section + "##hostSection";
      ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen;
      if (s_filter[0]) {
        ImGui::SetNextItemOpen(true);
      }
      if (RemixGui::CollapsingHeader(header.c_str(), flags)) {
        ImGui::Indent();
        for (auto& e : s_entries) {
          if (e.section == *section && matchesFilter(e, s_filter)) {
            drawEntry(e);
          }
        }
        ImGui::Unindent();
      }
    }
  }

  bool hasUserMenuSettings() {
    std::lock_guard lock { s_mutex };
    for (const auto& e : s_entries) {
      if (e.flags & REMIXAPI_HOST_SETTING_FLAG_USER_MENU) {
        return true;
      }
    }
    return false;
  }

  void showUserMenuSettings() {
    std::lock_guard lock { s_mutex };
    for (auto& e : s_entries) {
      if (e.flags & REMIXAPI_HOST_SETTING_FLAG_USER_MENU) {
        drawEntry(e);
      }
    }
  }

}
