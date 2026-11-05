#include <algorithm>
#include <cctype>
#include <cfloat>
#include <climits>
#include <mutex>
#include <optional>

#include "imgui.h"

#include "dxvk_imgui_option_search.h"
#include "../rtx_render/rtx_imgui.h"
#include "../rtx_render/rtx_option.h"

namespace dxvk {
  namespace {
    constexpr size_t kMaxResults = 200;

    std::string toLower(std::string text) {
      std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      return text;
    }

    struct IntRange {
      int minValue = 0;
      int maxValue = 0;
      ImGuiSliderFlags flags = ImGuiSliderFlags_None;
    };

    struct FloatRange {
      float minValue = 0.0f;
      float maxValue = 0.0f;
      ImGuiSliderFlags flags = ImGuiSliderFlags_None;
    };

    // Integer bounds sit in GenericValue::value, sign-extended for signed types and zero-extended for unsigned ones.
    int toIntBound(const GenericValue& bound) {
      return static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(bound.value), INT_MIN, INT_MAX));
    }

    // Returns the option's bounds in the form DragInt takes.
    // Returns an empty range, which DragInt treats as unbounded, when the option has no bounds.
    IntRange getIntRange(const RtxOptionImpl& option) {
      std::lock_guard<std::mutex> lock(RtxOptionImpl::getUpdateMutex());
      IntRange range;
      if (option.minValue.has_value() || option.maxValue.has_value()) {
        range.minValue = option.minValue.has_value() ? toIntBound(*option.minValue) : INT_MIN;
        range.maxValue = option.maxValue.has_value() ? toIntBound(*option.maxValue) : INT_MAX;
        range.flags = ImGuiSliderFlags_AlwaysClamp;
      }
      return range;
    }

    // Returns the option's bounds in the form DragFloat takes.
    // Returns an empty range, which DragFloat treats as unbounded, when the option has no bounds.
    FloatRange getFloatRange(const RtxOptionImpl& option) {
      std::lock_guard<std::mutex> lock(RtxOptionImpl::getUpdateMutex());
      FloatRange range;
      if (option.minValue.has_value() || option.maxValue.has_value()) {
        range.minValue = option.minValue.has_value() ? option.minValue->f : -FLT_MAX;
        range.maxValue = option.maxValue.has_value() ? option.maxValue->f : FLT_MAX;
        range.flags = ImGuiSliderFlags_AlwaysClamp;
      }
      return range;
    }

    void showEntryCount(const char* label, size_t count, RtxOptionImpl* option) {
      ImGui::Text("%s: %zu %s", label, count, count == 1 ? "entry" : "entries");
      if (ImGui::IsItemHovered()) {
        const std::string tooltip = RemixGui::BuildRtxOptionTooltip(option);
        RemixGui::SetTooltipUnformatted(tooltip.c_str());
      }
    }
  }

  bool ImGuiOptionSearch::show() {
    const ImGuiStyle& style = ImGui::GetStyle();
    const float clearButtonWidth = ImGui::CalcTextSize("Clear").x + style.FramePadding.x * 2.0f;
    ImGui::SetNextItemWidth(std::max(1.0f, ImGui::GetContentRegionAvail().x - clearButtonWidth - style.ItemSpacing.x));
    bool queryChanged = ImGui::InputTextWithHint("##Search Settings", "Search settings", m_query.data(), m_query.size());

    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
      m_query[0] = '\0';
      queryChanged = true;
    }

    // Options can register after the first search, so a registry size change also refreshes the results.
    if (queryChanged || (m_hasSearch && m_registrySize != RtxOptionImpl::getGlobalOptionMap().size())) {
      filter();
    }

    if (!m_hasSearch) {
      return false;
    }

    if (m_matchCount == 1) {
      ImGui::TextUnformatted("1 setting");
    } else {
      ImGui::Text("%zu settings", m_matchCount);
    }

    if (m_matchCount > m_results.size()) {
      ImGui::SameLine();
      ImGui::TextDisabled("(showing the first %zu, refine your search)", m_results.size());
    }

    RemixGui::Separator();

    for (const uint32_t index : m_results) {
      showEditor(m_entries[index]);
    }

    return true;
  }

  void ImGuiOptionSearch::buildEntries() {
    const auto& optionMap = RtxOptionImpl::getGlobalOptionMap();
    m_entries.clear();
    m_entries.reserve(optionMap.size());

    for (const auto& [hash, option] : optionMap) {
      Entry entry;
      entry.option = option;
      entry.fullName = option->getFullName();

      const char* description = option->getDescription();
      entry.searchText = toLower(entry.fullName + "\n" + (description ? description : ""));
      m_entries.push_back(std::move(entry));
    }

    std::sort(m_entries.begin(), m_entries.end(), [](const Entry& a, const Entry& b) { return a.fullName < b.fullName; });
    m_registrySize = optionMap.size();
  }

  void ImGuiOptionSearch::filter() {
    const std::string query = toLower(m_query.data());

    std::vector<std::string> words;
    size_t start = 0;
    while (start < query.size()) {
      const size_t end = std::min(query.find(' ', start), query.size());
      if (end > start) {
        words.push_back(query.substr(start, end - start));
      }
      start = end + 1;
    }

    m_results.clear();
    m_matchCount = 0;
    m_hasSearch = !words.empty();
    if (!m_hasSearch) {
      return;
    }

    if (m_registrySize != RtxOptionImpl::getGlobalOptionMap().size()) {
      buildEntries();
    }

    for (uint32_t index = 0; index < m_entries.size(); index++) {
      const std::string& text = m_entries[index].searchText;
      const bool matches = std::all_of(words.begin(), words.end(), [&text](const std::string& word) { return text.find(word) != std::string::npos; });
      if (!matches) {
        continue;
      }

      m_matchCount++;
      if (m_results.size() < kMaxResults) {
        m_results.push_back(index);
      }
    }
  }

  void ImGuiOptionSearch::showEditor(const Entry& entry) {
    RtxOptionImpl* option = entry.option;
    const char* label = entry.fullName.c_str();

    // Each OptionType maps to exactly one RtxOption<T>, except Int.
    // Int options of every integer and enum type keep their layer values in GenericValue::i,
    // which is what rtx.conf and user.conf read and write, so they are edited through RtxOption<int>.
    switch (option->getType()) {
    case OptionType::Bool:
      RemixGui::Checkbox(label, static_cast<RtxOption<bool>*>(option));
      break;
    case OptionType::Int: {
      const IntRange range = getIntRange(*option);
      RemixGui::DragInt(label, static_cast<RtxOption<int>*>(option), 0.2f, range.minValue, range.maxValue, "%d", range.flags);
      break;
    }
    case OptionType::Float: {
      const FloatRange range = getFloatRange(*option);
      RemixGui::DragFloat(label, static_cast<RtxOption<float>*>(option), 0.01f, range.minValue, range.maxValue, "%g", range.flags);
      break;
    }
    case OptionType::Vector2:
      RemixGui::DragFloat2(label, static_cast<RtxOption<Vector2>*>(option), 0.01f, 0.0f, 0.0f, "%g");
      break;
    case OptionType::Vector3:
      RemixGui::DragFloat3(label, static_cast<RtxOption<Vector3>*>(option), 0.01f, 0.0f, 0.0f, "%g");
      break;
    case OptionType::Vector4:
      RemixGui::DragFloat4(label, static_cast<RtxOption<Vector4>*>(option), 0.01f, 0.0f, 0.0f, "%g");
      break;
    case OptionType::Vector2i:
      RemixGui::DragInt2(label, static_cast<RtxOption<Vector2i>*>(option), 0.2f);
      break;
    case OptionType::String:
      RemixGui::InputText(label, static_cast<RtxOption<std::string>*>(option), ImGuiInputTextFlags_EnterReturnsTrue);
      break;
    case OptionType::VirtualKeys:
      RemixGui::InputKeyBind(label, static_cast<RtxOption<VirtualKeys>*>(option));
      break;
    case OptionType::HashSet:
      showEntryCount(label, option->getResolvedValue().hashSet->size(), option);
      break;
    case OptionType::HashVector:
      showEntryCount(label, option->getResolvedValue().hashVector->size(), option);
      break;
    }
  }
}
