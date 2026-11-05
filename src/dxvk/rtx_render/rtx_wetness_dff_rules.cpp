// SPDX-License-Identifier: MIT
// Local runtime feature source. See CREDITS.md.
#include "rtx_wetness_dff_rules.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iterator>
#include "../../../submodules/FidelityFX-SDK/framework/cauldron/framework/libs/json/json.h"

namespace dxvk {
  std::string WetnessDffRules::lower(std::string text) {
    for (char& c : text) {
      if (c >= 'A' && c <= 'Z') {
        c = char(c + ('a' - 'A'));
      }
    }
    return text;
  }

  std::vector<std::string> WetnessDffRules::patterns(const std::string& text) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin < text.size()) {
      const size_t end = text.find_first_of(";\r\n", begin);
      auto token = lower(text.substr(begin, end == std::string::npos ? end : end - begin));
      const auto first = token.find_first_not_of(" \t");
      if (first != std::string::npos) {
        result.push_back(token.substr(first, token.find_last_not_of(" \t") - first + 1));
      }
      if (end == std::string::npos) {
        break;
      }
      begin = end + 1;
    }
    return result;
  }

  void WetnessDffRules::load(const std::filesystem::path& folder) {
    m_names.clear();
    m_textureNames.clear();
    m_ambiguousTextures = 0;
    m_files = m_skipped = m_ambiguous = 0;
    std::error_code error;
    auto it = std::filesystem::recursive_directory_iterator(folder,
      std::filesystem::directory_options::skip_permission_denied, error);
    const auto end = std::filesystem::recursive_directory_iterator();
    for (; !error && it != end; it.increment(error)) {
      if (it->path().extension() != ".json" || !it->is_regular_file(error)) {
        continue;
      }
      if (it->file_size(error) > 64 * 1024 * 1024 || error) {
        ++m_skipped;
        error.clear();
        continue;
      }
      try {
        std::ifstream file(it->path(), std::ios::binary);
        const auto doc = nlohmann::json::parse(file, nullptr, false);
        if (!doc.is_object() || doc.value("schema", 0) != 1 || !doc.contains("dff") ||
            !doc["dff"].is_string() || !doc.contains("meshes") || !doc["meshes"].is_array()) {
          ++m_skipped;
          continue;
        }
        const std::string name = lower(doc["dff"].get<std::string>());
        if (name.empty()) {
          ++m_skipped;
          continue;
        }
        std::unordered_set<uint64_t> hashes;
        auto collectDraws = [&](const nlohmann::json& draws, const std::string& textureName) {
          if (!draws.is_array()) { return; }
          for (const auto& draw : draws) {
            if (!draw.is_object() || !draw.contains("hashes") || !draw["hashes"].is_array()) { continue; }
            for (const auto& record : draw["hashes"]) {
              if (!textureName.empty() && record.is_object() && record.contains("texture0_hash") && record["texture0_hash"].is_string()) {
                auto text = record["texture0_hash"].get<std::string>();
                if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) { text.erase(0, 2); }
                uint64_t hash = 0;
                const auto parsed = std::from_chars(text.data(), text.data() + text.size(), hash, 16);
                if (parsed.ec == std::errc() && parsed.ptr == text.data() + text.size() && hash != 0) {
                  m_textureNames[hash].insert(lower(textureName));
                }
              }
              if (!record.is_object() || !record.contains("remix_mesh_hash") || !record["remix_mesh_hash"].is_string()) { continue; }
              auto text = record["remix_mesh_hash"].get<std::string>();
              if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) { text.erase(0, 2); }
              uint64_t hash = 0;
              const auto parsed = std::from_chars(text.data(), text.data() + text.size(), hash, 16);
              if (parsed.ec == std::errc() && parsed.ptr == text.data() + text.size() && hash != 0) { hashes.insert(hash); }
            }
          }
        };
        for (const auto& mesh : doc["meshes"]) {
          if (!mesh.is_object()) { continue; }
          if (mesh.contains("materials") && mesh["materials"].is_array()) {
            for (const auto& material : mesh["materials"]) {
              if (material.is_object() && material.contains("draw_calls")) {
                const auto name = material.contains("texture") && material["texture"].is_string() ? material["texture"].get<std::string>() : std::string();
                collectDraws(material["draw_calls"], name);
              }
            }
          }
          if (mesh.contains("unmatched_draw_calls")) { collectDraws(mesh["unmatched_draw_calls"], {}); }
        }
        for (const auto hash : hashes) { m_names[hash].insert(name); }
        ++m_files;
      } catch (const std::exception&) {
        ++m_skipped;
      }
    }
    for (const auto& entry : m_names) { m_ambiguous += entry.second.size() > 1; }
    for (const auto& entry : m_textureNames) { m_ambiguousTextures += entry.second.size() > 1; }
    m_loaded = true;
    m_reload = false;
  }

  bool WetnessDffRules::refresh(const std::filesystem::path& folder, const std::string& enable, const std::string& disable) {
    bool reloaded = false;
    if (m_pendingLoad.valid() && m_pendingLoad.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      auto loaded = m_pendingLoad.get();
      m_names = std::move(loaded->m_names);
      m_textureNames = std::move(loaded->m_textureNames);
      m_ambiguousTextures = loaded->m_ambiguousTextures;
      m_files = loaded->m_files;
      m_skipped = loaded->m_skipped;
      m_ambiguous = loaded->m_ambiguous;
      m_loaded = true;
      reloaded = true;
    }
    if (!m_pendingLoad.valid() && (m_reload || !m_loaded)) {
      m_reload = false;
      // Capture files may be on a slow disk. Keep the current rules until a full snapshot is ready.
      m_pendingLoad = std::async(std::launch::async, [folder] {
        auto result = std::make_unique<WetnessDffRules>();
        result->load(folder);
        return result;
      });
    }
    if (!reloaded && enable == m_enable && disable == m_disable) { return false; }
    m_enable = enable;
    m_disable = disable;
    m_rules.clear();
    const auto enabled = patterns(enable), disabled = patterns(disable);
    for (const auto& entry : m_names) {
      // A shared mesh cannot identify which DFF is drawing it. Never guess.
      if (entry.second.size() != 1) { continue; }
      const auto& name = *entry.second.begin();
      auto matches = [&](const auto& list) {
        return std::any_of(list.begin(), list.end(), [&](const auto& p) { return name.find(p) != std::string::npos; });
      };
      if (matches(disabled)) { m_rules[entry.first] = Rule::Disable; }
      else if (matches(enabled)) { m_rules[entry.first] = Rule::Enable; }
    }
    return true;
  }

  WetnessDffRules::Rule WetnessDffRules::rule(uint64_t hash) const {
    const auto it = m_rules.find(hash);
    return it == m_rules.end() ? Rule::Default : it->second;
  }

  const std::string* WetnessDffRules::originalTextureName(uint64_t hash) const {
    const auto it = m_textureNames.find(hash);
    return it != m_textureNames.end() && it->second.size() == 1 ? &*it->second.begin() : nullptr;
  }
}
