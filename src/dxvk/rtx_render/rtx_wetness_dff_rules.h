// SPDX-License-Identifier: MIT
// Local runtime feature source. See CREDITS.md.
#pragma once
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dxvk {
  class WetnessDffRules {
  public:
    enum class Rule { Default, Enable, Disable };
    bool refresh(const std::filesystem::path& folder, const std::string& enable, const std::string& disable);
    Rule rule(uint64_t hash) const;
    void requestReload() { m_reload = true; }
    size_t mappedHashes() const { return m_names.size(); }
    size_t ambiguousHashes() const { return m_ambiguous; }
    size_t skippedFiles() const { return m_skipped; }
    size_t loadedFiles() const { return m_files; }
    bool loading() const { return m_pendingLoad.valid(); }
    const std::string* originalTextureName(uint64_t hash) const;
    size_t mappedTextures() const { return m_textureNames.size(); }
    size_t ambiguousTextures() const { return m_ambiguousTextures; }

  private:
    static std::string lower(std::string text);
    static std::vector<std::string> patterns(const std::string& text);
    void load(const std::filesystem::path& folder);
    std::unordered_map<uint64_t, std::unordered_set<std::string>> m_names;
    std::unordered_map<uint64_t, Rule> m_rules;
    std::unordered_map<uint64_t, std::unordered_set<std::string>> m_textureNames;
    size_t m_ambiguousTextures = 0;
    std::string m_enable, m_disable;
    size_t m_ambiguous = 0, m_skipped = 0, m_files = 0;
    bool m_loaded = false, m_reload = false;
    std::future<std::unique_ptr<WetnessDffRules>> m_pendingLoad;
  };
}
