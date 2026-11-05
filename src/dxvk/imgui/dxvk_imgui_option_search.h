#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace dxvk {
  class RtxOptionImpl;

  // Finds RtxOptions by full name or description and draws an editor for each match.
  class ImGuiOptionSearch {
  public:
    // Draws the search field, and the matching options while the field holds a search.
    // Returns true while the field holds a search.
    bool show();

  private:
    struct Entry {
      RtxOptionImpl* option = nullptr;
      std::string fullName;
      // Holds the lowercase full name and description, separated by a newline.
      std::string searchText;
    };

    void buildEntries();
    void filter();
    static void showEditor(const Entry& entry);

    std::array<char, 256> m_query {};
    std::vector<Entry> m_entries;
    std::vector<uint32_t> m_results;
    size_t m_matchCount = 0;
    size_t m_registrySize = 0;
    bool m_hasSearch = false;
  };
}
