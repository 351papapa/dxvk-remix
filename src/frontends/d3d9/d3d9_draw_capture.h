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
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>

namespace dxvk::frontends::d3d9 {
  // Append-only correlation logs consumed by external asset capture tools.
  // Each frontend owns its writer and calls it only on the render thread.
  class DrawCaptureLog {
  public:
    bool append(const std::filesystem::path& directory, uint32_t session,
                const std::string& hashes, const std::string& state) {
      if (!session) {
        return true;
      }
      if (session != m_session) {
        m_session = session;
        m_hashes.clear();
        m_states.clear();
      }
      std::error_code error;
      std::filesystem::create_directories(directory, error);
      if (error) {
        return false;
      }
      const bool hashesWritten = appendUnique(directory / "dff_remix_hashes.csv", hashes, m_hashes);
      const bool stateWritten = appendUnique(directory / "dff_remix_states.csv", state, m_states);
      return hashesWritten && stateWritten;
    }

  private:
    static bool appendUnique(const std::filesystem::path& path, const std::string& record,
                             std::unordered_set<std::string>& seen) {
      if (seen.find(record) != seen.end()) {
        return true;
      }
      std::ofstream stream(path, std::ios::app);
      stream << record << std::endl;
      if (!stream) {
        return false;
      }
      seen.insert(record);
      return true;
    }

    uint32_t m_session = 0;
    std::unordered_set<std::string> m_hashes;
    std::unordered_set<std::string> m_states;
  };
}
