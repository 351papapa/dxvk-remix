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
#include "../../../src/frontends/d3d9/d3d9_draw_capture.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

static std::string read(const std::filesystem::path& path) {
  std::ifstream file(path);
  return std::string(std::istreambuf_iterator<char>(file), {});
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "Pass an empty test directory");
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root = std::filesystem::path(argv[1]) / std::to_string(stamp);
    require(!std::filesystem::exists(root), "Test directory already exists");
    dxvk::frontends::d3d9::DrawCaptureLog log;
    require(log.append(root, 0, "ignored", "ignored"), "Disabled capture");
    require(!std::filesystem::exists(root), "Disabled capture created files");
    require(log.append(root, 1, "1,7,aa,bb,cc", "1,7,1"), "First draw");
    require(log.append(root, 1, "1,7,aa,bb,cc", "1,7,1"), "Repeated draw");
    require(read(root / "dff_remix_hashes.csv") == "1,7,aa,bb,cc\n", "Duplicate hash record");
    require(log.append(root, 1, "1,7,aa,bb,cc", "1,7,2"), "Changed state");
    require(read(root / "dff_remix_states.csv") == "1,7,1\n1,7,2\n", "Changed state lost");
    require(log.append(root, 2, "2,7,aa,bb,cc", "2,7,1"), "Session rollover");
    require(read(root / "dff_remix_hashes.csv") == "1,7,aa,bb,cc\n2,7,aa,bb,cc\n", "Session token reuse");
    const auto blocked = root / "blocked";
    std::ofstream(blocked) << "not a directory";
    require(!log.append(blocked, 3, "3,8,aa,bb,cc", "3,8,1"), "Invalid output accepted");
    require(log.append(root, 3, "3,8,aa,bb,cc", "3,8,1"), "Failed write cannot retry");
    require(read(root / "dff_remix_hashes.csv").find("3,8,aa,bb,cc\n") != std::string::npos, "Retry lost record");
    std::cout << "Draw capture tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
