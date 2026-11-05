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
// The pure checks of the scene interface boundary (rtx_scene_input.h): finite and invertible matrices, the triangle
// fixes, the bone fixes and the [SceneInput] line.

#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "../../../src/dxvk/rtx_render/rtx_scene_input.h"

namespace dxvk {
  // Logger needed by shared code used in this unit test.
  Logger Logger::s_instance("test_scene_input.log");
}

namespace {
  using namespace dxvk;

  int g_failures = 0;

  void expect(bool condition, const char* what) {
    if (!condition) {
      std::printf("FAILED: %s\n", what);
      ++g_failures;
    }
  }

  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  struct V {
    float x, y, z, pad;
  };

  void testMatrices() {
    Matrix4 m;
    expect(scene_input::finite(m) && scene_input::invertible(m), "identity is finite and invertible");
    m[2][1] = kNaN;
    expect(!scene_input::finite(m) && !scene_input::invertible(m), "a NaN element is not finite");
    m = Matrix4();
    m[3][0] = kInf;
    expect(!scene_input::finite(m), "an infinite translation is not finite");
    m = Matrix4();
    m[1] = Vector4(0.f, 0.f, 0.f, 0.f);
    expect(scene_input::finite(m) && !scene_input::invertible(m), "a zero row does not invert");
  }

  void testIndexedList() {
    // Two triangles sharing an edge. Vertex 3 is NaN, so the second triangle collapses.
    std::vector<V> v = { { 0, 0, 0, 0 }, { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { kNaN, 1, 0, 0 } };
    std::vector<uint16_t> idx = { 0, 1, 2, 1, 3, 2, 0, 9, 1 };
    const scene_input::TriangleFix fix =
      scene_input::fixTriangles(reinterpret_cast<uint8_t*>(v.data()), sizeof(V), 4, idx.data(), uint32_t(idx.size()), true);
    expect(fix.positions == 1 && fix.triangles == 1 && fix.indices == 1, "indexed list: one position, one triangle, one index");
    expect(v[3].x == 0.f && v[3].y == 0.f && v[3].z == 0.f, "a NaN position becomes 0");
    expect(idx[0] == 0 && idx[1] == 1 && idx[2] == 2, "a good triangle is kept");
    expect(idx[3] == 1 && idx[4] == 1 && idx[5] == 1, "a triangle with a fixed vertex collapses onto its first vertex");
    expect(idx[6] == 0 && idx[7] == 0 && idx[8] == 0, "a triangle with an index past the vertex count collapses onto 0");
  }

  void testStrip() {
    std::vector<V> v = { { 0, 0, 0, 0 }, { 1, 0, 0, 0 }, { 0, 1, 0, 0 } };
    std::vector<uint32_t> idx = { 0, 1, 2, 7 };
    const scene_input::TriangleFix fix =
      scene_input::fixTriangles(reinterpret_cast<uint8_t*>(v.data()), sizeof(V), 3, idx.data(), uint32_t(idx.size()), false);
    expect(fix.indices == 1 && idx[3] == 0 && idx[2] == 2, "strip: an index past the vertex count becomes 0");
  }

  void testNonIndexed() {
    std::vector<V> v = { { 0, 0, 0, 0 }, { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 2, 2, 2, 0 }, { 3, kInf, 3, 0 }, { 4, 4, 4, 0 } };
    const scene_input::TriangleFix fix =
      scene_input::fixTriangles<uint32_t>(reinterpret_cast<uint8_t*>(v.data()), sizeof(V), 6, nullptr, 0, true);
    expect(fix.positions == 1 && fix.triangles == 1, "non-indexed: one position, one triangle");
    expect(v[1].x == 1.f, "non-indexed: a good triangle is kept");
    expect(v[4].x == 2.f && v[5].x == 2.f && v[5].y == 2.f, "non-indexed: the triangle collapses onto its first vertex");
  }

  void testClean() {
    std::vector<V> v = { { 0, 0, 0, 0 }, { 1, 0, 0, 0 }, { 0, 1, 0, 0 } };
    std::vector<uint32_t> idx = { 0, 1, 2 };
    const scene_input::TriangleFix fix =
      scene_input::fixTriangles(reinterpret_cast<uint8_t*>(v.data()), sizeof(V), 3, idx.data(), 3, true);
    expect(fix.positions == 0 && fix.triangles == 0 && fix.indices == 0, "a clean mesh is not changed");
  }

  void testBones() {
    SceneInputRecord record;
    std::vector<Matrix4> bones(2);
    scene_input::fixBones(bones, 2, record);
    expect(record.fixed == 0 && bones.size() == 2, "complete bones are not changed");

    bones[1][0][0] = kNaN;
    scene_input::fixBones(bones, 5, record);
    expect(bones.size() == 5 && scene_input::finite(bones[1]) && bones[1][0][0] == 1.f && bones[4][3][3] == 1.f,
           "a NaN bone and missing bones become identity");
    expect(record.fixed == ((1u << uint32_t(SceneInput::Bone)) | (1u << uint32_t(SceneInput::BonesMissing))),
           "the bone and missing-bone fixes are recorded");

    SceneInputRecord limit;
    std::vector<Matrix4> many(kSceneInputMaxBones + 3);
    scene_input::fixBones(many, 1000, limit);
    expect(many.size() == kSceneInputMaxBones && limit.fixed == (1u << uint32_t(SceneInput::BonesLimit)),
           "bones past the limit are dropped, and used bones are capped at the limit");
  }

  void testStats() {
    SceneInputStats stats;
    expect(stats.take(100).empty(), "the first call only starts the interval");
    SceneInputRecord record;
    record.rejected = SceneInput::Transform;
    stats.add(record);
    stats.add(SceneInput::Position, 2);
    SceneInputRecord fixed;
    fixed.fix(SceneInput::Bone);
    fixed.fix(SceneInput::BonesMissing);
    stats.add(fixed);
    expect(stats.take(399).empty(), "nothing is logged before the interval ends");
    const std::string line = stats.take(400);
    expect(line == "[SceneInput] rejected=1 fixed=4 (frames 100-400) transform=1 position=2 bone=1 bonesMissing=1",
           "the line has the totals and the reasons");
    if (line.rfind("[SceneInput]", 0) != 0) {
      std::printf("line: %s\n", line.c_str());
    }
    expect(stats.take(700).empty(), "an interval without counts logs nothing");
    stats.add(SceneInput::LiveMesh, 3);
    expect(stats.take(1000) == "[SceneInput] rejected=0 fixed=0 repeated=3 (frames 700-1000) liveMesh=3",
           "repeated creates are counted apart from rejections");
    stats.add(SceneInput::VertexRange, 2);
    stats.add(SceneInput::Transform);
    expect(stats.take(1300) == "[SceneInput] rejected=1 fixed=0 rasterFallback=2 (frames 1000-1300) transform=1 vertexRange=2",
           "raster fallbacks are counted apart from rejections");
  }
}

int main() {
  testMatrices();
  testIndexedList();
  testStrip();
  testNonIndexed();
  testClean();
  testBones();
  testStats();
  if (g_failures != 0) {
    std::printf("%d checks failed\n", g_failures);
    return -1;
  }
  std::printf("All passed\n");
  return 0;
}
