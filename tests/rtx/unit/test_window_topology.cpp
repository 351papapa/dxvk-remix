// SPDX-License-Identifier: MIT
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "../../../src/dxvk/rtx_render/rtx_window_topology.h"

namespace {
  using namespace dxvk;
  struct Vertex { float p[3]; float uv[2]; };
  unsigned checks = 0;
  void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }
  std::vector<Vertex> quad() {
    return {{{0,0,0},{0,0}}, {{1,0,0},{1,0}}, {{0,1,0},{0,1}},
            {{0,1,0},{0,1}}, {{1,0,0},{1,0}}, {{1,1,0},{1,1}}};
  }
  WindowTopologyInput inputFor(const std::vector<Vertex>& vertices) {
    WindowTopologyInput input;
    const auto* bytes = reinterpret_cast<const uint8_t*>(vertices.data());
    input.positions = {bytes, vertices.size() * sizeof(Vertex)};
    input.texcoords = {bytes + offsetof(Vertex, uv), vertices.size() * sizeof(Vertex) - offsetof(Vertex, uv)};
    input.vertexCount = uint32_t(vertices.size());
    input.triangleCount = uint32_t(vertices.size() / 3);
    input.positionStride = input.texcoordStride = sizeof(Vertex);
    return input;
  }
  void expect(const WindowTopologyInput& input, const std::vector<uint32_t>& roots, const char* message) {
    require(buildWindowIslandRoots(input) == roots, message);
  }
  void testMatching() {
    auto vertices = quad();
    WindowTopologyDiagnostics stats;
    require(buildWindowIslandRoots(inputFor(vertices), &stats) == std::vector<uint32_t>({0,0}), "Duplicated non-indexed quad split");
    require(stats.status == WindowTopologyStatus::Complete && stats.trianglesValid == 2 && stats.matchedEdges == 1 && stats.mergedTriangles == 1,
      "Successful quad diagnostics incorrect");
    const double baseTolerance = stats.positionTolerance;
    for (auto& vertex : vertices) {
      vertex.p[0] += 1000000;
      vertex.p[1] -= 1000000;
    }
    require(buildWindowIslandRoots(inputFor(vertices), &stats) == std::vector<uint32_t>({0,0}) && stats.positionTolerance == baseTolerance,
      "Absolute object offset changed tolerance or connectivity");
    vertices = quad();
    vertices[3].p[0] += 2e-6f;
    vertices[4].p[1] += 3e-6f;
    vertices[3].uv[0] += 2e-6f;
    expect(inputFor(vertices), {0,0}, "Near duplicated endpoints split");
    vertices = quad();
    for (unsigned axis = 0; axis < 3; ++axis) {
      for (int direction : {-1, 1}) {
        auto perturbed = vertices;
        perturbed[3].p[axis] += float(direction) * 4e-6f;
        perturbed[4].p[axis] += float(direction) * 4e-6f;
        expect(inputFor(perturbed), {0,0}, "Neighboring midpoint bucket missed a shared edge");
      }
    }
    for (int direction : {-1, 1}) {
      auto crossing = quad();
      if (direction > 0) {
        crossing[0].p[2] = 4e-6f;
      }
      crossing[3].p[2] = crossing[4].p[2] = float(direction) * 4e-6f;
      require(buildWindowIslandRoots(inputFor(crossing), &stats) == std::vector<uint32_t>({0,0}), "Signed bucket boundary split a quad");
      const double origin = crossing[0].p[2];
      const double firstCell = std::floor(-origin / (2 * stats.positionTolerance));
      const double secondCell = std::floor((crossing[3].p[2] - origin) / (2 * stats.positionTolerance));
      require(firstCell != secondCell, "Boundary fixture did not span two buckets");
    }
    vertices[3].p[0] = -0.0f;
    vertices[4].p[2] = -0.0f;
    vertices[3].uv[0] = -0.0f;
    expect(inputFor(vertices), {0,0}, "Signed zero split endpoints");
    vertices = quad();
    vertices[3].uv[0] += .001f;
    expect(inputFor(vertices), {0,1}, "UV seam joined");
    vertices = quad();
    vertices[3].p[0] += .001f;
    expect(inputFor(vertices), {0,1}, "Distant endpoints joined");
    vertices = {{{0,0,0},{0,0}}, {{1,0,0},{1,0}}, {{0,1,0},{0,1}},
                {{0,0,0},{0,0}}, {{-1,0,0},{1,0}}, {{0,-1,0},{0,1}}};
    expect(inputFor(vertices), {0,1}, "Single shared corner joined");
    vertices = quad();
    auto second = quad();
    for (auto& vertex : second) {
      vertex.p[0] += 3;
    }
    vertices.insert(vertices.end(), second.begin(), second.end());
    expect(inputFor(vertices), {0,0,2,2}, "Separate quads joined or canonical roots changed");
    vertices = quad();
    std::swap(vertices[3], vertices[4]);
    expect(inputFor(vertices), {0,1}, "Opposite-facing triangles joined");
    vertices = quad();
    vertices[5] = {{.2f,.2f,0},{.2f,.2f}};
    std::swap(vertices[3], vertices[4]);
    expect(inputFor(vertices), {0,1}, "Same-facing overlapping triangles joined");
    vertices = quad();
    const auto front = vertices;
    vertices.insert(vertices.end(), {front[1], front[0], front[2], front[4], front[3], front[5]});
    require(buildWindowIslandRoots(inputFor(vertices), &stats) == std::vector<uint32_t>({0,0,0,0}) &&
      stats.matchedEdges == 2 && stats.ambiguousEdges == 0, "Two-sided quad split or shared-edge candidates blocked");
    std::rotate(vertices.begin(), vertices.begin() + 3, vertices.end());
    expect(inputFor(vertices), {0,0,0,0}, "Two-sided quad split with interleaved facing");
    vertices = quad();
    vertices.insert(vertices.end(), {front[1], front[0], front[2]});
    expect(inputFor(vertices), {0,0,0}, "One backface blocked the original shared edge");
    vertices = quad();
    vertices.insert(vertices.end(), {front[1], front[0], front[2]});
    vertices[8].uv[0] += .001f;
    expect(inputFor(vertices), {0,0,2}, "Backface UV seam was joined");
    vertices = quad();
    const auto duplicated = vertices;
    vertices.insert(vertices.end(), duplicated.begin(), duplicated.begin() + 3);
    buildWindowIslandRoots(inputFor(vertices), &stats);
    require(stats.ambiguousEdges >= 3 && stats.mergedTriangles == 0, "Non-manifold duplicate face joined");
    vertices = quad();
    vertices[3].p[0] += .001f;
    vertices[4].p[0] += .001f;
    for (int i = 0; i < 5; ++i) {
      const float x = 1e8f + float(i) * 3e7f;
      vertices.insert(vertices.end(), {{{x,0,0},{0,0}}, {{x+1e7f,0,0},{1,0}}, {{x,1e7f,0},{0,1}}});
    }
    const auto roots = buildWindowIslandRoots(inputFor(vertices), &stats);
    require(roots[0] == 0 && roots[1] == 1 && stats.positionTolerance > 1, "Large faces swallowed a small edge's local tolerance");
  }
  void testBuffersAndTopology() {
    auto vertices = quad();
    auto input = inputFor(vertices);
    const uint16_t list16[] {0,1,2,3,4,5};
    input.indices = {reinterpret_cast<const uint8_t*>(list16), sizeof(list16)};
    input.indexCount = 6; input.indexStride = input.indexWidth = 2;
    expect(input, {0,0}, "Independent 16-bit indices split quad");
    const uint32_t list32[] {0,1,2,3,4,5};
    input.indices = {reinterpret_cast<const uint8_t*>(list32), sizeof(list32)};
    input.indexStride = input.indexWidth = 4;
    expect(input, {0,0}, "Independent 32-bit indices split quad");
    input.indices = input.indices.first(5 * sizeof(uint32_t));
    WindowTopologyDiagnostics stats;
    require(buildWindowIslandRoots(input, &stats) == std::vector<uint32_t>({0,1}) && stats.trianglesInvalid == 1,
      "Truncated triangle partially joined an earlier triangle");
    input = inputFor(vertices);
    input.positions = input.positions.first(5 * sizeof(Vertex));
    expect(input, {0,1}, "Truncated positions partially joined a triangle");
    input = inputFor(vertices);
    input.texcoords = input.texcoords.first(5 * sizeof(Vertex));
    expect(input, {0,1}, "Truncated UVs partially joined a triangle");
    input = inputFor(vertices); input.texcoords = {};
    buildWindowIslandRoots(input, &stats);
    require(stats.status == WindowTopologyStatus::MissingTexcoords, "Missing UV status incorrect");
    input = inputFor(vertices); input.texcoords = {}; input.texcoordStride = 0;
    buildWindowIslandRoots(input, &stats);
    require(stats.status == WindowTopologyStatus::MissingTexcoords, "Undefined UV buffer was classified as unsupported");
    input = inputFor(vertices); input.positions = {};
    buildWindowIslandRoots(input, &stats);
    require(stats.status == WindowTopologyStatus::MissingPositions, "Missing positions status incorrect");
    input = inputFor(vertices); input.positions = {}; input.positionStride = 0;
    buildWindowIslandRoots(input, &stats);
    require(stats.status == WindowTopologyStatus::MissingPositions, "Undefined position buffer was classified as unsupported");
    input = inputFor(vertices); input.indexWidth = input.indexStride = 2;
    buildWindowIslandRoots(input, &stats);
    require(stats.status == WindowTopologyStatus::MissingIndices, "Missing indices status incorrect");
    input = inputFor(vertices); input.indexWidth = 2; input.indexStride = 0;
    buildWindowIslandRoots(input, &stats);
    require(stats.status == WindowTopologyStatus::MissingIndices, "Undefined index buffer was classified as unsupported");
    input = inputFor(vertices); input.topology = WindowPrimitiveTopology::Unsupported;
    buildWindowIslandRoots(input, &stats);
    require(stats.status == WindowTopologyStatus::UnsupportedLayout, "Unsupported topology accepted");
    input = inputFor(vertices); input.indexWidth = input.indexStride = 1;
    buildWindowIslandRoots(input, &stats);
    require(stats.status == WindowTopologyStatus::UnsupportedLayout, "Unsupported index width classified as missing input");
    const uint32_t invalid[] {0,1,2,3,4,UINT32_MAX};
    input = inputFor(vertices); input.indices = {reinterpret_cast<const uint8_t*>(invalid), sizeof(invalid)};
    input.indexWidth = input.indexStride = 4; input.indexCount = 6;
    expect(input, {0,1}, "Invalid last index partially joined a triangle");
    std::vector<Vertex> wide(65542);
    std::copy(vertices.begin(), vertices.begin()+3, wide.begin());
    std::copy(vertices.begin()+3, vertices.end(), wide.begin()+65539);
    const uint32_t wideIndices[] {0,1,2,65539,65540,65541};
    input = inputFor(wide); input.triangleCount = 2;
    input.indices = {reinterpret_cast<const uint8_t*>(wideIndices), sizeof(wideIndices)};
    input.indexCount = 6; input.indexWidth = input.indexStride = 4;
    expect(input, {0,0}, "Full 32-bit vertex index was truncated");
    vertices = {{{0,0,0},{0,0}}, {{1,0,0},{1,0}}, {{0,1,0},{0,1}}, {{1,1,0},{1,1}}};
    input = inputFor(vertices); input.topology = WindowPrimitiveTopology::TriangleStrip; input.triangleCount = 2;
    expect(input, {0,0}, "Non-indexed strip winding split quad");
    const uint16_t strip[] {0,1,2,3};
    input.indices = {reinterpret_cast<const uint8_t*>(strip), sizeof(strip)};
    input.indexCount = 4; input.indexWidth = input.indexStride = 2;
    expect(input, {0,0}, "Indexed strip split quad");
    std::swap(vertices[2], vertices[3]);
    input = inputFor(vertices); input.topology = WindowPrimitiveTopology::TriangleFan; input.triangleCount = 2;
    expect(input, {0,0}, "Non-indexed fan split quad");
    input.indices = {reinterpret_cast<const uint8_t*>(strip), sizeof(strip)};
    input.indexCount = 4; input.indexWidth = input.indexStride = 2;
    expect(input, {0,0}, "Indexed fan split quad");
    const uint16_t degenerateFan[] {0,1,1,2,3};
    input.indices = {reinterpret_cast<const uint8_t*>(degenerateFan), sizeof(degenerateFan)};
    input.indexCount = 5; input.triangleCount = 3;
    require(buildWindowIslandRoots(input, &stats) == std::vector<uint32_t>({0,1,1}) && stats.degenerateTriangles == 1,
      "Degenerate fan renumbered primitives");
    const uint16_t degenerateStrip[] {0,0,1,3,2};
    input.indices = {reinterpret_cast<const uint8_t*>(degenerateStrip), sizeof(degenerateStrip)};
    input.topology = WindowPrimitiveTopology::TriangleStrip;
    require(buildWindowIslandRoots(input, &stats) == std::vector<uint32_t>({0,1,1}) && stats.degenerateTriangles == 1,
      "Degenerate strip renumbered primitives");
  }
  void testInvalidAndLimits() {
    auto vertices = quad();
    WindowTopologyDiagnostics stats;
    vertices[5].p[0] = std::numeric_limits<float>::quiet_NaN();
    require(buildWindowIslandRoots(inputFor(vertices), &stats) == std::vector<uint32_t>({0,1}) && stats.trianglesInvalid == 1,
      "Non-finite last position partially joined triangle");
    vertices = quad(); vertices[5].uv[1] = std::numeric_limits<float>::infinity();
    expect(inputFor(vertices), {0,1}, "Non-finite UV joined");
    vertices = quad(); vertices[5].p[0] = 1; vertices[5].p[1] = 0;
    require(buildWindowIslandRoots(inputFor(vertices), &stats) == std::vector<uint32_t>({0,1}) && stats.degenerateTriangles == 1,
      "Degenerate edge joined");
    vertices = quad();
    vertices.insert(vertices.end(), {{{1e20f,0,0},{0,0}}, {{1e20f+1e15f,0,0},{1,0}}, {{1e20f,1e15f,0},{0,1}}});
    require(buildWindowIslandRoots(inputFor(vertices), &stats) == std::vector<uint32_t>({0,1,2}) && stats.status == WindowTopologyStatus::NumericalRange,
      "Bucket range overflow returned partially merged roots");
    vertices = quad();
    const auto base = vertices;
    for (unsigned i = 0; i < 1300; ++i) {
      vertices.insert(vertices.end(), base.begin(), base.begin()+3);
    }
    const auto roots = buildWindowIslandRoots(inputFor(vertices), &stats);
    bool independent = true;
    for (size_t i = 0; i < roots.size(); ++i) {
      independent &= roots[i] == i;
    }
    require(independent && stats.status == WindowTopologyStatus::CandidateLimit && stats.mergedTriangles == 0,
      "Candidate limit returned partially merged roots");
  }
}

int main() {
  try {
    testMatching(); testBuffersAndTopology(); testInvalidAndLimits();
    std::cout << "Window topology: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Window topology failure after " << checks << " checks: " << error.what() << '\n';
    return -1;
  }
}
