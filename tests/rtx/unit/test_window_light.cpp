// SPDX-License-Identifier: MIT
#include <array>
#include <iostream>
#include <limits>
#include "../../test_utils.h"
#include "../../../src/dxvk/rtx_render/rtx_materials.h"
#include "../../../src/dxvk/rtx_render/rtx_window_light.h"
#include "../../../src/dxvk/rtx_render/rtx_window_topology.h"

namespace dxvk {
  Logger Logger::s_instance("test_window_light.log");
  void require(bool value, const char* message) { if (!value) { throw DxvkError(message); } }
  void testClock() {
    WindowLightClock clock;
    WindowLightArgs args;
    clock.update(args, {23.0f, 5u, false}, 1.0f, 19.0f, 5.0f, 42u);
    require(std::abs(args.litRatio - .6f) < .0001f && args.streetLampOn, "Evening clock incorrect");
    const uint32_t seed = args.nightSeed;
    clock.update(args, {0.0f, 6u, false}, 1.0f, 19.0f, 5.0f, 42u);
    require(args.nightSeed == seed && args.litRatio == .5f, "Midnight reshuffled windows");
    clock.update(args, {}, 1.0f, 19.0f, 5.0f, 42u);
    require(args.nightSeed == seed && args.litRatio == .5f, "Missing inputs reset clock");
    clock.update(args, {12.0f, 6u, false}, 1.0f, 19.0f, 5.0f, 42u);
    require(args.litRatio == 0 && !args.streetLampOn, "Daytime emission remained on");
    clock.update(args, {std::nullopt, std::nullopt, true}, 1.0f, 19.0f, 5.0f, 42u);
    require(args.litRatio == 0 && args.streetLampOn, "Bad weather did not independently enable lamps");
    clock.update(args, {19.0f, 6u, false}, 1.0f, 19.0f, 5.0f, 42u);
    require(args.nightSeed != seed && args.litRatio == 1, "New night reused previous pattern");
    clock.update(args, {std::numeric_limits<float>::quiet_NaN(), {}, {}}, .4f, 5.0f, 5.0f, 0);
    require(args.litRatio == .4f && args.streetLampOn, "Constant clock mode failed");
    clock.update(args, {30.0f, 1u, false}, 2.0f, 5.0f, 8.0f, 0);
    require(std::abs(args.litRatio - 2.0f / 3.0f) < .0001f, "Wrapped hours / ratio clamp incorrect");
  }
  void testRulesAndPacking() {
    WindowLightRules rules;
    require(rules.configure("  Glass;\nWINDOW\r\n; ") && rules.textureRuleCount() == 2, "Rule parsing failed");
    require(rules.matchesTextureName("C:/textures/WINDOW_lit.dds") && !rules.matchesTextureName("wall"), "Case-insensitive matching failed");
    require(!rules.configure("  Glass;\nWINDOW\r\n; "), "Unchanged rules rebuilt");
    rules.configure("");
    require(!rules.matchesTextureName("glass"), "Removed rules remained active");
    RtSurface surface;
    surface.positionBufferIndex = 0;
    surface.wetnessTag = 0x02000abc;
    std::array<unsigned char, kSurfaceGPUSize> baseline {}, classified {};
    size_t offset = 0;
    surface.writeGPUData(baseline.data(), offset);
    for (uint8_t cls = 0; cls <= 2; ++cls) {
      surface.windowLightClass = cls;
      offset = 0;
      surface.writeGPUData(classified.data(), offset);
      require(offset == 256, "Window expanded GPU ABI");
      uint32_t flags = 0;
      std::memcpy(&flags, classified.data() + 216, 4);
      require(((flags >> 15) & 3) == cls, "Window class packing failed");
      flags &= ~(3u << 15);
      std::memcpy(classified.data() + 216, &flags, 4);
      require(classified == baseline, "Window class changed other surface fields");
    }
  }
  void testEyes() {
    RtSurface surface;
    surface.positionBufferIndex = 0;
    surface.windowLightClass = 1;
    surface.wetnessTag = 0x020001ff;
    surface.eyeParams.emplace();
    surface.eyeParams->eyeballOrigin = Vector3(1, 2, 3);
    std::array<unsigned char, kSurfaceGPUSize> record {};
    size_t offset = 0;
    surface.writeGPUData(record.data(), offset);
    uint32_t flags = 0, packed = 0;
    std::memcpy(&flags, record.data() + 216, 4);
    std::memcpy(&packed, record.data() + 252, 4);
    float origin[3]; std::memcpy(origin, record.data() + 240, 12);
    require((flags & (1u << 14)) && ((flags >> 15) & 3) == 0, "Window class contaminated eyes");
    require(origin[0] == 1 && origin[1] == 2 && origin[2] == 3, "Eye origin was repacked as window data");
    require((packed >> 16) == 0x21ffu, "Eye/window packing changed wetness");
  }
  void testTopology() {
    const std::array<float, 27> positions {0,0,0, 1,0,0, 0,1,0, 0,1,0, 1,0,0, 1,1,0, 3,0,0, 4,0,0, 3,1,0};
    WindowTopologyInput input;
    input.positions = {reinterpret_cast<const uint8_t*>(positions.data()), sizeof(positions)};
    input.vertexCount = 9; input.positionStride = 12; input.triangleCount = 3;
    auto roots = buildWindowIslandRoots(input);
    require(roots == std::vector<uint32_t>({0,0,2}), "Duplicated quad vertices split, or separate windows joined");
    const uint16_t indices16[] {0,1,2, 2,1,5, 6,7,8};
    input.indices = {reinterpret_cast<const uint8_t*>(indices16), sizeof(indices16)};
    input.indexCount = 9; input.indexStride = input.indexWidth = 2;
    require(buildWindowIslandRoots(input) == roots, "16-bit indexed topology changed islands");
    const uint32_t indices32[] {0,1,2, 2,1,5, 6,7,8};
    input.indices = {reinterpret_cast<const uint8_t*>(indices32), sizeof(indices32)};
    input.indexStride = input.indexWidth = 4;
    require(buildWindowIslandRoots(input) == roots, "32-bit indexed topology changed islands");
    input.indices = input.indices.first(12);
    require(buildWindowIslandRoots(input) == std::vector<uint32_t>({0,1,2}), "Truncated indices escaped bounds");
    const uint32_t bad[] {UINT32_MAX, 10, 99};
    input.indices = {reinterpret_cast<const uint8_t*>(bad), sizeof(bad)};
    input.triangleCount = 1;
    require(buildWindowIslandRoots(input) == std::vector<uint32_t>({0}), "Out-of-range vertices escaped bounds");
    input.indices = {}; input.triangleCount = 3;
    require(buildWindowIslandRoots(input) == std::vector<uint32_t>({0,1,2}), "Unmappable topology fallback failed");
    // A true 32-bit vertex index must not alias its low 16 bits.
    std::vector<float> largePositions(65539 * 3, 0);
    largePositions[3] = 1; largePositions[7] = 1;
    largePositions[65536 * 3] = 10;
    largePositions[65537 * 3] = 11;
    largePositions[65538 * 3] = 10; largePositions[65538 * 3 + 1] = 1;
    const uint32_t wideIndices[] {0,1,2, 65536,65537,65538};
    input.positions = {reinterpret_cast<const uint8_t*>(largePositions.data()), largePositions.size() * sizeof(float)};
    input.indices = {reinterpret_cast<const uint8_t*>(wideIndices), sizeof(wideIndices)};
    input.vertexCount = 65539; input.indexCount = 6; input.triangleCount = 2;
    require(buildWindowIslandRoots(input) == std::vector<uint32_t>({0,1}), "32-bit index was truncated");
  }
}
int main() {
  try {
    dxvk::testClock(); dxvk::testRulesAndPacking(); dxvk::testEyes(); dxvk::testTopology();
    std::cout << "Window clock, packing, matching and topology tests passed\n";
  } catch (const dxvk::DxvkError& error) {
    dxvk::Logger::err(error.message());
    return 1;
  }
}
