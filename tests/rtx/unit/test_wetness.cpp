// SPDX-License-Identifier: MIT
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>

#include "../../test_utils.h"
#include "../../../src/dxvk/rtx_render/rtx_materials.h"
#include "../../../src/dxvk/rtx_render/rtx_wetness_zones.h"
#include "../../../src/dxvk/rtx_render/rtx_wetness_preset.h"
#include "../../../src/dxvk/rtx_render/rtx_wetness_transition.h"
#include "../../../src/dxvk/rtx_render/rtx_wetness_dff_rules.h"
#include "../../../src/frontends/translator/draw_key.h"

namespace dxvk {
  Logger Logger::s_instance("test_wetness.log");

  void require(bool value, const char* message) {
    if (!value) {
      throw DxvkError(message);
    }
  }

  void testPackingAndIdentity() {
    static_assert(kSurfaceGPUSize == 256);
    RtSurface surface;
    surface.positionBufferIndex = 0;
    std::array<unsigned char, kSurfaceGPUSize> dry {}, wet {};
    size_t offset = 0;
    surface.writeGPUData(dry.data(), offset);
    for (uint32_t surfaceClass = 0; surfaceClass < 256; ++surfaceClass) {
      for (uint32_t flags = 0; flags < 4096; flags += 17) {
        surface.wetnessTag = (surfaceClass << 24) | flags;
        offset = 0;
        surface.writeGPUData(wet.data(), offset);
        require(offset == wet.size(), "Surface layout grew");
        require(std::memcmp(dry.data(), wet.data(), wet.size() - 2) == 0,
          "Wetness changed geometry, material or shell-slot bytes");
        uint32_t packed = 0;
        std::memcpy(&packed, wet.data() + wet.size() - 4, 4);
        const uint32_t decoded = ((packed >> 16) & 0xFFFu) | ((packed >> 28) << 24);
        const uint32_t expected = surfaceClass >= 1 && surfaceClass <= 3 ? surface.wetnessTag : 0;
        require(decoded == expected, "Wetness packing changed classification or flags");
      }
    }
    translator::DrawInputs a;
    a.material = 123;
    a.streamMask = 1;
    a.streams[0].buffer = { 1, 5 };
    a.streams[0].stride = 32;
    a.indices = { 2, 9 };
    for (uint8_t flag : { uint8_t(0), uint8_t(translator::DrawInputs::kDynamicSource), uint8_t(translator::DrawInputs::kUserPointer) }) {
      a.flags = flag;
      auto b = a;
      b.extra = a.extra ^ 42;
      require(a.identityKey() == b.identityKey(), "Rain changed retained object identity");
      require(a.contentSignature() == b.contentSignature(), "Rain changed instance matching identity");
      require(a.inputsKey(true) != b.inputsKey(true), "Rain change was treated as an unchanged draw");
      require(a.inputsKey(false, false) != b.inputsKey(false, false), "Rain change was treated as a transform-only update");
    }
  }

  void testZonesAndTransition() {
    WetnessZoneIndex zones;
    require(zones.update("1 2 0 1 0 1 0 1 4 6 4 6 4 6"), "Valid no-rain zones rejected");
    require(zones.contains(0.5f, 0.5f, 0.5f), "First leaf missing");
    require(zones.contains(5, 5, 5), "Second leaf missing");
    require(!zones.contains(3, 3, 3), "Gap between leaves blocked rain");
    require(!zones.contains(0, 0.5f, 0.5f), "Open boundary changed");
    for (const char* invalid : { "2 0", "1 4097", "1 1 1 0 0 1 0 1", "1 0 trailing", "1 1 nan 1 0 1 0 1" }) {
      require(!zones.update(invalid), "Invalid no-rain input accepted");
      require(zones.contains(5, 5, 5), "Invalid input discarded the last valid zone snapshot");
    }
    require(zones.update("1 0") && zones.nodes().empty(), "Empty zone update did not clear zones");
    WetnessTransition transition;
    require(transition.update(true, std::numeric_limits<float>::quiet_NaN(), 1) == 0, "NaN time changed transition");
    require(transition.update(true, 1, 1) == 1, "Rain transition did not reach its target");
    require(transition.update(false, 1, 1) == 0, "Dry transition did not return to authored roughness");
  }

  void testPresetBlend() {
    WetnessArgs base;
    base.puddleWaveStrength = 0.25f;
    base.waterWaveStrength = 0.4f;
    WetnessPreset preset;
    size_t rain = 0;
    size_t wave = 0;
    size_t waterWave = 0;
    for (size_t i = 0; i < std::size(kWetnessParameters); ++i) {
      if (kWetnessParameters[i].member == &WetnessArgs::weatherWetness) {
        rain = i;
      }
      if (kWetnessParameters[i].member == &WetnessArgs::puddleWaveStrength) {
        wave = i;
      }
      if (kWetnessParameters[i].member == &WetnessArgs::waterWaveStrength) {
        waterWave = i;
      }
    }
    preset.values[rain] = 0.2f;
    preset.values[wave] = 2.0f;
    preset.relative[wave] = true;
    preset.values[waterWave] = 0.8f;
    WetnessArgs rainy = base;
    preset.apply(rainy);
    require(rainy.weatherWetnessActive == 65535 && rainy.puddleWaveStrength == 0.5f,
      "Preset rain override or relative parameter failed");
    const WetnessArgs half = WetnessPreset::blend(base, rainy, 0.5f);
    require(half.weatherWetnessActive == 32768 && std::abs(half.weatherWetness - 0.2f) < 0.0001f,
      "Unconfigured rain blended as a constant instead of retaining the instance tag");
    require(half.puddleWaveStrength == rainy.puddleWaveStrength, "Incoming puddle waveform changed during transition");
    require(std::abs(half.waterWaveStrength - 0.6f) < 0.0001f, "Linear wetness parameter did not interpolate");
    const WetnessArgs start = WetnessPreset::blend(base, rainy, 0);
    const WetnessArgs end = WetnessPreset::blend(base, rainy, 1);
    require(start.weatherWetnessActive == 0 && end.weatherWetnessActive == 65535, "Rain override endpoints changed");
    WetnessArgs dry = rainy;
    dry.weatherWetness = 0.0f;
    require(std::abs(WetnessPreset::blend(dry, rainy, 0.5f).weatherWetness - 0.1f) < 0.0001f,
      "Two authored rain intensities did not blend");
    require(WetnessPreset::blend(base, rainy, std::numeric_limits<float>::quiet_NaN()).weatherWetnessActive == 0,
      "NaN weather transition escaped into frame constants");
    preset.values[wave] = std::numeric_limits<float>::infinity();
    preset.apply(base);
    require(base.puddleWaveStrength == 0.25f, "Invalid authored parameter reached the renderer");
  }

  void testRules(const std::filesystem::path& parent) {
    const auto root = parent / std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    require(!std::filesystem::exists(root), "Test folder already exists");
    std::filesystem::create_directories(root);
    std::ofstream(root / "road.json") << R"({"schema":1,"dff":"ROAD_test.dff","meshes":[{"materials":[{"texture":"ROADTEX","draw_calls":[{"hashes":[{"remix_mesh_hash":"0x11","texture0_hash":"aa"},{"remix_mesh_hash":"22"}]}]}]}]})";
    std::ofstream(root / "other.json") << R"({"schema":1,"dff":"other.dff","meshes":[{"unmatched_draw_calls":[{"hashes":[{"remix_mesh_hash":"22"}]}]}]})";
    std::ofstream(root / "invalid.json") << "not json";
    WetnessDffRules rules;
    rules.refresh(root, " road ", "test");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (rules.loading() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      rules.refresh(root, " road ", "test");
    }
    require(!rules.loading(), "Capture mapping load timed out");
    require(rules.rule(0x11) == WetnessDffRules::Rule::Disable, "Disable did not win over enable");
    require(rules.rule(0x22) == WetnessDffRules::Rule::Default, "Ambiguous shared mesh received guessed rules");
    require(rules.originalTextureName(0xaa) && *rules.originalTextureName(0xaa) == "roadtex", "Original texture mapping missing");
    require(rules.ambiguousHashes() == 1 && rules.skippedFiles() == 1, "Capture diagnostics incorrect");
    rules.refresh(root, "road", "");
    require(rules.rule(0x11) == WetnessDffRules::Rule::Enable, "Rule change was not applied live");
    rules.refresh(root, "", "");
    require(rules.rule(0x11) == WetnessDffRules::Rule::Default, "Removed rule remained latched");
  }
}

int main(int argc, char** argv) {
  try {
    dxvk::require(argc == 2, "Pass a test output directory");
    dxvk::testPackingAndIdentity();
    dxvk::testZonesAndTransition();
    dxvk::testRules(argv[1]);
    dxvk::testPresetBlend();
    std::cout << "Wetness packing, retained identity, zones, transitions and DFF rules passed\n";
  } catch (const dxvk::DxvkError& error) {
    dxvk::Logger::err(error.message());
    return 1;
  }
}
