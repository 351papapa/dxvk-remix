#pragma once
#include <cstdint>

// Fixed-width, pointer-free contract shared by the 64-bit runtime and 32-bit water integration.
namespace remixapi::water {
constexpr wchar_t kMappingName[] = L"Local\\RTXVC.WaterControls.v3";
constexpr wchar_t kMutexName[] = L"Local\\RTXVC.WaterControls.v3.Lock";
constexpr uint32_t kVersion = 3;
struct Controls {
  uint32_t version = kVersion;
  uint32_t bytes = sizeof(Controls);
  uint32_t tick = 0;
  uint32_t cameraValid = 0;
  float amplitude = 0.55f;
  float wavelength = 12.0f;
  float speed = 1.0f;
  float steepness = 0.55f;
  float interactionStrength = 2.0f;
  float interactionRadius = 1.5f;
  float propagationSpeed = 4.0f;
  float damping = 0.45f;
  uint32_t reduceOffscreen = 1;
  float worldToClip[16] = {};
  float simulationRadius = 54.0f;
  uint32_t enabled = 1;
};
static_assert(sizeof(Controls) == 124);
}
