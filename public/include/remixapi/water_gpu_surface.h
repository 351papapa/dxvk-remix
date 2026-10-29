#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>

namespace remixapi::water::gpu {
inline constexpr wchar_t kMappingName[] = L"Local\\RTXVC.WaterGpuSurface.v1";
inline constexpr wchar_t kMutexName[] = L"Local\\RTXVC.WaterGpuSurface.v1.Lock";
inline constexpr uint32_t kVersion = 1;
inline constexpr uint32_t kCoarseSide = 61, kFineSide = 91;
inline constexpr uint32_t kCoarseCount = kCoarseSide * kCoarseSide;
inline constexpr uint32_t kFineCount = kFineSide * kFineSide;
inline constexpr uint32_t kVertexCount = kCoarseCount + kFineCount;
inline constexpr uint32_t kIndexCount = (60 * 60 + 90 * 90) * 6;
// Scalar offsets also consumed by water_surface_args.h; no pointers or native bools in IPC.
inline constexpr uint32_t kHeight = 32;
inline constexpr uint32_t kWet = kHeight + kCoarseCount;
inline constexpr uint32_t kFade = kWet + kCoarseCount;
inline constexpr uint32_t kCoarseMask = kFade + kCoarseCount;
inline constexpr uint32_t kFineMask = kCoarseMask + 60 * 60;
inline constexpr uint32_t kFineWet = kFineMask + 90 * 90;
inline constexpr uint32_t kSections = kFineWet + kFineCount;
inline constexpr uint32_t kSplashes = kSections + 84 * 8;
inline constexpr uint32_t kFloatCount = kSplashes + 4 * 4;
struct Frame {
  uint32_t version = kVersion, bytes = sizeof(Frame);
  uint32_t tick = 0, sequence = 0, active = 0;
  uint32_t readyTick = 0, ready = 0;
  float data[kFloatCount] = {};
};
static_assert(offsetof(Frame, data) == 28);
static_assert(sizeof(Frame) == 28 + kFloatCount * sizeof(float));
inline void prepareWaveConstants(float* f) {
  constexpr float weights[4] = {1.f,.48f,.24f,.3f};
  constexpr float lengths[4] = {1.f,.61f,.37f,1.73f};
  const float amplitude = std::clamp(f[5],0.f,2.f), wavelength = std::clamp(f[6],6.f,40.f);
  float slope = 0;
  for (uint32_t i=0;i<4;++i) {
    f[16+i] = 6.2831853f/(wavelength*lengths[i]);
    f[20+i] = std::sqrt(9.81f*f[16+i])*f[4];
    f[24+i] = amplitude*weights[i];
    slope += f[16+i]*amplitude*weights[i];
  }
  f[28] = std::min(std::clamp(f[7],0.f,.9f),.75f/std::max(slope,.001f));
  f[29] = f[8] + 90.f * f[9];
}
}
