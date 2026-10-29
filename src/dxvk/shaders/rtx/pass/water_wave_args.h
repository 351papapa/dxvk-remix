#ifndef WATER_WAVE_ARGS_H
#define WATER_WAVE_ARGS_H
#ifdef __cplusplus
#include <cstdint>
#include <algorithm>
#include <cmath>
using WaterWaveUint = uint32_t;
using WaterWaveInt = int32_t;
#else
typedef uint WaterWaveUint;
typedef int WaterWaveInt;
#endif
#define WATER_WAVE_SOURCE 222
#define WATER_WAVE_TARGET 223
#define WATER_WAVE_MASK 224
#define WATER_WAVE_IMPULSES 225
#define WATER_WAVE_GROUP_SIZE 64
#define WATER_WAVE_SIDE 61
#define WATER_WAVE_COUNT 3721
#define WATER_WAVE_RESET 0
#define WATER_WAVE_SHIFT 1
#define WATER_WAVE_IMPULSE 2
#define WATER_WAVE_STEP 3
#define WATER_WAVE_APPLY_MASK 4
struct WaterWaveArgs {
  WaterWaveUint operation;
  WaterWaveInt shiftX;
  WaterWaveInt shiftY;
  WaterWaveUint impulseCount;
  float spacing;
  float stepSeconds;
  float speed;
  float damping;
  float maxHeight;
  WaterWaveUint circular;
};
struct WaterWaveImpulse {
  float x, y, radius, strength;
  WaterWaveUint minX, minY, maxX, maxY;
};
#ifdef __cplusplus
inline WaterWaveImpulse prepareWaterWaveImpulse(float x, float y, float radius, float strength, float spacing) {
  WaterWaveImpulse result {};
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(radius) || !std::isfinite(strength)) {
    return result;
  }
  const float requested = std::clamp(radius, .01f, 6.f);
  result.x = x;
  result.y = y;
  result.radius = std::clamp(requested, spacing, 6.f);
  result.strength = std::clamp(strength, -8.f, 8.f) * (requested / result.radius);
  result.minX = uint32_t(std::clamp(std::floor((x - 4 * result.radius) / spacing), 1.f, 59.f));
  result.minY = uint32_t(std::clamp(std::floor((y - 4 * result.radius) / spacing), 1.f, 59.f));
  result.maxX = uint32_t(std::clamp(std::ceil((x + 4 * result.radius) / spacing), 1.f, 59.f));
  result.maxY = uint32_t(std::clamp(std::ceil((y + 4 * result.radius) / spacing), 1.f, 59.f));
  return result;
}
static_assert(sizeof(WaterWaveArgs) == 40 && sizeof(WaterWaveImpulse) == 32);
#endif
#endif
