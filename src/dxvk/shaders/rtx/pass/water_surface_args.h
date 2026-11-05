#ifndef WATER_SURFACE_ARGS_H
#define WATER_SURFACE_ARGS_H
#ifdef __cplusplus
#include <cstdint>
using WaterSurfaceUint = uint32_t;
#else
typedef uint WaterSurfaceUint;
#endif
#define WATER_SURFACE_INPUT 222
#define WATER_SURFACE_POSITIONS 223
#define WATER_SURFACE_VERTICES 224
#define WATER_SURFACE_SHORE_DEPTH 228
#define WATER_SURFACE_SHORE_TERRAIN 229
#define WATER_SURFACE_SHORE_SEEDS 230
#define WATER_SURFACE_SHORE_NEXT 231
#define WATER_SURFACE_BEACH_MASK 232
#define WATER_RENDER_CELL_COUNT (60 * 60 + 90 * 90)
#define WATER_RENDER_VERTEX_COUNT (WATER_RENDER_CELL_COUNT * 12)
#define WATER_RENDER_INDEX_COUNT WATER_RENDER_VERTEX_COUNT
#define WATER_SURFACE_GROUP_SIZE 64
// Conservative sum of the RMS-normalized subwave weights used by ocean geometry.
#define WATER_OCEAN_AMPLITUDE_BOUND_SCALE 1.471863f
#define WATER_COARSE_COUNT 3721
#define WATER_FINE_COUNT 8281
#define WATER_VERTEX_COUNT (WATER_COARSE_COUNT + WATER_FINE_COUNT)
#define WATER_HEIGHT 32
#define WATER_WET (WATER_HEIGHT + WATER_COARSE_COUNT)
#define WATER_FADE (WATER_WET + WATER_COARSE_COUNT)
#define WATER_COARSE_MASK (WATER_FADE + WATER_COARSE_COUNT)
#define WATER_FINE_MASK (WATER_COARSE_MASK + 3600)
#define WATER_FINE_WET (WATER_FINE_MASK + 8100)
#define WATER_SECTIONS (WATER_FINE_WET + WATER_FINE_COUNT)
#define WATER_SPLASHES (WATER_SECTIONS + 84 * 8)
struct WaterSurfaceArgs {
  WaterSurfaceUint phase;
  float beachWaveHeight;
  float beachWaveLength;
  float beachWavePeriod;
  float beachWaveDepth;
  float timeSeconds;
  float oceanWaveStrength;
};
#endif
