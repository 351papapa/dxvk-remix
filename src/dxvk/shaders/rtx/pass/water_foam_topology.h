#pragma once
#define WATER_FOAM_AUTHORED_VERTICES (256u*256u*6u)
#define WATER_FOAM_COARSE_TRIANGLES (60u*60u*4u)
#define WATER_FOAM_FINE_TRIANGLES (90u*90u*4u)
#define WATER_FOAM_LIVE_TRIANGLES (WATER_FOAM_COARSE_TRIANGLES+WATER_FOAM_FINE_TRIANGLES)
#define WATER_FOAM_COARSE_SUBDIVISIONS 4u
#define WATER_FOAM_COARSE_VERTICES (WATER_FOAM_COARSE_TRIANGLES*3u*WATER_FOAM_COARSE_SUBDIVISIONS*WATER_FOAM_COARSE_SUBDIVISIONS)
#define WATER_FOAM_LIVE_VERTICES (WATER_FOAM_COARSE_VERTICES+WATER_FOAM_FINE_TRIANGLES*3u)
#define WATER_FOAM_TOTAL_VERTICES (WATER_FOAM_AUTHORED_VERTICES+WATER_FOAM_LIVE_VERTICES)

#ifdef __cplusplus
#include <cstdint>
using WaterFoamSlot = uint32_t;
#else
typedef uint WaterFoamSlot;
#endif

// Keep the remainder's dividend nonnegative in both CPU and GPU arithmetic.
// The adjusted negation also covers the most negative signed cell coordinate.
inline WaterFoamSlot waterFoamWorldCellSlot(int cell, WaterFoamSlot side) {
  WaterFoamSlot magnitude = cell < 0 ? (WaterFoamSlot)(-(cell + 1)) + 1u : (WaterFoamSlot)cell;
  WaterFoamSlot remainder = magnitude % side;
  return cell < 0 && remainder > 0u ? side - remainder : remainder;
}
