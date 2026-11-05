#ifndef WATER_HULL_ARGS_H
#define WATER_HULL_ARGS_H
#ifdef __cplusplus
#include <cstdint>
using HullUint = uint32_t;
using HullInt = int32_t;
#else
typedef uint HullUint;
typedef int HullInt;
#endif
#define HULL_SIDE 129
#define HULL_COUNT (HULL_SIDE * HULL_SIDE)
struct WaterHullArgs {
  HullUint operation;
  HullInt shiftX, shiftY;
  HullUint forceCount;
  float originX, originY, spacing, dt;
  float speed, damping, strength, maxHeight;
  float waterMapOriginX, waterMapOriginY, waterMapCellSize;
};
struct WaterHullForce { float x,y,vx,vy,radius,length,pad0,pad1; };
#endif
