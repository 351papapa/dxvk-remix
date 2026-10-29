#ifndef WATER_FLOW_ARGS_H
#define WATER_FLOW_ARGS_H
#ifdef __cplusplus
#include <cstdint>
using FlowUint = uint32_t;
using FlowInt = int32_t;
#else
typedef uint FlowUint;
typedef int FlowInt;
#endif
#define FLOW_STATE_SOURCE 222
#define FLOW_STATE_TARGET 223
#define FLOW_PRESSURE_SOURCE 224
#define FLOW_PRESSURE_TARGET 225
#define FLOW_DIVERGENCE 226
#define FLOW_SOURCES 227
#define FLOW_VERTICES 228
#define FLOW_WATER_MAP 229
#define FLOW_FORCES 230
#define FLOW_INDICES 231
#define FLOW_SIDE 129
#define FLOW_COUNT (FLOW_SIDE * FLOW_SIDE)
#define FLOW_GROUP_SIZE 64
struct WaterFlowArgs {
  FlowUint operation, sourceCount, forceCount;
  FlowInt shiftX, shiftY;
  float originX, originY, dt, spacing;
  float drag, strength, swirl, lifetime, time;
  float waterMapOriginX, waterMapOriginY, waterMapCellSize;
};
struct WaterFlowSource { float x,y,radius,amount,vx,vy; };
struct WaterFlowForce { float x,y,vx,vy,radius,length,pad0,pad1; };
#endif
