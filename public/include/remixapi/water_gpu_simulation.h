#pragma once
#include <cstdint>
#include "water_gpu_surface.h"

namespace remixapi::water::simulation {
inline constexpr wchar_t kMappingName[] = L"Local\\RTXVC.WaterGpuSimulation.v3";
inline constexpr wchar_t kMutexName[] = L"Local\\RTXVC.WaterGpuSimulation.v3.Lock";
inline constexpr uint32_t kVersion = 3, kQueueSize = 8, kMaxImpulses = 512;
inline constexpr uint32_t kFoamFlow = 1, kHullWater = 2;
inline constexpr uint32_t kMaxFlowSources = 512, kMaxFlowForces = 32;
struct FlowSource { float x, y, radius, amount, vx, vy; };
struct FlowForce { float x, y, vx, vy, radius, length, pad0 = 0, pad1 = 0; };
struct FlowFrame {
  uint32_t enabled = 0, resetSerial = 0, sourceCount = 0, forceCount = 0;
  float originX = 0, originY = 0, dt = 0, lifetime = 7, time = 0;
  FlowSource sources[kMaxFlowSources] = {};
  FlowForce forces[kMaxFlowForces] = {};
};
inline constexpr uint32_t kFoamVertices = 128 * 128 * 4;
inline constexpr uint32_t kFoamIndices = 128 * 128 * 6;
struct Impulse { float x, y, radius, strength; uint32_t minX, minY, maxX, maxY; };
struct Vertex { float position[3], normal[3], uv[2]; uint32_t color; };
struct Packet {
  gpu::Frame surface;
  uint32_t sequence = 0, resetSerial = 0, impulseCount = 0, steps = 0;
  float originX = 0, originY = 0, spacing = 1, stepSeconds = 1.f / 120;
  float speed = 3, damping = .65f, maxHeight = .75f;
  uint32_t circular = 1;
  float mask[gpu::kCoarseCount][2] = {};
  float seed[gpu::kCoarseCount][2] = {};
  Impulse impulses[kMaxImpulses] = {};
  FlowFrame flow;
};
struct Shared {
  uint32_t version = kVersion, bytes = sizeof(Shared), ready = 0, readyTick = 0;
  uint32_t active = 0, tick = 0, epoch = 0, writeSequence = 0, readSequence = 0;
  uint32_t flowReady = 0, waterMapReady = 0;
  // Row-major y*128+x, sampled at native Vice City water-cell centres.
  float waterMap[128 * 128];
  Packet packets[kQueueSize];
  uint32_t foamSequence = 0, foamVertexCount = 0, foamIndexCount = 0;
  Vertex foamVertices[kFoamVertices];
  uint32_t foamIndices[kFoamIndices];
};
static_assert(sizeof(Impulse) == 32 && sizeof(Vertex) == 36);
}
