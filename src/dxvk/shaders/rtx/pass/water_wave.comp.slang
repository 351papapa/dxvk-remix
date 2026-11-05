#include "rtx/pass/water_wave_args.h"

layout(push_constant) ConstantBuffer<WaterWaveArgs> args;
layout(binding = WATER_WAVE_SOURCE) StructuredBuffer<float2> source;
layout(binding = WATER_WAVE_TARGET) RWStructuredBuffer<float2> target;
layout(binding = WATER_WAVE_MASK) StructuredBuffer<float2> mask;
layout(binding = WATER_WAVE_IMPULSES) StructuredBuffer<WaterWaveImpulse> impulses;

[shader("compute")]
[numthreads(WATER_WAVE_GROUP_SIZE,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if (id >= WATER_WAVE_COUNT) { return; }
  int2 cell = int2(id % WATER_WAVE_SIDE, id / WATER_WAVE_SIDE);
  bool interior = all(cell > 0) && all(cell < WATER_WAVE_SIDE - 1);
  if (args.operation == WATER_WAVE_RESET) {
    target[id] = float2(0);
    return;
  }
  if (args.operation == WATER_WAVE_SHIFT) {
    int2 oldCell = cell + int2(args.shiftX, args.shiftY);
    bool valid = interior && all(oldCell > 0) && all(oldCell < WATER_WAVE_SIDE - 1);
    if (valid) { target[id] = source[oldCell.y * WATER_WAVE_SIDE + oldCell.x]; }
    else { target[id] = float2(0); }
    return;
  }
  float2 value = source[id];
  if (args.operation == WATER_WAVE_APPLY_MASK) {
    target[id] = mask[id].x > 0 ? value : float2(0);
    return;
  }
  if (args.operation == WATER_WAVE_IMPULSE) {
    if (interior && mask[id].x > 0) {
      for (uint i = 0; i < args.impulseCount; ++i) {
        WaterWaveImpulse impulse = impulses[i];
        if (impulse.radius <= 0) { continue; }
        float radius = impulse.radius;
        float strength = impulse.strength;
        int2 lower = int2(impulse.minX, impulse.minY);
        int2 upper = int2(impulse.maxX, impulse.maxY);
        if (any(cell < lower) || any(cell > upper)) { continue; }
        float2 delta = float2(cell) * args.spacing - float2(impulse.x, impulse.y);
        float q = dot(delta, delta) / (radius * radius);
        value.y += strength * (1 - .5f * q) * exp(-.5f * q);
      }
    }
  } else if (args.operation == WATER_WAVE_STEP) {
    float dt = clamp(args.stepSeconds, 1.f / 120.f, 1.f / 30.f);
    if (interior && mask[id].x > 0) {
      float lap = (source[id - 1].x + source[id + 1].x
        + source[id - WATER_WAVE_SIDE].x + source[id + WATER_WAVE_SIDE].x
        - 4 * value.x) / (args.spacing * args.spacing);
      float edge = min(min(cell.x, cell.y), min(60 - cell.x, 60 - cell.y));
      float loss = args.circular != 0 ? 1 - mask[id].y : max(0.f, 1 - edge / 5);
      float damping = clamp(args.damping, .05f, 3.f) + 6 * loss;
      float speed = clamp(args.speed, 1.f, 8.f);
      value.y = (value.y + speed * speed * lap * dt) * exp(-damping * dt);
    }
    float limit = isfinite(args.maxHeight) ? clamp(args.maxHeight, .05f, 3.f) : .75f;
    float next = value.x + value.y * dt;
    value.x = clamp(next, -limit, limit);
    if ((next >= limit && value.y > 0) || (next <= -limit && value.y < 0)) { value.y = 0; }
  }
  target[id] = value;
}
