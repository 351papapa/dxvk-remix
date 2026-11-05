// SPDX-License-Identifier: MIT
// Local runtime feature source. See CREDITS.md.
#include "rtx/pass/post_fx/lens_rain.h"

layout(binding = LENS_RAIN_DROPS) RWStructuredBuffer<LensRainDrop> Drops;
layout(binding = LENS_RAIN_TILES) RWStructuredBuffer<uint> DropTiles;
layout(binding = LENS_RAIN_STATES) RWStructuredBuffer<LensRainState> States;
layout(push_constant) ConstantBuffer<LensRainArgs> args;

float randomDrop(inout uint seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return float(seed >> 8) * (1.0f / 16777216.0f);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint id : SV_DispatchThreadID) {
  if (id >= args.dropCount) { return; }
  LensRainState drop;
  if (id >= args.previousCount) {
    drop = (LensRainState)0;
    drop.seed = 0x738a2d51u ^ ((id + 1u) * 0x9e3779b9u);
  } else {
    drop = States[id];
    if (args.previousAspect > 0) { drop.position.x *= args.aspect / args.previousAspect; }
  }
  float margin = drop.radius * args.sizeScale * 2.0f + 0.01f;
  bool expired = drop.lifetime <= 0 || drop.age >= drop.lifetime
    || drop.position.x < -margin || drop.position.x > args.aspect + margin
    || drop.position.y < -margin || drop.position.y > 1.0f + margin;
  if (expired) {
    drop.lifetime = 0;
    if (!args.allowSpawning) { States[id] = drop; return; }
    drop.position = float2(randomDrop(drop.seed) * args.aspect, randomDrop(drop.seed));
    drop.velocity = float2(0);
    drop.axis = float2(0, 1);
    drop.motionDirection = normalize(float2((randomDrop(drop.seed) - 0.5f) * 0.5f,
      0.1f + randomDrop(drop.seed) * 0.65f));
    bool fine = randomDrop(drop.seed) < 0.35f;
    drop.radius = (0.13f + randomDrop(drop.seed) * 0.17f) / args.density * (fine ? 0.55f : 1.0f);
    drop.lifetime = 3.0f + randomDrop(drop.seed) * 7.0f;
    drop.age = id >= args.previousCount ? drop.lifetime * (0.05f + randomDrop(drop.seed) * 0.65f) : 0;
    drop.mobility = (0.25f + randomDrop(drop.seed) * 1.25f) * (fine ? 0.4f : 1.0f);
    float gravityA = randomDrop(drop.seed);
    float gravityB = randomDrop(drop.seed);
    drop.gravity = (0.002f + gravityA * gravityB * 0.04f) * (fine ? 0.2f : 1.0f);
    drop.responseTime = 0.07f + randomDrop(drop.seed) * 0.28f;
    drop.drift = (randomDrop(drop.seed) - 0.5f) * 0.012f;
    drop.phase = randomDrop(drop.seed) * 6.2831853f;
    if (randomDrop(drop.seed) < (1.0f / 6.0f)) {
      drop.mobility *= 0.03f;
      drop.gravity *= 0.05f;
      drop.drift *= 0.05f;
      drop.lifetime += 1.25f;
    }
    drop.strength = fine ? 0.5f : 0.8f + randomDrop(drop.seed) * 0.3f;
  }
  drop.age += args.dt * args.slideSpeed;
  float slip = 0.7f + 0.3f * sin(drop.age * 1.7f + drop.phase);
  float2 target = args.flowVelocity.xy * (drop.mobility * 0.25f)
    + drop.motionDirection * (args.flowVelocity.z * drop.mobility * slip)
    + float2(drop.drift * sin(drop.age + drop.phase), drop.gravity) * args.slideSpeed;
  drop.velocity += (target - drop.velocity) * (1.0f - exp(-args.dt / drop.responseTime));
  drop.position += drop.velocity * args.dt;
  float speed = length(drop.velocity);
  if (speed > 0.002f) {
    drop.axis += (drop.velocity / speed - drop.axis) * (1.0f - exp(-args.dt / 0.15f));
    float axisLength = length(drop.axis);
    drop.axis = axisLength > 0.0001f ? drop.axis / axisLength : float2(0, 1);
  }
  States[id] = drop;
  float life = saturate(drop.age / 0.2f)
    * saturate((drop.lifetime - drop.age) / (drop.lifetime * 0.25f));
  if (life <= 0.001f) { return; }
  float radius = drop.radius * args.sizeScale;
  float longRadius = radius * (1.0f + min(speed * 1.2f, 0.9f));
  float bound = longRadius + 3.0f / args.imageSize.y;
  if (drop.position.x + bound < 0 || drop.position.x - bound > args.aspect
    || drop.position.y + bound < 0 || drop.position.y - bound > 1) { return; }
  LensRainDrop visible;
  visible.centerRadii = float4(drop.position, radius, longRadius);
  visible.axisOpacityStrength = float4(drop.axis, life, drop.strength);
  Drops[id] = visible;
  float2 tileScale = float2(args.tileCount) / float2(args.aspect, 1);
  uint2 lo = uint2(clamp(floor((drop.position - bound) * tileScale), 0, float2(args.tileCount - 1)));
  uint2 hi = uint2(clamp(floor((drop.position + bound) * tileScale), 0, float2(args.tileCount - 1)));
  uint headerSize = args.tileCount.x * args.tileCount.y * 2;
  for (uint y = lo.y; y <= hi.y; ++y) {
    for (uint x = lo.x; x <= hi.x; ++x) {
      uint tile = y * args.tileCount.x + x;
      uint slot;
      InterlockedAdd(DropTiles[tile * 2 + 1], 1, slot);
      // Each of the 1024 drop slots visits a tile at most once.
      DropTiles[headerSize + tile * LENS_RAIN_MAX_DROPS + slot] = id;
    }
  }
}
