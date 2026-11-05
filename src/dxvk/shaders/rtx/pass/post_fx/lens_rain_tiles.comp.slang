// SPDX-License-Identifier: MIT
// Local runtime feature source. See CREDITS.md.
#include "rtx/pass/post_fx/lens_rain.h"

layout(binding = LENS_RAIN_TILES) RWStructuredBuffer<uint> DropTiles;
layout(push_constant) ConstantBuffer<LensRainArgs> args;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint id : SV_DispatchThreadID) {
  uint tileTotal = args.tileCount.x * args.tileCount.y;
  if (id >= tileTotal) { return; }
  DropTiles[id * 2] = tileTotal * 2 + id * LENS_RAIN_MAX_DROPS;
  DropTiles[id * 2 + 1] = 0;
}
