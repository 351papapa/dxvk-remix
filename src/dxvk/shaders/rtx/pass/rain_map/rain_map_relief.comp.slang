// Adapted from xoxor4d/dxvk-remix, commit 7f5210e1a7cd1356bf0271230f82e6132b08ad8c.
// Modified for world units and the runtime's rain-map subsystem.
// The original zlib/libpng license is retained in ThirdPartyLicenses.txt.

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/rain_map/rain_map_bindings.slangh"
#include "rtx/utility/math.slangh"

float2 rainMapLoad(int2 texel) {
  const int last = int(cb.wetnessArgs.rainMapResolution) - 1;
  return rainMapHeightInput[clamp(texel, int2(0, 0), int2(last, last))].xy;
}

bool rainMapLoadHeight(int2 texel, out float height) {
  height = rainMapLoad(texel).x;
  return height > cb.wetnessArgs.rainMapTraceBottom - 0.5f * max(cb.metersToWorldUnitScale, 0.0001f);
}

float rainMapCover(float2 value) {
  if (value.x <= cb.wetnessArgs.rainMapTraceBottom - 0.5f * max(cb.metersToWorldUnitScale, 0.0001f)) {
    return -1.0f;
  }
  const float minCoverHeight = max(cb.wetnessArgs.rainOcclusionMinHeight,
    0.01f * max(cb.metersToWorldUnitScale, 0.0001f));
  return value.y - value.x >= minCoverHeight ? 1.0f : 0.0f;
}

// An 8x8 workgroup shares the clamped cover samples in its feather footprint.
static const int kGroupSize = 8;
static const int kMaxRadius = 12;
static const int kMaxTileWidth = kGroupSize + 2 * kMaxRadius;
groupshared float s_cover[kMaxTileWidth * kMaxTileWidth];
groupshared uint s_hasGround;

[shader("compute")]
[numthreads(kGroupSize, kGroupSize, 1)]
void main(uint2 threadId : SV_DispatchThreadID, uint2 localId : SV_GroupThreadID, uint lane : SV_GroupIndex) {
  const WetnessArgs args = cb.wetnessArgs;
  const float units = max(cb.metersToWorldUnitScale, 0.0001f);
  const bool inBounds = all(threadId < args.rainMapResolution);
  const float2 center = rainMapLoad(int2(threadId));
  const float height = center.x;
  const bool missing = height <= args.rainMapTraceBottom - 0.5f * units;
  const int radius = clamp(int(round(args.rainOcclusionFeather / args.rainMapTexel)), 0, kMaxRadius);
  const int width = kGroupSize + 2 * radius;
  const int2 centerInTile = int2(localId) + radius;
  s_cover[centerInTile.y * width + centerInTile.x] = rainMapCover(center);

  if (radius > 0) {
    if (lane == 0) {
      s_hasGround = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    const bool waveHasGround = WaveActiveAnyTrue(inBounds && !missing);
    if (WaveIsFirstLane() && waveHasGround) {
      InterlockedOr(s_hasGround, 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    // A uniformly empty group needs no halo reads. All lanes help partial or edge groups.
    if (s_hasGround != 0) {
      const int2 tileOrigin = int2(threadId) - centerInTile;
      for (int i = int(lane); i < width * width; i += kGroupSize * kGroupSize) {
        const int2 p = int2(i % width, i / width);
        if (any(p < radius) || any(p >= radius + kGroupSize)) {
          s_cover[i] = rainMapCover(rainMapLoad(tileOrigin + p));
        }
      }
      GroupMemoryBarrierWithGroupSync();
    }
  }

  if (!inBounds) {
    return;
  }
  if (missing) {
    rainMapOutput[threadId] = float4(0.0f, args.rainMapTraceBottom - units, 0.0f, 0.0f);
    return;
  }

  // Two rings of eight neighbours measure relief at 1.5 and 3 meters.
  const int r1 = max(1, int(round(1.5f * units / args.rainMapTexel)));
  const int r2 = max(r1 + 1, int(round(3.0f * units / args.rainMapTexel)));
  const int d1 = max(1, int(round(float(r1) * 0.7071f)));
  const int d2 = max(d1 + 1, int(round(float(r2) * 0.7071f)));
  const int2 offsets[16] = {
    int2(r1, 0), int2(-r1, 0), int2(0, r1), int2(0, -r1), int2(d1, d1), int2(-d1, d1), int2(d1, -d1), int2(-d1, -d1),
    int2(r2, 0), int2(-r2, 0), int2(0, r2), int2(0, -r2), int2(d2, d2), int2(-d2, d2), int2(d2, -d2), int2(-d2, -d2)
  };
  float sum = 0.0f;
  float count = 0.0f;
  float sxx = 0.0f;
  float syy = 0.0f;
  float sxh = 0.0f;
  float syh = 0.0f;
  [unroll]
  for (int i = 0; i < 16; ++i) {
    float neighbour;
    if (rainMapLoadHeight(int2(threadId) + offsets[i], neighbour) && abs(neighbour - height) < units) {
      const float dh = neighbour - height;
      sum += dh;
      count += 1.0f;
      if (i < 8) {
        const float2 d = float2(offsets[i]) * args.rainMapTexel;
        sxx += d.x * d.x;
        syy += d.y * d.y;
        sxh += d.x * dh;
        syh += d.y * dh;
      }
    }
  }
  const float depression = count > 0.0f ? sum / count : 0.0f;
  const float gx = sxx > 0.0f ? sxh / sxx : 0.0f;
  const float gy = syy > 0.0f ? syh / syy : 0.0f;

  float coverSum = 0.0f;
  float coverWeight = 0.0f;
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const float covered = s_cover[(centerInTile.y + dy) * width + centerInTile.x + dx];
      if (covered >= 0.0f) {
        const float weight = float((radius + 1 - abs(dx)) * (radius + 1 - abs(dy)));
        coverSum += covered * weight;
        coverWeight += weight;
      }
    }
  }
  const float coverage = coverWeight > 0.0f ? coverSum / coverWeight : 0.0f;
  rainMapOutput[threadId] = float4(coverage, height, depression, length(float2(gx, gy)));
}
