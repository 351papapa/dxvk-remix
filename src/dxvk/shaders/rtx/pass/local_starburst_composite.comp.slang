//!variant local_starburst_composite.comp
//!end-variants
#include "rtx/pass/local_starbursts.h"

layout(push_constant) ConstantBuffer<LocalStarburstArgs> args;
layout(binding = STARBURST_SPRITES) StructuredBuffer<LocalStarburstSprite> sprites;
layout(binding = STARBURST_COLOR) RWTexture2D<float4> color;
layout(binding = STARBURST_MASK) Sampler2D starMask;

groupshared uint tileMask[8];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID, uint lane : SV_GroupIndex, uint2 group : SV_GroupID) {
  if (lane < 8) {
    tileMask[lane] = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  if (lane < args.count) {
    const LocalStarburstSprite sprite = sprites[lane];
    const float2 tileCenter = float2(group * 16) + 8.0f;
    if (sprite.tint.w > 0 && all(abs(tileCenter - sprite.centerHalfSize.xy) <= 8.0f + sprite.centerHalfSize.zw)) {
      InterlockedOr(tileMask[lane / 32], 1u << (lane % 32));
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (pixel.x >= args.width || pixel.y >= args.height) {
    return;
  }
  float3 sum = 0;
  for (uint word = 0; word < (args.count + 31) / 32; ++word) {
    uint bits = tileMask[word];
    while (bits) {
      const uint i = word * 32 + firstbitlow(bits);
      bits &= bits - 1;
      const LocalStarburstSprite s = sprites[i];
      const float2 uv = (float2(pixel) + 0.5f - s.centerHalfSize.xy) / (2.0f * s.centerHalfSize.zw) + 0.5f;
      if (all(uv >= 0) && all(uv <= 1)) {
        sum += max(starMask.SampleLevel(uv, 0).xyz, 0.0f) * s.tint.xyz;
      }
    }
  }
  if (any(sum > 0)) {
    const float4 original = color[pixel];
    color[pixel] = float4(original.xyz + sum, original.w);
  }
}
