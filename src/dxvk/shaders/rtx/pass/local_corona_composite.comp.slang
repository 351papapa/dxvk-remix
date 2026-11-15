//!variant local_corona_composite.comp
//!end-variants
#include "rtx/pass/local_coronas.h"

layout(push_constant) ConstantBuffer<LocalCoronaArgs> args;
layout(binding = CORONA_SPRITES) StructuredBuffer<LocalCoronaSprite> sprites;
layout(binding = CORONA_COLOR) RWTexture2D<float4> color;

groupshared uint tileMask[CORONA_SOURCE_LIMIT / 32];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID, uint lane : SV_GroupIndex, uint2 group : SV_GroupID) {
  if (lane < CORONA_SOURCE_LIMIT / 32) {
    tileMask[lane] = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint index = lane; index < args.count; index += 256) {
    const LocalCoronaSprite sprite = sprites[index];
    const float2 tileCenter = float2(group * 16) + 8.0f;
    if (sprite.tint.w > 0 && all(abs(tileCenter - sprite.centerHalfSize.xy) <= 8.0f + sprite.centerHalfSize.zw)) {
      InterlockedOr(tileMask[index / 32], 1u << (index % 32));
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
      const LocalCoronaSprite s = sprites[i];
      const float2 offset = (float2(pixel) + 0.5f - s.centerHalfSize.xy) / s.centerHalfSize.zw;
      const float radiusSquared = dot(offset, offset);
      if (radiusSquared < 1) {
        // Radial core reaches exactly zero at the original board's boundary.
        const float edge = exp2(-s.tint.w);
        const float profile = saturate((exp2(-s.tint.w * radiusSquared) - edge) / (1 - edge));
        sum += profile * s.tint.xyz;
      }
    }
  }
  if (any(sum > 0)) {
    const float4 original = color[pixel];
    color[pixel] = float4(original.xyz + sum, original.w);
  }
}
