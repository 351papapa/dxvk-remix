// SPDX-License-Identifier: MIT
// GPU selection for the local runtime's nearest-sphere sampling feature.
#include "rtx/utility/shader_types.h"
#include "rtx/concept/light/light.h"

#include "rtx/pass/rtxdi/local_sphere_select.h"
layout(binding = 0) StructuredBuffer<MemoryPolymorphicLight> SourceLights;
layout(binding = 1) StructuredBuffer<uint2> InputCandidates;
layout(binding = 2) RWStructuredBuffer<uint2> OutputCandidates;
layout(push_constant) ConstantBuffer<LocalSphereSelectArgs> args;
groupshared uint2 Candidates[512];

bool precedes(uint2 a, uint2 b) {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint tid : SV_GroupIndex) {
  for (uint local = tid; local < 512; local += 256) {
    uint index = group.x * 512 + local;
    uint2 candidate = uint2(0x7f800000u, 0xffffffffu);
    if (index < args.inputCount) {
      if (args.firstPass) {
        MemoryPolymorphicLight light = SourceLights[index];
        uint type = (light.data3.w >> 29) & 7u;
        // The sphere-light GPU contract stores its float position in data0.xyz.
        if (type == 0) {
          float3 delta = asfloat(light.data0.xyz) - args.cameraPosition;
          float distanceSquared = dot(delta, delta);
          if (isfinite(distanceSquared)) { candidate = uint2(asuint(max(0.1f, distanceSquared)), index << 3); }
        }
      } else {
        candidate = InputCandidates[index + 1];
      }
    }
    Candidates[local] = candidate;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint size = 2; size <= 512; size <<= 1) {
    for (uint step = size >> 1; step; step >>= 1) {
      for (uint local = tid; local < 512; local += 256) {
        uint other = local ^ step;
        if (other > local) {
          uint2 a = Candidates[local];
          uint2 b = Candidates[other];
          bool ascending = (local & size) == 0;
          if (ascending ? precedes(b, a) : precedes(a, b)) {
            Candidates[local] = b;
            Candidates[other] = a;
          }
        }
      }
      GroupMemoryBarrierWithGroupSync();
    }
  }
  if (tid < 64) { OutputCandidates[group.x * 64 + tid + 1] = Candidates[tid]; }
  if (group.x == 0 && tid == 0) {
    uint count = 0;
    if (args.groupCount == 1) {
      while (count < min(args.sampleCount, 64u) && Candidates[count].y != 0xffffffffu) { ++count; }
    }
    OutputCandidates[0] = uint2(count, 0);
  }
}
