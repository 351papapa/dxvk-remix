#include "rtx/pass/grass/grass_args.h"
#include "rtx/pass/instance_definitions.h"
#include "rtx/concept/surface/surface.h"
#include "rtx/concept/surface_material/surface_material.h"

layout(binding = GRASS_CONSTANTS) ConstantBuffer<GrassConstants> cb;
layout(binding = GRASS_SELECTED) StructuredBuffer<GrassSelection> selected;
layout(binding = GRASS_INSTANCES) RWByteAddressBuffer instances;
layout(binding = GRASS_SURFACES) RWStructuredBuffer<Surface> surfaces;
layout(binding = GRASS_MATERIALS) RWStructuredBuffer<MemoryPolymorphicSurfaceMaterial> materials;
layout(binding = GRASS_MAPPING) RWStructuredBuffer<uint> mapping;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint slot = tid.x;
  if (slot >= cb.budget) { return; }
  GrassSelection root = selected[slot];
  // Keep inactive transforms nonsingular and their BLAS references valid.
  float4x4 transform = cb.prototypeTransform;
  if (root.valid != 0) { transform = mul(root.transform, cb.prototypeTransform); }
  uint surfaceIndex = cb.outputSurface + slot;
  if (root.valid != 0) {
    Surface surface = surfaces[cb.templateSurface];
    float4x4 previousTransform = mul(root.previousTransform, cb.prototypeTransform);
    surface.objectToWorld = (mat4x3)transform;
    surface.prevObjectToWorld = (mat4x3)previousTransform;
    surface.isStatic = all(transform[0] == previousTransform[0]) && all(transform[1] == previousTransform[1]) && all(transform[2] == previousTransform[2]);
    surface.normalObjectToWorld = transpose(inverse((float3x3)transform));
    surfaces[surfaceIndex] = surface;
    materials[surfaceIndex] = materials[cb.templateSurface];
    if (cb.previousSurface != GRASS_INVALID && root.previousSlot < cb.previousInstanceCount) {
      mapping[cb.previousSurface + root.previousSlot] = surfaceIndex;
    }
  }
  uint offset = cb.instanceOffset + slot * 64;
  instances.Store4(offset, asuint(transform[0]));
  instances.Store4(offset + 16, asuint(transform[1]));
  instances.Store4(offset + 32, asuint(transform[2]));
  uint mask = root.valid != 0 ? cb.instanceMask : 0;
  instances.Store(offset + 48, cb.customIndexFlags | (surfaceIndex & CUSTOM_INDEX_SURFACE_MASK) | (mask << 24));
  instances.Store(offset + 52, cb.sbtOffsetAndFlags);
  instances.Store2(offset + 56, uint2(cb.blasLo, cb.blasHi));
}
