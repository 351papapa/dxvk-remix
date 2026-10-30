//!variant point_instancer_culling.comp
//!variant point_instancer_culling_frustum.comp
//!>       ENABLE_FRUSTUM_CULLING=1
//!end-variants

#ifndef ENABLE_FRUSTUM_CULLING
#define ENABLE_FRUSTUM_CULLING 0
#endif

/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// GPU compute shader for radius-based culling of USD PointInstancer instances.
//
// Every thread writes a full VkAccelerationStructureInstanceKHR entry into the
// TLAS instance buffer unconditionally.  Visible instances get the real mask;
// culled instances get mask=0 so the TLAS builder ignores them.  This avoids
// any divergent early-outs and guarantees no stale data in the buffer.
//
// Each PointInstancer instance gets its own surface AND material entry so that it
// can have a unique surface/material ID in the customInstanceIndex field.  The CPU
// reserves N contiguous surface and material slots and populates only the first
// (the "template").  This shader copies the template data to each instance's slot,
// patches per-instance objectToWorld/prevObjectToWorld/normalObjectToWorld transforms
// in the surface buffer, and duplicates the material entry (a plain copy).

#include "rtx/pass/instance_culling/point_instancer_culling_bindings.slangh"

// Size of VkAccelerationStructureInstanceKHR in bytes
static const uint kVkAccelInstanceSize = 64;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (threadId.x >= cb.threadCount) {
    return;
  }

  // All batches of the frame share one thread list. Look up this thread's batch, then its index in the batch.
  const uint globalThread = cb.firstThread + threadId.x;
  const PointInstancerCullingBatch batch = batches[threadBatch[globalThread]];
  const uint instanceIdx = globalThread - batch.firstThread;

  // Read the instance-to-object transform (column-major float4x4)
  float4x4 instanceToObject = inputTransforms[batch.firstTransform + instanceIdx];

  // Compute the combined object-to-world * instance-to-object transform.
  // The translation column (col 3) gives us the world-space origin directly.
  // Note: In HLSL/Slang, matrix[i] returns ROW i, not column i.
  // Translation lives in column 3, accessed as (row0.w, row1.w, row2.w).
  float4x4 fullTransform = mul(batch.objectToWorld, instanceToObject);
  float3 worldPos = float3(fullTransform[0].w, fullTransform[1].w, fullTransform[2].w);

  // Squared-distance cull (avoid sqrt in the common rejection case)
  float3 delta    = worldPos - cb.cameraPosition;
  float  distSq   = dot(delta, delta);
  float  radiusSq = cb.cullingRadius * cb.cullingRadius;

  // Host (Remix API) instancing is not culled unless rtx.pointInstancer.cullHostInstancers is on.
  const bool cullable = batch.cullable != 0;
  bool visible = !cullable || (distSq <= radiusSq);

#if ENABLE_FRUSTUM_CULLING
  // Nearby instances keep their off-screen contribution. Rejected instances still
  // write a complete mask-zero TLAS entry below.
  if (cullable && visible && cb.frustumCull && distSq > cb.frustumNearSkip * cb.frustumNearSkip) {
    for (uint plane = 0; plane < 4; ++plane) {
      if (dot(cb.viewPlanes[plane].xyz, worldPos) + cb.viewPlanes[plane].w < -cb.frustumSpherePad) {
        visible = false;
        break;
      }
    }
  }

#endif

  // Optional density fade: stochastically reject instances in the fade region.
  if (cullable && visible && cb.fadeStartRadius > 0.0 && distSq > cb.fadeStartRadius * cb.fadeStartRadius && cb.cullingRadius > cb.fadeStartRadius) {
    float dist       = sqrt(distSq);
    float fadeRange  = cb.cullingRadius - cb.fadeStartRadius;
    float fadeFactor = (dist - cb.fadeStartRadius) / fadeRange; // 0..1

    // Deterministic hash-based rejection (stable per instance, no flickering)
    uint  hash   = instanceIdx * 2654435761u; // Knuth multiplicative hash
    float rand01 = float(hash & 0xFFFFu) / 65535.0;

    if (rand01 < fadeFactor) {
      visible = false;
    }
  }

  uint mask = visible ? batch.instanceMask : 0;

  // === Per-instance surface data: copy template and patch transforms ===

  // Each instance gets its own surface index: baseSurfaceIndex + instanceIdx.
  // Copy the template surface, then patch per-instance transforms using
  // the Surface struct properties so that any future layout changes are
  // reflected here automatically.
  // The SSS twin of a batch shares the primary's surface slots; the primary
  // writes them (same values), so the twin only writes its TLAS entries.
  if (batch.writeSurfaceData != 0) {
    Surface surface = surfaceBuffer[batch.baseSurfaceIndex];

    float4x4 prevFull = mul(batch.prevObjectToWorld, instanceToObject);
    float3x3 normalMat = transpose(inverse((float3x3)fullTransform));

    surface.objectToWorld       = (mat4x3)fullTransform;
    surface.prevObjectToWorld   = (mat4x3)prevFull;
    surface.normalObjectToWorld = normalMat;

    surfaceBuffer[batch.baseSurfaceIndex + instanceIdx] = surface;

    // === Per-instance material data: copy template ===
    // Material data is identical for all instances of a PointInstancer;
    // this is a plain struct copy from the template slot.
    materialBuffer[batch.baseSurfaceIndex + instanceIdx] = materialBuffer[batch.baseSurfaceIndex];
  }

  // === Write VkAccelerationStructureInstanceKHR unconditionally ===

  // VkTransformMatrixKHR is a 3x4 ROW-major matrix.
  // In HLSL/Slang, matrix[i] returns row i, so we write each row directly:
  //   Row 0 = fullTransform[0] = (r00, r01, r02, tx)
  //   Row 1 = fullTransform[1] = (r10, r11, r12, ty)
  //   Row 2 = fullTransform[2] = (r20, r21, r22, tz)
  uint byteOffset = batch.instanceBufferOffset + instanceIdx * kVkAccelInstanceSize;

  // Row 0
  instanceBuffer.Store(byteOffset +  0, asuint(fullTransform[0].x));
  instanceBuffer.Store(byteOffset +  4, asuint(fullTransform[0].y));
  instanceBuffer.Store(byteOffset +  8, asuint(fullTransform[0].z));
  instanceBuffer.Store(byteOffset + 12, asuint(fullTransform[0].w));
  // Row 1
  instanceBuffer.Store(byteOffset + 16, asuint(fullTransform[1].x));
  instanceBuffer.Store(byteOffset + 20, asuint(fullTransform[1].y));
  instanceBuffer.Store(byteOffset + 24, asuint(fullTransform[1].z));
  instanceBuffer.Store(byteOffset + 28, asuint(fullTransform[1].w));
  // Row 2
  instanceBuffer.Store(byteOffset + 32, asuint(fullTransform[2].x));
  instanceBuffer.Store(byteOffset + 36, asuint(fullTransform[2].y));
  instanceBuffer.Store(byteOffset + 40, asuint(fullTransform[2].z));
  instanceBuffer.Store(byteOffset + 44, asuint(fullTransform[2].w));

  // instanceCustomIndex:24 | mask:8 — each instance gets its own surface index
  uint perInstanceSurfaceIndex = batch.baseSurfaceIndex + instanceIdx;
  uint customIndexAndMask = (batch.customIndexFlags | (perInstanceSurfaceIndex & CUSTOM_INDEX_SURFACE_MASK)) | (mask << 24);
  instanceBuffer.Store(byteOffset + 48, customIndexAndMask);

  // instanceShaderBindingTableRecordOffset:24 | flags:8
  instanceBuffer.Store(byteOffset + 52, batch.sbtOffsetAndFlags);

  // accelerationStructureReference (uint64LE)
  instanceBuffer.Store(byteOffset + 56, batch.blasRefLo);
  instanceBuffer.Store(byteOffset + 60, batch.blasRefHi);
}
