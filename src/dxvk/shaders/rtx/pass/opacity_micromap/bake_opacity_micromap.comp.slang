/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "bake_opacity_micromap_binding_indices.h"

#include "bake_opacity_micromap_rtx_dependencies.slangh"
#include "bake_opacity_micromap_utils.slangh"

layout(binding = BINDING_BAKE_OPACITY_MICROMAP_CONSTANTS)
ConstantBuffer<BakeOpacityMicromapArgs> cb;

[allow("parameterBindingsOverlap")]
layout(set = BINDING_SET_BINDLESS_RAW_BUFFER, binding = 0) 
StructuredBuffer<float> Geometries[];

[allow("parameterBindingsOverlap")]
layout(set = BINDING_SET_BINDLESS_RAW_BUFFER, binding = 0)
StructuredBuffer<uint16_t> Indices[];

[allow("parameterBindingsOverlap")]
layout(set = BINDING_SET_BINDLESS_RAW_BUFFER, binding = 0)
StructuredBuffer<uint32_t> Indices32[];


layout(binding = BINDING_BAKE_OPACITY_MICROMAP_TEXCOORD_INPUT)
StructuredBuffer<float> Texcoords;

layout(binding = BINDING_BAKE_OPACITY_MICROMAP_OPACITY_INPUT)
Sampler2D Opacity;      

layout(binding = BINDING_BAKE_OPACITY_MICROMAP_SECONDARY_OPACITY_INPUT)
Sampler2D SecondaryOpacity;

layout(binding = BINDING_BAKE_OPACITY_MICROMAP_BINDING_SURFACE_DATA_INPUT) 
StructuredBuffer<Surface> Surfaces;

layout(binding = BINDING_BAKE_OPACITY_MICROMAP_ARRAY_OUTPUT)
RWStructuredBuffer<uint8_t> OpacityMicromapArray;

void loadVertexData(uint triIdx, Surface surface, out float2 vertexTexcoords[3], out float16_t vertexOpacities[3]) 
{
  for (uint i = 0; i < 3; i++) 
  {
    uint index;

    const uint indexIndex = (triIdx + cb.triangleOffset) * 3 + i + surface.firstIndex;

    if (surface.indexBufferIndex != BINDING_INDEX_INVALID)
    {
      const uint indexBufferIndex = surface.indexBufferIndex;

      if (surface.indexStride == 4)
      {
        index = BUFFER_ARRAY(Indices32, indexBufferIndex, indexIndex);
      }
      else
      {
        index = BUFFER_ARRAY(Indices, indexBufferIndex, indexIndex);
      }
    }
    else
    {
      index = indexIndex;
    }

    // Get texcoords
    const uint texcoordIndex = (cb.texcoordOffset + index * cb.texcoordStride) / 4;
    vertexTexcoords[i] = float2(Texcoords[texcoordIndex], Texcoords[texcoordIndex + 1]);

    // Apply texture transform (FF)
    vertexTexcoords[i] = mul(surface.textureTransform, float4(vertexTexcoords[i], 1.f, 1.f)).xy;

    // Get opacity
    vertexOpacities[i] = 1.0f;

    if (surface.color0BufferIndex != BINDING_INDEX_INVALID)
    {
      const uint color0BufferIndex = surface.color0BufferIndex;
      const uint color0ElementIndex = (surface.color0Offset + index * uint(surface.color0Stride)) / 4;

      // VK_FORMAT_B8G8R8A8_UNORM
      const uint colorBits = floatBitsToUint(BUFFER_ARRAY(Geometries, color0BufferIndex, color0ElementIndex));

      vertexOpacities[i] = unorm8ToF16(uint8_t(colorBits >> 24));
    }
  }
}

void applyVertexAndTextureOperationsToOpacity(inout float opacity, vec3 albedo, float3 bary, Surface surface, float16_t vertexOpacities[3])
{
  if (cb.isOpaqueMaterial)
  {
    float vertexInterpolatedOpacity = 1.0f;

    if (surface.color0BufferIndex != BINDING_INDEX_INVALID)
    {
      vertexInterpolatedOpacity = interpolateHitAttribute(vertexOpacities, bary); 
    }

    // Fixed function texture stage state for alpha channel
    float textureAlphaArg1, textureAlphaArg2;
    float tFactorAlpha = float((surface.tFactor >> 24) & 0xff) / 255.0f;
    chooseTextureArgument(textureAlphaArg1, surface.textureAlphaArg1Source, opacity, vertexInterpolatedOpacity, tFactorAlpha, opacity);
    chooseTextureArgument(textureAlphaArg2, surface.textureAlphaArg2Source, opacity, vertexInterpolatedOpacity, tFactorAlpha, float(1.0));
    chooseTextureOperationAlpha(opacity, surface.textureAlphaOperation, textureAlphaArg1, textureAlphaArg2);

    // Unused
    float emissiveBlendOverrideInfluence = float(0);

    // This must be enabled as the opacities are otherwise zeroed out by the handler
    const bool enableOpacityEmissiveBlendModes = true;

    calcOpaqueSurfaceMaterialOpacity(
      albedo, opacity, surface, opacity, emissiveBlendOverrideInfluence, enableOpacityEmissiveBlendModes);
  }
}

vec4 sampleColorOpacityTexture(Sampler2D textureSampler, float2 texcoords)
{
  return textureSampler.SampleLevel(texcoords, 0);
}

vec4 sampleColorOpacityRaw(float2 texcoords)
{
  vec4 colorOpacity = sampleColorOpacityTexture(Opacity, texcoords);

  if (cb.isRayPortalMaterial)
  {
    // TODO: match RT resolve transparency calculation for portals, handle animation
    // For now take the max opacity from both textures
    colorOpacity.a = max(colorOpacity.a, sampleColorOpacityTexture(SecondaryOpacity, texcoords).a);
  }
  
  return colorOpacity;
}

float16_t sampleOpacity(float2 texcoords, Surface surface, float3 bary, float16_t vertexOpacities[3])
{
  vec4 colorOpacity = sampleColorOpacityRaw(texcoords);
  float opacity = colorOpacity.a;
  
  if (cb.applyVertexAndTextureOperations)
  {
    applyVertexAndTextureOperationsToOpacity(opacity, colorOpacity.rgb, bary, surface, vertexOpacities);
  }

  return float16_t(opacity);
}

// The mean opacity of the taps taken for the current micro-triangle, which splits mixed (4-state "unknown")
// micro-triangles into UnknownOpaque (coverage >= 0.5) and UnknownTransparent. Rays that trace FORCE_OPAQUE see no
// difference between the two, because both hit and are alpha-tested. Rays that force 2-state OMM
// (rtx.foliageRays.fastRayMode = Omm2State) keep the ones at least 50% covered.
// It stays at 0.5 (UnknownOpaque) when no footprint could be sampled.
static float g_microTriangleCoverage = 0.5f;

// Samples opacity for a micro triangle with conservative estimation
// Samples a lod 0 texture quad that fully covers the uTriangle finding min and max opacities
void sampleOpacityConservative(Surface surface,
                              float2 microTriangleVertexTexcoords[3], 
                              float3 microTriangleVertexBarys[3],
                              float2 vertexTexcoords[3], 
                              float16_t vertexOpacities[3],
                              out float16_t minOpacity, 
                              out float16_t maxOpacity) 
{
  // Find texcoord bbox for the utriangle
  float2 texcoordsMin = calculateTriangleTexcoordBBoxMin(microTriangleVertexTexcoords);
  float2 texcoordsMax = calculateTriangleTexcoordBBoxMax(microTriangleVertexTexcoords);
  
  // Check for invalid values should garbage texcoord data be fed in
  if (!isValidValue(texcoordsMin) || !isValidValue(texcoordsMax))
  {
    minOpacity = 0;
    maxOpacity = 1;
    return;
  }
  
  // NV-DXVK start: [OmmBake] a texture extent outside [1, MAX] (zero, NaN, stale) cannot produce a
  // meaningful footprint; report unknown rather than derive a loop bound from it.
  if (!(cb.textureResolution.x >= 1.0f && cb.textureResolution.x <= float(BAKE_OPACITY_MICROMAP_MAX_TEXTURE_DIMENSION) &&
        cb.textureResolution.y >= 1.0f && cb.textureResolution.y <= float(BAKE_OPACITY_MICROMAP_MAX_TEXTURE_DIMENSION)))
  {
    minOpacity = 0;
    maxOpacity = 1;
    return;
  }
  // NV-DXVK end

  // Align bbox to actual texel centers that fully cover the bbox
  // Align with a top left pixel relative to the bbox min
  const float2 texcoordsIndexMin = floor(texcoordsMin * cb.textureResolution - 0.5);
  // Align with a bottom right pixel relative to the bbox max
  const float2 texcoordsIndexMax = floor(texcoordsMax * cb.textureResolution + 0.5);

  // Calculate number of texels in the given texcoord bbox
  // +1: include the end point of the bbox
  const float2 texelSampleDimsFloat = texcoordsIndexMax - texcoordsIndexMin + 1;
  texcoordsMin = (texcoordsIndexMin + 0.5) * cb.rcpTextureResolution;

  // NV-DXVK start: [OmmBake] NaN-safe budget check. The original test rejected with `dims > max`,
  // which is FALSE for NaN: finite texcoords around 1e35 overflow to +inf once scaled by the texture
  // extent, inf - inf = NaN, the NaN passed every `>` test, and uint(NaN) is undefined in SPIR-V (the
  // product check could then wrap, e.g. 0xFFFFFFFF^2 == 1). Every comparison below is written so NaN
  // fails it, and the tap budget itself is capped by a compile-time constant.
  const float maxTaps = float(min(cb.conservativeEstimationMaxTexelTapsPerMicroTriangle, uint(BAKE_OPACITY_MICROMAP_MAX_TEXEL_TAPS_HARD_CAP)));
  if (!(texelSampleDimsFloat.x >= 1.0f && texelSampleDimsFloat.x <= maxTaps &&
        texelSampleDimsFloat.y >= 1.0f && texelSampleDimsFloat.y <= maxTaps &&
        texelSampleDimsFloat.x * texelSampleDimsFloat.y <= maxTaps) ||
      !isValidValue(texcoordsMin))
  {
    // Sampling budget overflowed or the footprint is garbage, mark the microtriangle state as unknown/anyHit
    minOpacity = 0;
    maxOpacity = 1;
    return;
  }
  const uint2 texelSampleDims = min(uint2(texelSampleDimsFloat), uint2(BAKE_OPACITY_MICROMAP_MAX_TEXEL_TAPS_HARD_CAP, BAKE_OPACITY_MICROMAP_MAX_TEXEL_TAPS_HARD_CAP));
  // NV-DXVK end

  // Go through all texels within the texcoord bbox, sampling at their centers and calculating the min max opacity
  minOpacity = 1;
  maxOpacity = 0;
  uint numTaps = 0;
  float opacitySum = 0.0f; // coverage for the unknown-state split
  for (uint v = 0; v < texelSampleDims.y; v++)
  {
    for (uint u = 0; u < texelSampleDims.x; u++)
    {
      // NV-DXVK: [OmmBake] belt and braces - a hard stop independent of the bounds computed above
      if (++numTaps > BAKE_OPACITY_MICROMAP_MAX_TEXEL_TAPS_HARD_CAP)
      {
        minOpacity = 0;
        maxOpacity = 1;
        return;
      }
      const float2 texcoords = texcoordsMin + uint2(u, v) * cb.rcpTextureResolution;
      
      float3 bary;

      // Note if the texcoords is out of the triangle bounds, it'll retun bary for the texcoords snapped
      // to a triangle edge. This is better than not having barys and thus skipping vertex opacity blending 
      texcoordToBary(texcoords, vertexTexcoords, bary);

      const float16_t opacity = sampleOpacity(texcoords, surface, bary, vertexOpacities);
      minOpacity = min(opacity, minOpacity);
      maxOpacity = max(opacity, maxOpacity);
      opacitySum += float(opacity);
    }
  }
  g_microTriangleCoverage = opacitySum / float(max(numTaps, 1u));
}

enum OpacityState 
{
  // Values match those of VkOpacityMicromapSpecialIndexEXT
  Transparent = 0,   
  Opaque = 1,
  UnknownTransparent = 2,
  UnknownOpaque = 3, 
};

#define OMM_2_STATE_UNKNOWN_STATE OpacityState::Opaque
#define OMM_4_STATE_UNKNOWN_STATE OpacityState::UnknownTransparent

OpacityState getOpacityState(float opacity)
{
  if (opacity <= cb.resolveTransparencyThreshold)
  {
    return OpacityState::Transparent;
  }
  else if (opacity >= cb.resolveOpaquenessThreshold)
  {
    return OpacityState::Opaque;
  }
  else if (cb.is2StateOMMFormat)
  {
    return OMM_2_STATE_UNKNOWN_STATE;
  }
  else
  {
    return OMM_4_STATE_UNKNOWN_STATE;
  }
}

OpacityState calculateOpacityMicromapValue(float16_t minOpacity, float16_t maxOpacity)
{
  OpacityState minOpacityState = getOpacityState(minOpacity);
  OpacityState maxOpacityState = getOpacityState(maxOpacity);

  OpacityState mixedOpacityState = cb.is2StateOMMFormat ? OMM_2_STATE_UNKNOWN_STATE : OMM_4_STATE_UNKNOWN_STATE;

  const OpacityState state = minOpacityState == maxOpacityState ? minOpacityState : mixedOpacityState;

  // See g_microTriangleCoverage.
  if (state == OpacityState::UnknownTransparent && g_microTriangleCoverage >= 0.5f)
  {
    return OpacityState::UnknownOpaque;
  }

  return state;
}

OpacityState calculateOpacityMicromapValue(uint uIdx, float rcpNumSubdivisions, Surface surface, float2 vertexTexcoords[3], float16_t vertexOpacities[3])
{
  float2 microTriangleCenterTexcoord;
  float2 microTriangleVertexTexcoords[3];
  float3 microTriangleCenterBary;
  float3 microTriangleVertexBarys[3];
  
  micromeshDistToTexcoord(uIdx, cb.subdivisionLevel, rcpNumSubdivisions, vertexTexcoords,
                          microTriangleCenterTexcoord, microTriangleVertexTexcoords, 
                          microTriangleCenterBary, microTriangleVertexBarys);
                          
  float16_t maxOpacity;
  float16_t minOpacity;
  g_microTriangleCoverage = 0.5f;

  if (cb.useConservativeEstimation)
  {
    sampleOpacityConservative(surface, 
                              microTriangleVertexTexcoords, microTriangleVertexBarys,
                              vertexTexcoords, vertexOpacities,
                              minOpacity, maxOpacity); 
  }
  else 
  {
    // Center tap with LOD given microtriangle diameter
    float16_t opacity = sampleOpacity(microTriangleCenterTexcoord, surface, microTriangleCenterBary, vertexOpacities);
  
    maxOpacity = opacity;
    minOpacity = opacity;
    float opacitySum = float(opacity);

    // Taps at corners of the microtriangle at lod 0
    for (uint i = 0; i < 3; i++)
    {
      opacity = sampleOpacity(microTriangleVertexTexcoords[i], surface, microTriangleVertexBarys[i], vertexOpacities);

      minOpacity = min(opacity, minOpacity);
      maxOpacity = max(opacity, maxOpacity);
      opacitySum += float(opacity);
    }
    g_microTriangleCoverage = opacitySum * 0.25f;
  }

  return calculateOpacityMicromapValue(minOpacity, maxOpacity);
}

[shader("compute")]
[numthreads(BAKE_OPACITY_MICROMAP_NUM_THREAD_PER_COMPUTE_BLOCK, 1, 1)]
void main(uint globalId: SV_DispatchThreadID, uint groupId : SV_GroupID, uint threadId : SV_GroupThreadID)
{
  if (globalId >= cb.numActiveThreads)
  {
    return;
  }

  // NV-DXVK start: [OmmBake] reject constants that would make the index math below meaningless
  // (division by zero, shifts past 31). The CPU validates the same before dispatching.
  if (cb.numMicroTrianglesPerTriangle == 0 || cb.subdivisionLevel > 12)
  {
    return;
  }
  // NV-DXVK end

  const uint numBitsPerMicroTriangle = cb.is2StateOMMFormat ? 1 : 2;
  const uint numSubdivisionsPerEdge = 1 << cb.subdivisionLevel;
  const float rcpNumSubdivisions = 1.f / numSubdivisionsPerEdge;

  // log2(number of uTri per 32-bit output word), so 5 (32 tris) when each uTri output 1 bit, or 4 (16 tris) when each uTri outputs 2 bits
  // NV-DXVK: [OmmBake] capped by a compile-time constant (one output byte holds at most 8 uTris)
  const uint numTrisPerThread = min(cb.numMicroTrianglesPerThread, uint(BAKE_OPACITY_MICROMAP_MAX_MICRO_TRIANGLES_PER_THREAD));
  
  const uint globalThreadlIdx = cb.threadIndexOffset + globalId;
  const uint triIdx = globalThreadlIdx * numTrisPerThread / cb.numMicroTrianglesPerTriangle;
  
  float2 vertexTexcoords[3];
  float16_t vertexOpacities[3];
  loadVertexData(triIdx, cb.surface, vertexTexcoords, vertexOpacities);
  
  uint8_t compactResult = 0;
  for (uint i=0; i<numTrisPerThread; i++)
  {
    const uint sharedResultOffset = (i << (numBitsPerMicroTriangle - 1));
  
    const uint uIdx = (globalThreadlIdx * numTrisPerThread + i) - triIdx * cb.numMicroTrianglesPerTriangle;

    // Calculate opacity micromap value for the u-triangle processed by this thread
    const uint8_t ommValue = (uint8_t)calculateOpacityMicromapValue(uIdx, rcpNumSubdivisions, cb.surface, vertexTexcoords, vertexOpacities);

    compactResult |= ommValue << sharedResultOffset;
  }
  OpacityMicromapArray[cb.threadIndexOffset + globalId] = compactResult;
}
