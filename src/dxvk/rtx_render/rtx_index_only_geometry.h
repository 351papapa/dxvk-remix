/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once

namespace dxvk::index_only_geometry {

template<typename Buffer>
bool sameAttribute(const Buffer& a, const Buffer& b) {
  if (a.defined() != b.defined()) {
    return false;
  }
  return !a.defined() ||
    (a.buffer().ptr() == b.buffer().ptr() && a.offset() == b.offset() && a.length() == b.length() &&
     a.offsetFromSlice() == b.offsetFromSlice() && a.stride() == b.stride() && a.vertexFormat() == b.vertexFormat());
}

// A hash of positions alone does not cover normals or colors. This nonzero
// producer contract covers the complete immutable vertex bytes; exact backing
// resources and every attribute range/layout must agree as well.
template<typename Geometry>
bool sameImmutableVertexData(const Geometry& now, const Geometry& last) {
  return now.immutableVertexDataId && now.immutableVertexDataId == last.immutableVertexDataId &&
    now.vertexCount && now.vertexCount == last.vertexCount && now.positionBuffer.defined() &&
    sameAttribute(now.positionBuffer, last.positionBuffer) &&
    sameAttribute(now.normalBuffer, last.normalBuffer) &&
    sameAttribute(now.texcoordBuffer, last.texcoordBuffer) &&
    sameAttribute(now.color0Buffer, last.color0Buffer) &&
    sameAttribute(now.blendWeightBuffer, last.blendWeightBuffer) &&
    sameAttribute(now.blendIndicesBuffer, last.blendIndicesBuffer);
}

template<typename Geometry>
void preserveUnchangedPositions(Geometry& geometry, bool verticesUpdatedThisFrame) {
  if (!verticesUpdatedThisFrame || !geometry.previousPositionBuffer.defined()) {
    // The previous frame used these exact positions. Alias them instead of
    // copying the whole VB. A subsequent real vertex write must detach [0]
    // from [1], including a second update in this same frame.
    geometry.historyBuffer[1] = geometry.historyBuffer[0];
    geometry.previousPositionBuffer = geometry.positionBuffer;
  }
}

template<typename Buffer>
bool needsDetachedVertexWrite(const Buffer& current, const Buffer& previous, bool aliasable) {
  return current == nullptr || aliasable || current == previous;
}

}
