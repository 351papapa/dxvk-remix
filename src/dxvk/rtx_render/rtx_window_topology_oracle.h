#pragma once

#include "rtx_window_topology.h"
#include "rtx_window_topology_reference.h"

namespace dxvk {
  inline std::vector<uint32_t> buildWindowIslandRootsOriginal(const WindowTopologyInput& in, WindowTopologyDiagnostics& diagnostics) {
    namespace oracle = window_topology_reference;
    oracle::WindowTopologyInput source;
    source.positions = {in.positions.data(), in.positions.size()};
    source.texcoords = {in.texcoords.data(), in.texcoords.size()};
    source.indices = {in.indices.data(), in.indices.size()};
    source.vertexCount = in.vertexCount;
    source.indexCount = in.indexCount;
    source.positionStride = in.positionStride;
    source.texcoordStride = in.texcoordStride;
    source.indexStride = in.indexStride;
    source.indexWidth = in.indexWidth;
    source.triangleCount = in.triangleCount;
    source.topology = static_cast<oracle::WindowPrimitiveTopology>(in.topology);
    oracle::WindowTopologyDiagnostics expected;
    auto roots = oracle::buildWindowIslandRoots(source, &expected);
    diagnostics = {static_cast<WindowTopologyStatus>(expected.status), expected.trianglesValid,
      expected.trianglesInvalid, expected.degenerateTriangles, expected.matchedEdges, expected.mergedTriangles,
      expected.ambiguousEdges, expected.rejectedFacePairs, expected.candidateComparisons,
      expected.positionTolerance, expected.uvTolerance};
    return roots;
  }
}
