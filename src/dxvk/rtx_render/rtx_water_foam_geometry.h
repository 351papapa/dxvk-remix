#pragma once
// GPU-generated foam surface geometry.
//
// Purpose: give intersection foam the same lighting as the GPU wake foam by
// making it real geometry with a real material, instead of colour written into the
// final output by a composite pass. A composite pass cannot receive the scene's
// shadows or indirect light, so any foam drawn that way has to invent its own
// illumination.
//
// Why this header exists at all: the public remixapi mesh path
// (remixapi_CreateMesh -> buildExternalMeshSurfacesFromOwned in rtx_remix_api.cpp)
// allocates HOST_VISIBLE buffers and memcpys from CPU pointers. That makes it
// impossible to feed vertices that were produced on the GPU, and reading a GPU
// field back every frame to rebuild a CPU mesh is both a stall and exactly the CPU
// cost the user asked to remove. Nothing about RasterGeometry actually requires
// host-visible memory though - it holds DxvkBufferSlices - so the internal path
// below builds the same geometry shape over DEVICE_LOCAL buffers that compute
// writes directly. No public API is extended or changed.
//
// Timing contract, which is the part that is easy to get wrong:
//   1. foam vertices are written by compute into a persistent vertex buffer;
//   2. the acceleration structure is built/updated afterwards and reads that buffer;
//   3. ray tracing and material shading then read both.
// The vertex buffer therefore must carry
// VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR and
// VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, and the compute write must be
// barriered before the AS build. Geometry reads the completed density snapshot
// and current generated ocean positions; the late update uses this frame's TLAS. It never
// queries an acceleration structure while generating its own geometry.

#include "rtx_types.h"
#include "rtx_common_object.h"
#include "rtx_water_foam_sheet_contract.h"
#include "../../util/sync/sync_signal.h"
#include <array>
#include <atomic>
#include <memory>
#include <cstdint>

namespace dxvk {

// Layout and topology constants live in rtx_water_foam_sheet_contract.h so they can
// be unit tested without linking the engine. The static_asserts that pin them are
// there, next to the definitions, rather than duplicated here.

// Device-local persistent geometry for the foam sheet. Created once, written by
// compute every frame; optional diagnostics perform low-frequency fence-gated readback.
class WaterFoamGpuGeometry : public CommonDeviceObject {
public:
  explicit WaterFoamGpuGeometry(DxvkDevice* device) : CommonDeviceObject(device) {}

  // Parameters captured once per frame by the caller and then reused for the dispatch
  // and the snapshot, so the vertices the GPU writes and the box the acceleration
  // structure gets cannot disagree about the up axis or the grid placement.
  struct FrameParams {
    bool zUp = true;
    bool frustumCulling = false;
    Matrix4 worldToClip;
    // World-space origin of the density field, in the two in-plane axes, and the world
    // size of one density cell. These come from the foam update pass, which owns the
    // field placement; the display surface is laid over exactly that domain so the
    // foam line sits on its own density rather than near it.
    float gridOriginX = 0.0f;
    float gridOriginY = 0.0f;
    float gridSpacing = 0.0f;
    // User foam strength, applied as a coverage scale.
    float coverageScale = 1.0f;
    // Lift above the recorded water surface, to avoid depth fighting.
    float surfaceOffset = 0.0f;
    // Seconds since start, for erosion and UV drift. Passed explicitly because this pass
    // runs before the frame constant buffer exists.
    float time = 0.0f;
    // World-space repeat distance, shared with the screen composite.
    float textureScale = 5.0f;
    // Water height range the density pass produced, so the bounding box is derived from
    // real water instead of a fixed band that may not contain it. A half range of zero
    // means "not known yet" and falls back to the contact probe's ray clamp.
    float waterHeightCentre = 0.0f;
    float waterHeightHalfRange = 0.0f;
    // Camera height of the density snapshot. The generator uses it as the anchor for a
    // collapsed triangle when no water is nearby, so a degenerate triangle still lands on
    // a finite, in-domain position instead of an invented one.
    float cameraUp = 0.0f;
  };

  // Allocates the persistent vertex and index buffers. Safe to call repeatedly.
  void ensureResources();
  static void showImguiSettings();

  // Fills the index buffer once (fixed topology). Returns true on success.
  bool ensureTopology(const Rc<DxvkContext>& ctx);

  // Dispatches the surface generator into the persistent vertex buffer.
  //   densityField  R16G16B16A16 water foam density, flow xy and terrain wetness
  //   heightField   R32G32B32A32 water height/validity and terrain height/validity
  // Both are read on the GPU; nothing is read back to the CPU. They are separate
  // resources because the density field is half precision and cannot hold a world height
  // without quantising it to half or whole units, which is coarser than the surface offset.
  void dispatchSheetUpdate(const Rc<DxvkContext>& ctx, const FrameParams& params,
                           const Resources::Resource& densityField,
                           const Resources::Resource& heightField,
                           const Rc<DxvkBuffer>& waterInput, const Rc<DxvkBuffer>& waterPositions,
                           const Rc<DxvkBuffer>& waterDepth, const Rc<DxvkBuffer>& waterVertices);

  bool ready() const { return m_vertexBuffer != nullptr && m_previousVertexBuffer != nullptr && m_indexBuffer != nullptr && m_topologyReady; }
  bool ownsGeometry(const RasterGeometry& geometry) const;
  bool ownsVertexBuffer(const Rc<DxvkBuffer>& buffer) const;
  uint32_t vertexCount() const { return m_vertexCount; }
  uint32_t indexCount() const { return m_indexCount; }

  // Diagnostics: counts of what was actually produced, for the debug views and for
  // the low-frequency log. Never read back per frame from the field itself.
  struct Stats {
    uint32_t verticesWritten = 0;
    uint32_t indicesWritten = 0;
    bool dispatchedThisFrame = false;
  };

  // World bounds of the current density domain and observed water-height range.
  static AxisAlignedBoundingBox placedSheetBox(const FrameParams& params);

  const Stats& stats() const { return m_stats; }

  // Content generation. Bumped once per submission whose vertex contents changed.
  // The mesh handle, topology, layout and material identity stay fixed; only the
  // VertexPosition content hash carries this value, which is what moves the object
  // onto the BVH-update path instead of the reuse path without changing its
  // topological bucket.
  uint64_t contentGeneration() const { return m_contentGeneration; }

  // Immutable per-frame metadata snapshot. References the SAME persistent GPU buffers;
  // it exists so the BlasEntry keeps its old snapshot alive rather than being mutated
  // through a const_cast.
  //
  // params must be the SAME FrameParams handed to dispatchSheetUpdate for this frame.
  // The bounding box is derived from them, so a snapshot built from different
  // parameters than the dispatch would describe a sheet the GPU never produced. An
  // earlier version of this function omitted the bounding box entirely, which meant the
  // corrected box maths never reached anything the acceleration structure could see.
  std::shared_ptr<const RasterGeometry> buildGeometrySnapshot(remixapi_MaterialHandle material,
                                                              const FrameParams& params, uint64_t waterTopologyHash);

  // ---- GPU source-vertex readback (low frequency, fence gated) ----
  //
  // Copies three vertices (first, middle, last) out of the generated vertex buffer into
  // a small host-visible staging slot, then signals a fence once those commands retire.
  // A later frame polls fence->value() and only then maps and reads the slot, so this
  // never waits, sleeps or calls waitIdle on the render thread.
  //
  // Samples check finite attributes and unit normals. They do not establish whole-mesh
  // correctness or prove that the geometry caches, BVH and shading consumed the data.
  struct ReadbackSample {
    uint32_t slot = 0;
    uint64_t generation = 0;
    uint64_t frame = 0;
    bool zUp = true;
    float translation[3] = { 0.0f, 0.0f, 0.0f };
    // Read back verbatim from the GPU buffer.
    float position[3][3] = {};
    float normal[3][3] = {};
    float texcoord[3][2] = {};
    uint32_t color[3] = {};
    bool valid = false;
  };

  // Issues a sample if a slot is free. Never blocks; returns false when it skipped.
  bool tryIssueReadback(const Rc<DxvkContext>& ctx, const FrameParams& params, uint64_t frameIndex);

  // Returns at most one completed sample (oldest first) and clears it. Never waits.
  bool tryConsumeReadback(ReadbackSample& out);

  // Number of slots currently awaiting completion, for diagnostics.
  uint32_t pendingReadbackCount() const;

private:
  void validateTrianglePairReuse(const Rc<DxvkContext>& ctx, const FrameParams& params,
                                const WaterFoamSurfaceArgs& args, bool liveWater,
                                bool selectedReuse, uint32_t groups);
  // Set false only after the fence callback finishes reading/logging. No this capture
  // in the callback, and no more than one full comparison can be outstanding.
  std::shared_ptr<std::atomic<bool>> m_pairReuseValidationPending;
  uint64_t m_pairReuseValidationLastFrame = ~uint64_t(0);

  struct ReadbackSlot {
    Rc<DxvkBuffer> buffer;
    bool pending = false;
    uint64_t targetValue = 0;
    ReadbackSample sample {};
  };
  static constexpr uint32_t kReadbackSlotCount = 3;
  // Byte offsets of the sampled vertices in the interleaved buffer, and the three
  // per-slot regions they land in.
  static constexpr uint32_t kReadbackVertexIndices[3] = { 0u, kWaterFoamSheetVertexCount / 2u, kWaterFoamSheetVertexCount - 1u };

  Rc<DxvkBuffer> m_vertexBuffer;
  Rc<DxvkBuffer> m_previousVertexBuffer;
  Rc<DxvkBuffer> m_indexBuffer;
  Rc<DxvkBuffer> m_indexStaging;
  uint32_t m_vertexCount = 0;
  uint32_t m_indexCount = 0;
  bool m_topologyReady = false;
  uint64_t m_contentGeneration = 0;
  Stats m_stats {};

  Rc<sync::Fence> m_readbackFence = new sync::Fence {};
  uint64_t m_readbackSignalValue = 0;
  uint32_t m_readbackMapFailures = 0;
  std::array<ReadbackSlot, kReadbackSlotCount> m_readbackSlots {};
};

} // namespace dxvk
