/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
// CPU tests of the policy used by Compat and RetainedScene, not a GPU visibility test.
#define REMIX_ALLOW_X86
#include <remix/remix_retained_validation.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include "../../../src/dxvk/rtx_render/rtx_retained_compat_policy.h"
#include "../../../src/util/util_matrix.h"
#include "../../../src/dxvk/rtx_render/rtx_retained_rigid_parts.h"

namespace {
unsigned checks = 0;
void require(bool condition, const char* message) {
  ++checks;
  if (!condition) { std::printf("FAILED %u: %s\n", checks, message); std::exit(1); }
}
remixapi_RetainedInstance instance() {
  remixapi_RetainedInstance result {};
  result.id = 1; result.mesh = 2;
  for (unsigned i = 0; i < 3; ++i) result.transform.matrix[i][i] = 1.f;
  return result;
}
void testProtocol() {
  using namespace remixapi_retained;
  require(supportedVersion(2) && supportedVersion(3) && !supportedVersion(1) && !supportedVersion(4), "only v2/v3");
  auto record = instance();
  require(valid(record, 2) && valid(record, 3), "ordinary v2/v3 record");
  record.flags = REMIXAPI_RETAINED_HIDDEN;
  require(!valid(record, 2) && !valid(record) && valid(record, 3), "v2 always rejects HIDDEN, even after a v3 query");
  record.flags |= REMIXAPI_RETAINED_ORDERED;
  require(!valid(record, 2) && !valid(record, 3), "ORDERED/HIDDEN mutually exclusive");
  record.flags = REMIXAPI_RETAINED_ORDERED;
  require(valid(record, 2) && valid(record, 3), "ordered layout unchanged");
  record.flags = 4;
  require(!valid(record, 2) && !valid(record, 3), "unknown flag");
  record = instance(); record.transform.matrix[1][2] = std::numeric_limits<float>::quiet_NaN();
  require(!valid(record, 3), "hidden support does not bypass transform validation");
  require(std::strcmp(clientQueryKey(clientVisibilityKey), bridgeVisibilityKey) == 0, "client key requires matching bridge");
  require(std::strcmp(clientQueryKey(runtimeVisibilityKey), runtimeVisibilityKey) == 0, "native key remains native");
  require(std::strcmp(clientQueryKey(clientDeltaKey), bridgeDeltaKey) == 0 &&
          std::strcmp(bridgeDeltaKey, runtimeDeltaKey) != 0, "delta capability requires the matching server");
  uint64_t ids[] = { UINT64_MAX, 7, 7 };
  require(validVisibilityBatch(1, 0, 3, ids) && validVisibilityBatch(1, 1, 3, ids), "full-width and repeated IDs are valid");
  require(!validVisibilityBatch(0, 1, 3, ids) && !validVisibilityBatch(1, 2, 3, ids), "scope and boolean validation");
  require(!validVisibilityBatch(1, 0, 0, ids) && !validVisibilityBatch(1, 1, 4097, ids) &&
          !validVisibilityBatch(1, 1, 3, nullptr), "count and storage validation");
  ids[2] = 0;
  require(!validVisibilityBatch(1, 1, 3, ids), "reject a zero ID at the end of the batch");
  require(std::strcmp(clientVisibilityKey, runtimeVisibilityKey) && std::strcmp(bridgeVisibilityKey, runtimeVisibilityKey),
          "old bridge forwarding unknown key cannot obtain native capability");
  record = instance(); require(visibleUnordered(record), "visible unordered counted");
  record.flags = REMIXAPI_RETAINED_HIDDEN; require(!visibleUnordered(record), "hidden excluded from submitted count");
  record.flags = REMIXAPI_RETAINED_ORDERED; require(!visibleUnordered(record), "ordered counted only at DrawOrdered");
}
void testClassification() {
  using namespace remixapi_retained;
  const auto original = instance(); auto value = original;
  require(sameDescription(original, value) && sameTransform(original, value), "unchanged");
  value.transform.matrix[0][3] = 20;
  require(sameDescription(original, value) && !sameTransform(original, value), "translation only is fast");
  value.flags = REMIXAPI_RETAINED_HIDDEN;
  require(sameDescription(original, value), "transform plus visibility only is fast");
  value = original; value.mesh++; require(!sameDescription(original, value), "mesh replacement is full build");
  value = original; value.id++; require(!sameDescription(original, value), "identity is immutable");
  value = original; value.categoryFlags++; require(!sameDescription(original, value), "category re-resolved");
  value = original; value.doubleSided++; require(!sameDescription(original, value), "cull re-resolved");
  value = original; value.wetTag++; require(!sameDescription(original, value), "wetness re-resolved");
  value = original; value.objectPickingValue++; require(!sameDescription(original, value), "picking re-resolved");
  value = original; value.useBlend++; require(!sameDescription(original, value), "blend use re-resolved");
  value = original; value.flags = REMIXAPI_RETAINED_ORDERED; require(!sameDescription(original, value), "ordered transition is full build");
  for (unsigned i = 0; i < sizeof(value.blend) / sizeof(uint32_t); ++i) {
    value = original;
    const uint32_t word = 1;
    std::memcpy(reinterpret_cast<uint8_t*>(&value.blend) + i * sizeof(word), &word, sizeof(word));
    require(!sameDescription(original, value), "every blend word is part of description");
  }
}
struct Entry {
  bool desiredVisible = true;
  bool visible = false;
  std::vector<bool> children;
  unsigned visibilityWrites = 0;
  void apply(bool active) {
    dxvk::retained_compat::applyVisibility(*this, active, [&](bool target) {
      for (size_t i = 0; i < children.size(); ++i) children[i] = target;
      ++visibilityWrites;
    });
  }
  void rebuild(size_t count) {
    children.assign(count, true); // A newly built RetainedScene child defaults to visible.
    dxvk::retained_compat::finishBuild(*this, [&](bool target) {
      for (size_t i = 0; i < children.size(); ++i) children[i] = target;
      ++visibilityWrites;
    });
  }
  bool all(bool value) const { for (bool child : children) if (child != value) return false; return true; }
};
void testLifetime() {
  Entry entry; entry.desiredVisible = false;
  entry.rebuild(2);
  require(!entry.visible && entry.all(false), "new HIDDEN parent never leaves its children visible");
  entry.apply(true);
  require(!entry.visible && entry.all(false), "commit does not unhide a HIDDEN parent");
  entry.rebuild(4); // Material/mesh replacement invalidation changes child count.
  require(!entry.visible && entry.all(false) && !entry.desiredVisible, "asset invalidation keeps desired HIDDEN");
  entry.apply(false);
  require(!entry.visible && !entry.desiredVisible, "missing commit preserves HIDDEN intent");
  entry.desiredVisible = true; entry.apply(false);
  require(!entry.visible && entry.all(false), "unhide before a current commit remains disabled");
  entry.apply(true);
  require(entry.visible && entry.all(true), "current commit resumes without recreating children");
  const auto writes = entry.visibilityWrites;
  entry.apply(true); entry.apply(true);
  require(entry.visibilityWrites == writes, "stable commits perform no child visibility writes");
  entry.apply(false);
  require(!entry.visible && entry.desiredVisible && entry.all(false), "missing commit only changes applied state");
  entry.apply(true);
  require(entry.visible && entry.desiredVisible && entry.all(true), "resume restores visible intent");
  entry.desiredVisible = false; entry.apply(true);
  entry.desiredVisible = true; entry.apply(true);
  entry.desiredVisible = false; entry.apply(true);
  require(!entry.visible && entry.all(false), "same-frame toggles end at the latest desired state");
  entry.rebuild(1);
  require(!entry.visible && entry.all(false), "hide then replacement rebuild stays hidden");
  entry.children.clear(); entry.visible = false; // destroy/clear drops children, not host visibility defaults.
  entry.rebuild(3);
  require(!entry.visible && entry.all(false), "scene reset retains HIDDEN description");
  entry = Entry {};
  require(entry.desiredVisible && !entry.visible && entry.children.empty(), "new epoch starts with no inherited children");
}
void testWorldUpdateLifetime() {
  using State = dxvk::retained_compat::WorldBatchUpdateState;
  State state;
  require(!state.incremental(0) && !state.incremental(1), "unpublished cache cannot own incremental views");
  state.prepared(4);
  require(state.incremental(4) && !state.incremental(3), "fast path requires exact prepared membership generation");
  state.hide();
  require(state.incremental(4) && state.presentationDirty, "missing commit preserves membership but requires republish");
  state.presented();
  require(state.incremental(4) && !state.presentationDirty, "resume clears presentation only after publication");
  state.invalidateStructure();
  require(!state.incremental(4), "asset or material change forbids visibility-only update even after generic queues clear");
  state.prepared(5);
  require(state.incremental(5) && !state.incremental(4), "full prepare binds the new generation");
  state.lostPublication();
  require(!state.ready && !state.incremental(5) && state.structureDirty,
          "failed prefix publication requires canonical fallback and future full republish");
  state.hide(); state.presented();
  require(!state.incremental(5), "presentation changes cannot regain lost ownership readiness");
  state.prepared(0);
  require(!state.incremental(0), "zero helper revision never establishes ownership");
  state = {};
  require(!state.ready && state.structureDirty && !state.presentationDirty && !state.revision, "clear resets world publication state");

  Entry entry; entry.children.assign(2, true); entry.visible = true;
  State resumed; resumed.prepared(9);
  entry.apply(false); resumed.hide();
  require(entry.all(false) && resumed.presentationDirty, "missing commit hides canonical inputs and marks cached views for republish");
  entry.apply(true);
  require(entry.all(true) && resumed.incremental(9) && resumed.presentationDirty,
          "unchanged cached triangle mask still requires world presentation on resume");
  resumed.presented();
  require(!resumed.presentationDirty, "successful world presentation consumes the resume gate");
}

void testAdmissionAndPublishedSources() {
  using Admission = dxvk::retained_compat::WorldBatchAdmissionState;
  Admission admission;
  const auto start = Admission::Clock::time_point {};
  require(!admission.pending() && !admission.due(start), "no maintenance without pending admission");
  admission.request();
  require(admission.due(start), "first pending admission can run immediately");
  admission.begin(start);
  admission.request();
  require(!admission.due(start + std::chrono::milliseconds(999)) && admission.due(start + std::chrono::seconds(1)),
          "admission cadence uses monotonic time without delaying canonical rendering");
  admission.begin(start + std::chrono::seconds(1));
  require(!admission.pending(), "failed attempt does not automatically busy retry");
  admission.request(); admission.cancel();
  require(!admission.due(start + std::chrono::seconds(10)), "cancelled failure requires a new event");
  struct Source { unsigned mesh; float x; };
  struct Record { unsigned mesh; float x; };
  dxvk::retained_compat::PublishedSourceState<Source, Record> published;
  auto a = std::make_shared<const Source>(Source {1, 0});
  auto b = std::make_shared<const Source>(Source {2, 5});
  published.publish(a, Record {1, 0});
  published.stage();
  require(published.source == a && published.pending, "staging B does not advance published A");
  published.updateApplied(b, [](Source& old, const Source& current) { old.x = current.x; });
  published.record.x = b->x;
  require(published.source->mesh == 1 && published.source->x == 5 && published.pending,
          "fast transform preserves published old geometry while B is pending");
  require(published.source->mesh != b->mesh, "global refresh must use the published recipe, not staged B");
  using Destruction = dxvk::retained_compat::SourceDestruction;
  using dxvk::retained_compat::sourceDestruction;
  require(sourceDestruction(false, false) == Destruction::None, "unrelated mesh destruction changes neither source");
  require(sourceDestruction(false, true) == Destruction::ForgetPublished,
          "only published A destruction releases its view while pending B survives");
  require(sourceDestruction(true, false) == Destruction::RemoveEntry && sourceDestruction(true, true) == Destruction::RemoveEntry,
          "current B destruction retains original whole-record removal semantics");
  published.forget();
  published.updateApplied(b, [](Source& old, const Source& current) { old.x = current.x; });
  require(!published.source && published.pending && b->mesh == 2,
          "destroying published A leaves pending valid B staged for the next commit");
  published.publish(b, Record {2, 5});
  require(published.source == b && !published.pending, "successful canonical build advances source atomically");
  auto movedB = std::make_shared<const Source>(Source {2, 9});
  published.updateApplied(movedB, [](Source& old, const Source& current) { old.x = current.x; });
  require(published.source == movedB && !published.pending, "applied transform without pending geometry shares immutable source");
}

void testRigidExpandedParts() {
  using namespace dxvk;
  enum class Texgen { None, Authored };
  struct Transforms {
    Matrix4 objectToWorld, objectToView, worldToView, textureTransform;
    Texgen texgenMode = Texgen::Authored;
    int instancesToObject = 23;
    bool instancesFromHost = true;
  } parent;
  struct Draw {
    Transforms transforms; unsigned category = 0;
    Transforms& modifyTransformData() { return transforms; }
    unsigned& modifyCategoryFlags() { return category; }
  } draw;
  struct Part {
    Matrix4 submeshToObject;
    bool meshReplacement = true;
    int instancesToObject = 0;
    unsigned categories = 0;
    const int* material = nullptr;
  };
  const int original = 5, bark = 13, leaves = 29;
  std::array<Part, 3> parts {};
  parts[0].material = &bark; parts[0].categories = 31;
  auto& local = parts[0].submeshToObject;
  local[0][0] = 1.5f; local[1][1] = 1.5f; local[2][2] = 1.4f;
  local[3].x = -0.788252f; local[3].y = -0.000046f; local[3].z = 0.012931f;
  parts[1].material = &leaves; parts[1].categories = 63;
  const float angle = -15.f * 3.14159265f / 180.f;
  auto& turned = parts[1].submeshToObject;
  turned[0][0] = 0.9f * std::cos(angle); turned[0][1] = 0.9f * std::sin(angle);
  turned[1][0] = -turned[0][1]; turned[1][1] = turned[0][0]; turned[2][2] = 0.9f;
  parts[2].meshReplacement = false;
  parent.objectToWorld[0][0] = 0.f; parent.objectToWorld[0][1] = 1.f;
  parent.objectToWorld[1][0] = -1.f; parent.objectToWorld[1][1] = 0.f;
  parent.objectToWorld[3].x = -240.46234f; parent.objectToWorld[3].y = -1183.42578f;
  parent.worldToView[3].x = 130.f; parent.textureTransform[3].y = 44.f;
  std::vector<int> updates;
  size_t failed = 1;
  const auto run = [&] {
    return retained_compat::updateRigidParts(parts.size(), [&](size_t i){ return i != failed; }, [&](size_t i){
      const int& mat = retained_compat::applyExpandedPart(draw, parent, parts[i], original);
      updates.push_back(mat);
      const Matrix4 world = parent.objectToWorld * parts[i].submeshToObject;
      const Matrix4 view = parent.worldToView * world;
      require(!std::memcmp(&world, &draw.transforms.objectToWorld, sizeof(world)) &&
              !std::memcmp(&view, &draw.transforms.objectToView, sizeof(view)), "expanded part uses parent-local matrices and current camera");
      require(draw.category == parts[i].categories, "expanded parts retain independent categories");
      if (i < 2) {
        require(draw.transforms.textureTransform[3].y == 0 && draw.transforms.texgenMode == Texgen::None &&
                draw.transforms.instancesToObject == 0 && !draw.transforms.instancesFromHost, "mesh replacement resets native texture state");
      } else {
        require(draw.transforms.textureTransform[3].y == 44 && draw.transforms.texgenMode == Texgen::Authored &&
                draw.transforms.instancesToObject == 23 && draw.transforms.instancesFromHost, "includeOriginal inherits its source state");
      }
    });
  };
  require(!run() && updates.empty(), "invalid middle part never updates a healthy prefix");
  failed = 2; require(!run() && updates.empty(), "invalid last part preserves all temporal updates for the full rebuild");
  failed = parts.size(); require(run() && updates == std::vector<int>({bark,leaves,original}), "distinct bark, leaf and fallback materials in authored order");
  require(!retained_compat::updateRigidParts(0, [](size_t){return true;}, [](size_t){}), "empty expansion handled separately");
}

void testRelativeTransform() {
  using namespace dxvk;
  Matrix4 parent; parent[0][0] = 2; parent[1][1] = 3; parent[3].x = 20;
  Matrix4 local; local[3].x = 3; local[3].y = 5;
  const Matrix4 first = retained_compat::childTransform(parent, &local);
  require(first[3].x == 26 && first[3].y == 15, "parent transform composes before replacement local");
  parent[3].x = 50;
  const Matrix4 moved = retained_compat::childTransform(parent, &local);
  require(moved[3].x == 56 && moved[3].y == 15 && local[3].x == 3, "moving parent preserves authored local matrix");
  Matrix4 rotated; rotated[0][0] = 0; rotated[0][1] = 1; rotated[1][0] = -1; rotated[1][1] = 0;
  rotated[3].x = 10; rotated[3].y = 20;
  const Matrix4 turned = retained_compat::childTransform(rotated, &local);
  require(turned[3].x == 5 && turned[3].y == 23, "rotation keeps the authored local translation relative");
  const Matrix4 original = retained_compat::childTransform<Matrix4>(parent, nullptr);
  require(std::memcmp(&original, &parent, sizeof(parent)) == 0, "original surface stays at parent transform");
}

void testVisibilityDelta() {
  using namespace dxvk;
  struct Entry { remixapi_RetainedInstance record {}; bool desiredVisible = true; } entry;
  entry.record = instance();
  uint32_t count = 1, updates = 0;
  std::vector<uint64_t> queued;
  const auto update = [&](bool target, const std::vector<uint64_t>& ids) {
    return retained_compat::applyValidatedGroup(ids,
      [&](uint64_t id) { return id == entry.record.id && !(entry.record.flags & REMIXAPI_RETAINED_ORDERED); },
      [&](uint64_t) {
        retained_compat::setVisibilityIntent(entry, target, count,
          [&](uint64_t id) { queued.push_back(id); }, [&] { ++updates; });
        return true;
      });
  };
  const auto original = entry.record;
  require(!update(false, {entry.record.id, 0}) && entry.desiredVisible && queued.empty(), "invalid suffix leaves the group intact");
  require(update(false, {entry.record.id, entry.record.id}) && count == 0 && updates == 1, "duplicate hide applies once");
  require(!entry.desiredVisible && (entry.record.flags & REMIXAPI_RETAINED_HIDDEN), "hidden intent is stored in the original record");
  require(update(true, {entry.record.id}) && count == 1 && updates == 2 &&
          std::memcmp(&entry.record, &original, sizeof(original)) == 0, "show restores the exact unchanged description");
  entry.record.flags = REMIXAPI_RETAINED_ORDERED;
  require(!update(false, {entry.record.id}) && count == 1 && updates == 2, "ordered record stays outside the visibility lane");
  entry.record.flags = 0;
  try {
    retained_compat::setVisibilityIntent(entry, false, count, [](uint64_t) { throw std::bad_alloc(); }, [&] { ++updates; });
    require(false, "allocation failure must propagate");
  } catch (const std::bad_alloc&) {
    require(entry.desiredVisible && entry.record.flags == 0 && count == 1 && updates == 2, "queue failure preserves intent");
  }
}

void testEmptyOrderedReplacement() {
  using namespace dxvk;
  using namespace dxvk::retained_compat;
  struct Bucket { std::atomic<bool> stale {false}; std::vector<int> replacements; };
  struct Expansion {
    std::shared_ptr<const Bucket> sourceBucket;
    std::vector<int> primSlots, lightTemplates, lights, graphTemplates, graphs;
  } expansion;
  auto built = std::make_shared<Bucket>();
  expansion.sourceBucket = built;
  std::shared_ptr<const Bucket> current = built;
  require(currentOrderedBucket(expansion.sourceBucket, current) && suppressesOnly(expansion), "current empty bucket suppresses original");
  built->replacements.push_back(1);
  require(!suppressesOnly(expansion), "even an uncached prim or attached service needs the normal build");
  built->replacements.clear();
  for (auto* pending : {&expansion.primSlots, &expansion.lightTemplates, &expansion.lights, &expansion.graphTemplates, &expansion.graphs}) {
    pending->push_back(1);
    require(!suppressesOnly(expansion), "an expansion with a live or pending service is not empty");
    pending->clear();
  }
  current.reset();
  require(!currentOrderedBucket(expansion.sourceBucket, current), "removing a replacement restores the full build");
  current = std::make_shared<Bucket>();
  require(!currentOrderedBucket(expansion.sourceBucket, current), "replacement by another empty bucket still rebuilds");
  built->stale.store(true);
  require(!currentOrderedBucket(expansion.sourceBucket, expansion.sourceBucket), "a stale winning bucket rebuilds");
  struct Transforms { Matrix4 objectToWorld, worldToView, objectToView; } transforms;
  transforms.worldToView[0][0] = 2;
  Matrix4 absolute;
  absolute[3].x = 7;
  absolute[1][0] = std::numeric_limits<float>::denorm_min();
  storeOrderedTransform(transforms, absolute);
  require(std::memcmp(&transforms.objectToWorld, &absolute, sizeof(absolute)) == 0 && transforms.objectToView[3].x == 14,
    "suppressed original keeps the exact current matrix and derived view placement for reappearance");
}

void testOrderedMoves() {
  using namespace dxvk::retained_compat;
  struct Entry {
    remixapi_RetainedInstance record = instance();
    bool dirty = false, visible = true, orderedTransformPending = false;
    PublishedSourceState<int, remixapi_RetainedInstance> published;
    unsigned sourceCount = 1, orderedFrame = 10;
    std::vector<uint64_t> children = {87};
  } entry;
  entry.record.flags = REMIXAPI_RETAINED_ORDERED;
  auto moved = entry.record;
  moved.transform.matrix[1][0] = std::numeric_limits<float>::denorm_min();
  require(!canStageOrderedMove(entry, moved), "first ordered draw has no published source");
  entry.published.publish(std::make_shared<int>(1), entry.record);
  require(canStageOrderedMove(entry, moved), "bit-exact change to matrix basis is a move");
  require(!canStageOrderedMove(entry, entry.record), "identical matrix cannot imply a move");
  for (unsigned i = 0; i < 10; ++i) {
    auto changed = moved;
    switch (i) {
      case 0: ++changed.mesh; break;
      case 1: ++changed.id; break;
      case 2: ++changed.wetTag; break;
      case 3: ++changed.categoryFlags; break;
      case 4: ++changed.objectPickingValue; break;
      case 5: ++changed.doubleSided; break;
      case 6: ++changed.useBlend; break;
      case 7: ++changed.blend.tFactor; break;
      case 8: changed.flags = 0; break;
      case 9: changed.flags |= REMIXAPI_RETAINED_HIDDEN; break;
    }
    require(!canStageOrderedMove(entry, changed), "every full description change rebuilds");
  }
  entry.orderedTransformPending = true;
  require(!canStageOrderedMove(entry, moved), "second upsert before draw does not skip a full update");
  entry.orderedTransformPending = false; entry.dirty = true;
  require(!canStageOrderedMove(entry, moved), "existing asset/scene invalidation wins");
  entry.dirty = false; entry.published.stage();
  require(!canStageOrderedMove(entry, moved), "stale published description rebuilds");
  entry.published.publish(std::make_shared<int>(1), entry.record); entry.sourceCount = 3;
  require(!canStageOrderedMove(entry, moved), "fused or multisurface source rebuilds");
  entry.sourceCount = 1; entry.children.push_back(99);
  require(!canStageOrderedMove(entry, moved), "additional children rebuild as a unit");
  entry.children.resize(1);
  unsigned attempted = 0;
  auto move = [&](uint64_t id) { require(id == 87, "retain child identity"); ++attempted; return true; };
  require(!tryApplyOrderedMove(entry, 10, false, move), "same-frame repeat must redraw");
  require(!tryApplyOrderedMove(entry, 12, false, move), "frame gap redraws");
  require(!tryApplyOrderedMove(entry, 11, true, move), "capture/picking preserve original build");
  entry.visible = false;
  require(!tryApplyOrderedMove(entry, 11, false, move), "unseen-to-visible redraws");
  entry.visible = true; entry.dirty = true;
  require(!tryApplyOrderedMove(entry, 11, false, move) && !attempted, "invalidation between upsert and draw wins");
  entry.dirty = false;
  require(!tryApplyOrderedMove(entry, 11, false, [](uint64_t){return false;}), "unsupported child requests a full build now");
  require(tryApplyOrderedMove(entry, 11, false, move) && attempted == 1, "only one native-order move is applied");
  OrderedMoveGate gate = OrderedMoveGate::Accepted;
  require(!tryApplyOrderedMove(entry, 11, true, move, &gate) && gate == OrderedMoveGate::Inspect, "trace distinguishes inspection");
  require(!tryApplyOrderedMove(entry, 12, true, move, &gate) && gate == OrderedMoveGate::PriorFrame, "only the first rejecting gate counts");
  auto rejectedChild = [&](uint64_t) { gate = OrderedMoveGate::MaterialReplacement; return false; };
  require(!tryApplyOrderedMove(entry, 11, false, rejectedChild, &gate) && gate == OrderedMoveGate::MaterialReplacement,
          "child rejection reaches the counter unchanged");
}

void testOrderedValidation() {
  using namespace dxvk::retained_compat;
  constexpr uint64_t rule = 65;
  require(checkSurface(false, false, false, 0, rule, false) == SurfaceCheck::Valid,
          "ordinary mesh allows its default material");
  require(checkSurface(true, false, false, 0, rule, false) == SurfaceCheck::MissingMaterial,
          "a destroyed external material is an error on ordinary meshes");
  require(checkSurface(false, false, true, rule, rule, false) == SurfaceCheck::IncompatibleLegacy,
          "legacy mesh requires a valid opaque external material");
  require(checkSurface(true, true, true, rule, rule + 1, true) == SurfaceCheck::IncompatibleLegacy,
          "legacy mesh checks the current hash-generation rule");
  require(checkSurface(true, true, true, rule, rule, false) == SurfaceCheck::IncompatibleLegacy,
          "legacy mesh rejects a material recreated as translucent");

  std::vector<unsigned> ids {2, 1, 2};
  std::vector<unsigned> validated, drawn;
  bool middleMaterialLive = true;
  const auto validate = [&](unsigned id) {
    validated.push_back(id);
    return id < 3 && checkSurface(true, id != 1 || middleMaterialLive, true, rule, rule, true) == SurfaceCheck::Valid;
  };
  const auto draw = [&](unsigned id) { drawn.push_back(id); return true; };
  require(applyValidatedGroup(ids, validate, draw) && validated == ids && drawn == ids,
          "valid ordered group preserves duplicate ids and draw order");
  validated.clear(); drawn.clear();
  middleMaterialLive = false;
  require(!applyValidatedGroup(ids, validate, draw) && validated == std::vector<unsigned>({2, 1}) && drawn.empty(),
          "a material destroyed after successful replay prevents the next whole group from drawing");
  middleMaterialLive = true;
  require(applyValidatedGroup(ids, validate, draw) && drawn == ids,
          "recheck accepts the recreated valid material");
  validated.clear(); drawn.clear();
  ids[1] = 7;
  require(!applyValidatedGroup(ids, validate, draw) && drawn.empty(),
          "invalid middle record leaves even the valid prefix undrawn");
}
}
int main() {
  testEmptyOrderedReplacement();
  testRigidExpandedParts();
  testVisibilityDelta();
  testOrderedMoves();
  testOrderedValidation();
  testProtocol(); testClassification(); testLifetime(); testWorldUpdateLifetime(); testAdmissionAndPublishedSources(); testRelativeTransform();
  std::printf("Retained Compat v2/v3: %u CPU policy checks passed\n", checks);
  return 0;
}
