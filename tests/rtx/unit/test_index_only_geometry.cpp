/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#include "../../../src/dxvk/rtx_render/rtx_index_only_geometry.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
  unsigned checks = 0;
  void require(bool value, const char* text) {
    ++checks;
    if (!value)throw std::runtime_error(text);
  }
  struct Allocation {int value = 0;};
  struct Attribute {
    std::shared_ptr<Allocation> storage;
    uint64_t sliceOffset = 0, sliceLength = 4096;
    uint32_t attributeOffset = 0, byteStride = 64, format = 1;
    struct Handle {Allocation* value;Allocation* ptr()const{return value;}};
    bool defined()const{return storage != nullptr;}
    Handle buffer()const{return {storage.get()};}
    uint64_t offset()const{return sliceOffset;}
    uint64_t length()const{return sliceLength;}
    uint32_t offsetFromSlice()const{return attributeOffset;}
    uint32_t stride()const{return byteStride;}
    uint32_t vertexFormat()const{return format;}
  };
  struct Source {
    uint64_t immutableVertexDataId = 1;
    uint32_t vertexCount = 32;
    Attribute positionBuffer, normalBuffer, texcoordBuffer, color0Buffer, blendWeightBuffer, blendIndicesBuffer;
    // The visible triangle list is deliberately outside the vertex proof.
    uint64_t indexContent = 5;
    uint32_t indexCount = 96;
  };
  struct Cached {
    std::shared_ptr<Allocation> historyBuffer[2];
    Attribute positionBuffer, previousPositionBuffer;
  };
}

int main() {
  try {
    using namespace dxvk::index_only_geometry;
    Source original;
    original.positionBuffer.storage = std::make_shared<Allocation>();
    original.normalBuffer = original.texcoordBuffer = original.color0Buffer = original.positionBuffer;
    original.normalBuffer.attributeOffset = 12;
    original.texcoordBuffer.attributeOffset = 24;
    original.color0Buffer.attributeOffset = 32;
    require(sameImmutableVertexData(original, original), "Identical immutable source rejected");
    Source now = original;
    now.indexContent = 800;now.indexCount = 3;
    require(sameImmutableVertexData(now, original), "Index-only visibility change invalidated stable vertices");
    now.immutableVertexDataId = 0;
    require(!sameImmutableVertexData(now, original), "Unproven producer accepted");
    now = original;now.immutableVertexDataId++;
    require(!sameImmutableVertexData(now, original), "Changed complete content identity accepted");
    now = original;now.vertexCount++;
    require(!sameImmutableVertexData(now, original), "Changed vertex count accepted");
    now = original;now.vertexCount = 0;
    require(!sameImmutableVertexData(now, original), "Empty vertex range accepted");
    now = original;now.positionBuffer.storage.reset();
    require(!sameImmutableVertexData(now, now), "Missing positions accepted");

    const std::array<Attribute Source::*,6> fields {{&Source::positionBuffer, &Source::normalBuffer,
      &Source::texcoordBuffer, &Source::color0Buffer, &Source::blendWeightBuffer, &Source::blendIndicesBuffer}};
    for(auto field:fields) {
      Source before = original;
      (before.*field).storage = original.positionBuffer.storage;
      for(unsigned mutation=0;mutation<7;++mutation) {
        now = before;auto& changed = now.*field;
        switch(mutation) {
          case 0:changed.storage = std::make_shared<Allocation>();break;
          case 1:changed.sliceOffset++;break;
          case 2:changed.sliceLength--;break;
          case 3:changed.attributeOffset++;break;
          case 4:changed.byteStride++;break;
          case 5:changed.format++;break;
          case 6:changed.storage.reset();break;
        }
        require(!sameImmutableVertexData(now, before), "Changed attribute backing/range/layout accepted");
        require(!sameImmutableVertexData(before, now), "Attribute comparison was asymmetric");
      }
    }
    // Vertex-buffer identity must survive a freed source address being recycled:
    // the previous source owns its resource, and a new allocation differs.
    std::weak_ptr<Allocation> held = original.positionBuffer.storage;
    Source previous = original;original = {};
    require(!held.expired(), "Previous immutable input failed to own its backing resource");

    Cached cached;
    cached.historyBuffer[0] = std::make_shared<Allocation>();cached.historyBuffer[0]->value = 100;
    cached.historyBuffer[1] = std::make_shared<Allocation>();cached.historyBuffer[1]->value = 90;
    cached.positionBuffer.storage = cached.historyBuffer[0];
    cached.previousPositionBuffer.storage = cached.historyBuffer[1];
    const auto current = cached.historyBuffer[0], stalePrevious = cached.historyBuffer[1];
    preserveUnchangedPositions(cached, false);
    require(cached.historyBuffer[0] == current && cached.historyBuffer[1] == current &&
      cached.previousPositionBuffer.storage == current, "Index-only update copied/dropped unchanged current positions");
    require(stalePrevious->value == 90, "Old history was overwritten");
    require(needsDetachedVertexWrite(cached.historyBuffer[0], cached.historyBuffer[1], false),
      "Same-frame real vertex update would overwrite aliased previous positions");
    // Exercise the production write-detachment policy: the next real vertex
    // update may modify current data while the captured previous positions stay.
    if(needsDetachedVertexWrite(cached.historyBuffer[0], cached.historyBuffer[1], false))
      cached.historyBuffer[0] = std::make_shared<Allocation>();
    cached.historyBuffer[0]->value = 110;
    require(cached.historyBuffer[1]->value == 100 && cached.previousPositionBuffer.storage->value == 100,
      "Vertex update destroyed last-frame positions after index-only update");
    cached.positionBuffer.storage = cached.historyBuffer[0];
    preserveUnchangedPositions(cached, true);
    require(cached.previousPositionBuffer.storage->value == 100 && cached.historyBuffer[0]->value == 110,
      "Second topology update in a frame erased real vertex motion history");
    cached.previousPositionBuffer = {};
    preserveUnchangedPositions(cached, true);
    require(cached.previousPositionBuffer.storage == cached.historyBuffer[0], "First-frame topology update lacked zero-motion positions");
    auto separate = std::make_shared<Allocation>();
    require(!needsDetachedVertexWrite(cached.historyBuffer[0], separate, false), "Private destination unnecessarily detached");
    require(needsDetachedVertexWrite(cached.historyBuffer[0], separate, true), "Aliasable immutable input would be written");
    require(needsDetachedVertexWrite(std::shared_ptr<Allocation>{}, separate, false), "Missing destination accepted");
    std::cout << "Index-only immutable vertex/history policy " << checks << " checks passed\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << error.what() << '\n';return 1;
  }
}
