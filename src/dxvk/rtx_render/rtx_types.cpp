/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include <algorithm>

#include "rtx_types.h"
#include "rtx_asset_replacer.h"
#include "rtx_options.h"
#include "rtx_instance_manager.h"
#include "rtx_light_manager.h"
#include "graph/rtx_graph_instance.h"
#include "dxvk_scoped_annotation.h"

namespace dxvk {

  // Instance constructor, getter, and assignment operator
  PrimInstance::PrimInstance(RtInstance* instance) : m_type(Type::Instance) {
    m_ptr.instance = instance;
  }
  RtInstance* PrimInstance::getInstance() const {
    if (m_type != Type::Instance) {
      return nullptr;
    }
    return m_ptr.instance;
  }

  // Light constructor, getter, and assignment operator
  PrimInstance::PrimInstance(RtLight* light) : m_type(Type::Light) {
    m_ptr.light = light;
  }
  RtLight* PrimInstance::getLight() const {
    if (m_type != Type::Light) {
      return nullptr;
    }
    return m_ptr.light;
  }

  // Graph constructor, getter, and assignment operator
  PrimInstance::PrimInstance(GraphInstance* graph) : m_type(Type::Graph) {
    m_ptr.graph = graph;
  }
  GraphInstance* PrimInstance::getGraph() const {
    if (m_type != Type::Graph) {
      return nullptr;
    }
    return m_ptr.graph;
  }

  PrimInstance::Type PrimInstance::getType() const {
    if (m_ptr.untyped == nullptr) {
      return Type::None;
    }
    return m_type;
  }

  PrimInstance::PrimInstance(void* owner, Type type) : m_type(type) {
    m_ptr.untyped = owner;
  }

  void* PrimInstance::getUntyped() const {
    return m_ptr.untyped;
  }

  void PrimInstance::setReplacementInstance(ReplacementInstance* replacementInstance, size_t replacementIndex) {
    PrimInstanceOwner* prim = nullptr;
    if (m_type == Type::Instance) {
      prim = &m_ptr.instance->getPrimInstanceOwner();
    } else if (m_type == Type::Light) {
      prim = &m_ptr.light->getPrimInstanceOwner();
    } else if (m_type == Type::Graph) {
      prim = &m_ptr.graph->getPrimInstanceOwner();
    }

    if (prim) {
      prim->setReplacementInstance(replacementInstance, replacementIndex, m_ptr.untyped, m_type);
    }
  }

  std::ostream& operator << (std::ostream& os, PrimInstance::Type type) {
    switch (type) {
      ENUM_NAME(PrimInstance::Type::Instance);
      ENUM_NAME(PrimInstance::Type::Light);
      ENUM_NAME(PrimInstance::Type::Graph);
      ENUM_NAME(PrimInstance::Type::None);
    }
    return os << static_cast<uint8_t>(type);
  }

  ReplacementInstance::~ReplacementInstance() {
    clear();
  }

  void ReplacementInstance::clear() {
    // Mark all prim entities for GC and detach their back-pointers, then drop
    // the prim/root slots, the active-replacements tracking pointer, and the
    // cached aggregate bounding boxes so the RI is in a clean "no replacement
    // attached" state. setup() is the matching re-init.
    for (size_t i = 0; i < prims.size(); i++) {
      RtInstance* subInstance = prims[i].getInstance();
      if (subInstance) {
        subInstance->markForGarbageCollection();
      }
      GraphInstance* graphInstance = prims[i].getGraph();
      if (graphInstance) {
        graphInstance->removeInstance();
      }
      RtLight* light = prims[i].getLight();
      if (light) {
        light->markForGarbageCollection();
      }
      prims[i].setReplacementInstance(nullptr, kInvalidReplacementIndex);
    }
    prims.clear();
    root = PrimInstance();
    activeReplacements.reset();
    legacyMaterialIdentityHash = kEmptyHash;
    geometryBoundingBox.invalidate();
    lightBoundingBox.invalidate();
    boundingBoxDirty = true;
    dirtyFlags = kAllDirtyFlags;
  }

  ReplacementInstance::ReplacementInstance(const LookupKey& key, uint32_t newId, uint32_t frameId)
      : id(newId)
      , identityHash(key.identityHash)
      , spatialMapHash(key.spatialMapHash)
      , materialHash(key.materialHash)
      , vertexPositionHash(key.vertexPositionHash)
      , centroid(key.worldPos)
      , frameCreated(frameId)
      , textureTransform(key.textureTransform)
      , texgenMode(key.texgenMode) {
    // No prior data to diff against; every field is effectively new. Set all
    // dirty bits so downstream update logic that gates individual steps on
    // specific bits runs the full update on the RI's first submission.
    dirtyFlags = kAllDirtyFlags;
  }

  void ReplacementInstance::setup(PrimInstance newRoot, size_t numPrims,
                                  std::shared_ptr<const ReplacementBucket> replacements) {
    clear();
    prims.resize(numPrims);
    root = newRoot;
    activeReplacements = std::move(replacements);
  }

  void ReplacementInstance::recalculateBoundingBox(
      const Matrix4& newObjectToWorld,
      const AxisAlignedBoundingBox* originalGeometryBBox) {
    objectToWorld = newObjectToWorld;

    if (!boundingBoxDirty) {
      return;
    }

    AxisAlignedBoundingBox geoBBox;
    AxisAlignedBoundingBox litBBox;

    if (activeReplacements == nullptr) {
      geoBBox = *originalGeometryBBox;
    } else {
      for (const auto& replacement : activeReplacements->replacements) {
        if (replacement.includeOriginal && originalGeometryBBox != nullptr) {
          geoBBox.unionWith(*originalGeometryBBox);
        } else if (replacement.type == AssetReplacement::eMesh && replacement.geometry != nullptr) {
          const AxisAlignedBoundingBox& srcBBox = replacement.geometry->data.boundingBox;
          if (srcBBox.isValid()) {
            const Vector3& mn = srcBBox.minPos;
            const Vector3& mx = srcBBox.maxPos;
            const Vector3 corners[8] = {
              Vector3(mn.x, mn.y, mn.z), Vector3(mx.x, mn.y, mn.z),
              Vector3(mn.x, mx.y, mn.z), Vector3(mn.x, mn.y, mx.z),
              Vector3(mx.x, mx.y, mn.z), Vector3(mn.x, mx.y, mx.z),
              Vector3(mx.x, mn.y, mx.z), Vector3(mx.x, mx.y, mx.z)
            };
            for (const Vector3& corner : corners) {
              const Vector3 transformed = (replacement.replacementToObject * Vector4(corner, 1.0f)).xyz();
              for (uint32_t j = 0; j < 3; j++) {
                geoBBox.minPos[j] = std::min(geoBBox.minPos[j], transformed[j]);
                geoBBox.maxPos[j] = std::max(geoBBox.maxPos[j], transformed[j]);
              }
            }
          }
        } else if (replacement.type == AssetReplacement::eLight && replacement.lightData.has_value()) {
          RtLight objectSpaceLight = replacement.lightData->toRtLight();
          const Vector3 pos = objectSpaceLight.getPosition();
          float lightRadius = 0.f;
          if (objectSpaceLight.getType() == RtLightType::Sphere) {
            lightRadius = objectSpaceLight.getSphereLight().getRadius();
          }
          for (uint32_t j = 0; j < 3; j++) {
            litBBox.minPos[j] = std::min(litBBox.minPos[j], pos[j] - lightRadius);
            litBBox.maxPos[j] = std::max(litBBox.maxPos[j], pos[j] + lightRadius);
          }
        }
      }
    }

    if (geoBBox.isValid()) {
      geometryBoundingBox = geoBBox;
    }
    if (litBBox.isValid()) {
      lightBoundingBox = litBBox;
    }
    boundingBoxDirty = false;
  }

  bool PrimInstanceOwner::isRoot(const void* owner) const {
    return m_replacementInstance != nullptr
      && m_replacementIndex != ReplacementInstance::kInvalidReplacementIndex
      && m_replacementInstance->root.getUntyped() == owner;
  }

  void PrimInstanceOwner::setReplacementInstance(ReplacementInstance* replacementInstance, size_t replacementIndex, void* owner, PrimInstance::Type type) {
    // No-op if already linked to the same slot
    if (m_replacementInstance == replacementInstance && m_replacementIndex == replacementIndex) {
      return;
    }

    // Unlink from current ReplacementInstance
    if (m_replacementInstance != nullptr &&
        m_replacementIndex < m_replacementInstance->prims.size()) {
      PrimInstance& currentSlot = m_replacementInstance->prims[m_replacementIndex];
      if (currentSlot.getUntyped() == owner) {
        currentSlot = PrimInstance();
      }
      if (m_replacementInstance->root.getUntyped() == owner) {
        m_replacementInstance->root = PrimInstance();
      }
    }

    // Link to new ReplacementInstance
    m_replacementInstance = replacementInstance;
    m_replacementIndex = replacementIndex;

    if (m_replacementInstance != nullptr &&
        m_replacementIndex < m_replacementInstance->prims.size()) {
      PrimInstance& targetSlot = m_replacementInstance->prims[m_replacementIndex];
      if (targetSlot.getUntyped() != nullptr && targetSlot.getUntyped() != owner) {
        targetSlot.setReplacementInstance(nullptr, ReplacementInstance::kInvalidReplacementIndex);
      }
      targetSlot = PrimInstance(owner, type);
    }
  }

  uint32_t RasterGeometry::calculatePrimitiveCount() const {
    const uint32_t elementCount = usesIndices() ? indexCount : vertexCount;
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return elementCount / 3;

    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return elementCount >= 3
        ? elementCount - 2
        : 0;

    default:
      assert(!"Unsupported primitive topology");
      return UINT32_MAX;
    }
  }

  bool SceneDraw::isEye() const {
    if (RtxOptions::Eye::enable() && RtxOptions::Eye::assumeViewTexgenModeAsEye()) {
      return getTransformData().texgenMode == TexGenMode::ViewPositions;
    }
    return false;
  }

  void SceneDraw::setCategory(InstanceCategories category, bool doSet) {
    if (doSet) {
      categories.set(category);
    }
  }

  void SceneDraw::removeCategory(InstanceCategories category) {
    categories.clr(category);
  }

  BlasEntry::BlasEntry(const SceneDraw& input_)
    : input(input_) {
    }

  void BlasEntry::unlinkInstance(RtInstance* instance) {
    if (m_linkedInstances.erase(instance) == 0) {
      ONCE(Logger::err("Tried to unlink an instance, which was never linked!"));
    }
  }

} // namespace dxvk
