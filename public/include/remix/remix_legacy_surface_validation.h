#pragma once

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>
#include <remix/remix_legacy_surface_v2.h>
#include <remix/remix_retained.h>

namespace remixapi::legacy_surface {

inline constexpr char kClientKey[] = "vc.client.legacyVertexAttributesVersion";
inline constexpr char kBridgeKey[] = "vc.bridge.legacyVertexAttributesVersion";
inline constexpr char kRuntimeKey[] = "vc.api.legacyVertexAttributesVersion";
inline constexpr uint32_t kWireMagic = 0x3241564c;
inline constexpr uint32_t kWireVersion = 2;

struct WireHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t recordSize;
  uint32_t count;
};
static_assert(sizeof(WireHeader) == 16, "Legacy V2 header ABI");

inline const char* clientQueryKey(const char* key) {
  return std::strcmp(key, kClientKey) == 0 ? kBridgeKey : key;
}
inline const char* serverQueryKey(const char* key) {
  return std::strcmp(key, kBridgeKey) == 0 ? kRuntimeKey : key;
}
inline bool isIntermediateCapabilityKey(const char* key) {
  return std::strcmp(key, kClientKey) == 0 || std::strcmp(key, kBridgeKey) == 0;
}
inline bool supportsV2(remixapi_ErrorCode result, const char* value, uint32_t size) {
  return result == REMIXAPI_ERROR_CODE_SUCCESS && value && size == 2 && value[0] == '2' && value[1] == '\0';
}
// Call after the public GetGameValue argument validation.
inline bool getCapabilityValue(const char* key, char* buffer, uint32_t capacity, uint32_t* actual) {
  if (isIntermediateCapabilityKey(key)) {
    *actual = 0;
    return true;
  }
  if (std::strcmp(key, kRuntimeKey) != 0) {
    return false;
  }
  *actual = 2;
  if (capacity >= 2) {
    buffer[0] = '2';
    buffer[1] = '\0';
  }
  return true;
}
inline bool validRecord(const remixapi_LegacySurfaceIdentityV2& record) {
  return record.reserved == 0 &&
    (record.vertexAttributeFlags & ~REMIXAPI_LEGACY_VERTEX_ATTRIBUTE_NORMAL_ABSENT) == 0;
}
inline bool validIdentity(const remixapi_LegacySurfaceIdentity& identity, uint64_t rule) {
  // Untextured legacy draws have material hash zero and draw hash == geometry hash.
  return identity.legacyDrawHash && identity.legacyGeometryHash &&
    identity.ruleFingerprint == rule &&
    (identity.legacyGeometryHash ^ identity.legacyMaterialHash) == identity.legacyDrawHash;
}

struct Input {
  const remixapi_MeshInfoLegacySurfaceEXT* v1 = nullptr;
  const remixapi_MeshInfoLegacySurfaceV2EXT* v2 = nullptr;
  bool hasLegacy() const { return v1 || v2; }
  uint32_t count() const { return v2 ? v2->surfaces_count : v1 ? v1->surfaces_count : 0; }
  const remixapi_LegacySurfaceIdentity& identity(uint32_t i) const {
    return v2 ? v2->surfaces_values[i].identity : v1->surfaces_values[i];
  }
  bool normalAbsent(uint32_t i) const {
    return v2 && (v2->surfaces_values[i].vertexAttributeFlags & REMIXAPI_LEGACY_VERTEX_ATTRIBUTE_NORMAL_ABSENT) != 0;
  }
};

inline bool readInput(const remixapi_MeshInfo& mesh, Input& result) {
  result = {};
  const void* next = mesh.pNext;
  for (uint32_t depth = 0; next && depth < 64; ++depth) {
    remixapi_StructType type;
    std::memcpy(&type, next, sizeof(type));
    if (type == REMIXAPI_STRUCT_TYPE_MESH_INFO_LEGACY_SURFACE_EXT) {
      if (result.hasLegacy()) {
        return false;
      }
      result.v1 = static_cast<const remixapi_MeshInfoLegacySurfaceEXT*>(next);
    } else if (type == REMIXAPI_STRUCT_TYPE_MESH_INFO_LEGACY_SURFACE_V2_EXT) {
      if (result.hasLegacy()) {
        return false;
      }
      result.v2 = static_cast<const remixapi_MeshInfoLegacySurfaceV2EXT*>(next);
    }
    const void* following = nullptr;
    std::memcpy(&following, static_cast<const uint8_t*>(next) + offsetof(remixapi_MeshInfoLegacySurfaceV2EXT, pNext), sizeof(following));
    next = following;
  }
  if (next) {
    return false;
  }
  if (!result.hasLegacy()) {
    return true;
  }
  if (!result.count() || (result.v2 && result.count() > REMIXAPI_RETAINED_MAX_BATCH) ||
      result.count() != mesh.surfaces_count || !mesh.surfaces_values ||
      (result.v1 && !result.v1->surfaces_values) || (result.v2 && !result.v2->surfaces_values)) {
    return false;
  }
  if (result.v2) {
    for (uint32_t i = 0; i < result.count(); ++i) {
      if (!validRecord(result.v2->surfaces_values[i]) ||
          (result.normalAbsent(i) && mesh.surfaces_values[i].skinning_hasvalue)) {
        return false;
      }
    }
  }
  return true;
}

inline bool encode(const Input& input, std::vector<uint8_t>& blob) {
  blob.clear();
  const uint32_t count = input.count();
  if (!input.hasLegacy() || !count || count > REMIXAPI_RETAINED_MAX_BATCH ||
      (input.v1 && input.v2) || (input.v1 && !input.v1->surfaces_values) ||
      (input.v2 && !input.v2->surfaces_values)) {
    return false;
  }
  if (input.v2) {
    for (uint32_t i = 0; i < count; ++i) {
      if (!validRecord(input.v2->surfaces_values[i])) {
        return false;
      }
    }
    const WireHeader header {kWireMagic, kWireVersion, sizeof(remixapi_LegacySurfaceIdentityV2), count};
    blob.resize(sizeof(header) + size_t(count) * sizeof(remixapi_LegacySurfaceIdentityV2));
    std::memcpy(blob.data(), &header, sizeof(header));
    std::memcpy(blob.data() + sizeof(header), input.v2->surfaces_values, blob.size() - sizeof(header));
  } else {
    blob.resize(size_t(count) * sizeof(remixapi_LegacySurfaceIdentity));
    std::memcpy(blob.data(), input.v1->surfaces_values, blob.size());
  }
  return true;
}

struct Decoded {
  std::vector<remixapi_LegacySurfaceIdentity> v1;
  std::vector<remixapi_LegacySurfaceIdentityV2> v2;
};

inline bool decode(const void* data, uint32_t size, uint32_t count, Decoded& result) {
  result = {};
  if (!data || !count || count > REMIXAPI_RETAINED_MAX_BATCH) {
    return false;
  }
  if (size == uint64_t(count) * sizeof(remixapi_LegacySurfaceIdentity)) {
    result.v1.resize(count);
    std::memcpy(result.v1.data(), data, size);
    return true;
  }
  if (size < sizeof(WireHeader)) {
    return false;
  }
  WireHeader header;
  std::memcpy(&header, data, sizeof(header));
  if (header.magic != kWireMagic || header.version != kWireVersion ||
      header.recordSize != sizeof(remixapi_LegacySurfaceIdentityV2) || header.count != count ||
      uint64_t(sizeof(header)) + uint64_t(header.count) * header.recordSize != size) {
    return false;
  }
  std::vector<remixapi_LegacySurfaceIdentityV2> records(count);
  std::memcpy(records.data(), static_cast<const uint8_t*>(data) + sizeof(header), size - sizeof(header));
  for (const auto& record : records) {
    if (!validRecord(record)) {
      return false;
    }
  }
  result.v2 = std::move(records);
  return true;
}

} // namespace remixapi::legacy_surface
