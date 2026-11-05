#pragma once

#include <stddef.h>
#include <remix/remix_legacy_surface.h>

#define REMIXAPI_STRUCT_TYPE_MESH_INFO_LEGACY_SURFACE_V2_EXT ((remixapi_StructType)0x56430003)
#define REMIXAPI_LEGACY_VERTEX_ATTRIBUTE_NORMAL_ABSENT 0x1u

// Immutable original-draw attributes; authored USD geometry keeps its own attributes.
typedef struct remixapi_LegacySurfaceIdentityV2 {
  remixapi_LegacySurfaceIdentity identity;
  uint32_t vertexAttributeFlags;
  uint32_t reserved;
} remixapi_LegacySurfaceIdentityV2;

typedef struct remixapi_MeshInfoLegacySurfaceV2EXT {
  remixapi_StructType sType;
  void* pNext;
  uint32_t surfaces_count;
  const remixapi_LegacySurfaceIdentityV2* surfaces_values;
} remixapi_MeshInfoLegacySurfaceV2EXT;

// Bridge consumers must first require GetGameValue("vc.client.legacyVertexAttributesVersion") == "2".
// An older client may silently discard this extension. V1 and V2 must not coexist in a chain.
#ifdef __cplusplus
static_assert(sizeof(remixapi_LegacySurfaceIdentityV2) == 40, "Legacy V2 wire ABI");
static_assert(offsetof(remixapi_LegacySurfaceIdentityV2, vertexAttributeFlags) == 32, "Legacy V2 flags offset");
static_assert(offsetof(remixapi_LegacySurfaceIdentityV2, reserved) == 36, "Legacy V2 reserved offset");
#endif
