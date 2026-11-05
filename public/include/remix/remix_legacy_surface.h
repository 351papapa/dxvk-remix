#pragma once
#include <remix/remix_c.h>

#define REMIXAPI_STRUCT_TYPE_MESH_INFO_LEGACY_SURFACE_EXT ((remixapi_StructType)0x56430002)

// Supplied only from this session's observed original draw, in API-surface order.
// These are authored-asset keys, never the hashes used to build/cache API geometry.
typedef struct remixapi_LegacySurfaceIdentity {
  uint64_t legacyDrawHash;
  uint64_t legacyGeometryHash;
  uint64_t legacyMaterialHash;
  // low32: geometryAssetHashRule.raw(); high32: geometryHashGenerationRule.raw().
  // Read GetGameValue("vc.api.legacySurfaceRule") as hexadecimal at warmup.
  uint64_t ruleFingerprint;
} remixapi_LegacySurfaceIdentity;

typedef struct remixapi_MeshInfoLegacySurfaceEXT {
  remixapi_StructType sType;
  void* pNext;
  uint32_t surfaces_count;
  const remixapi_LegacySurfaceIdentity* surfaces_values;
} remixapi_MeshInfoLegacySurfaceEXT;

#ifdef __cplusplus
static_assert(sizeof(remixapi_LegacySurfaceIdentity) == 32, "Legacy identity wire ABI");
#endif
