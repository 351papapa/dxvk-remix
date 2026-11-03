#pragma once
#include <remix/remix_c.h>

#define REMIXAPI_RETAINED_VERSION 3u
#define REMIXAPI_RETAINED_VERSION_2 2u
#define REMIXAPI_RETAINED_ORDERED 1u
#define REMIXAPI_RETAINED_HIDDEN 2u
#define REMIXAPI_RETAINED_MAX_BATCH 4096u
#define REMIXAPI_GET_RETAINED_INTERFACE_NAME "remixapi_GetRetainedInterface"

// Versions 2 and 3 have the same layout. Version 3 adds persistent HIDDEN unordered records.
// ORDERED and HIDDEN are mutually exclusive. Mesh surfaces own their material handles.
// Every field on the wire has fixed width; no pNext or client pointers are retained.
#pragma pack(push, 8)
typedef struct remixapi_RetainedBlend {
  uint32_t alphaTestEnabled;
  uint32_t alphaTestReferenceValue;
  uint32_t alphaTestCompareOp;
  uint32_t alphaBlendEnabled;
  uint32_t srcColorBlendFactor;
  uint32_t dstColorBlendFactor;
  uint32_t colorBlendOp;
  uint32_t textureColorArg1Source;
  uint32_t textureColorArg2Source;
  uint32_t textureColorOperation;
  uint32_t textureAlphaArg1Source;
  uint32_t textureAlphaArg2Source;
  uint32_t textureAlphaOperation;
  uint32_t tFactor;
  uint32_t isTextureFactorBlend;
  uint32_t srcAlphaBlendFactor;
  uint32_t dstAlphaBlendFactor;
  uint32_t alphaBlendOp;
  uint32_t writeMask;
  uint32_t isVertexColorBakedLighting;
} remixapi_RetainedBlend;

typedef struct remixapi_RetainedInstance {
  uint64_t id;
  // (uint64_t)(uintptr_t) of the mesh handle returned by this process's Remix API.
  uint64_t mesh;
  remixapi_Transform transform;
  uint32_t categoryFlags;
  uint32_t doubleSided;
  uint32_t wetTag;
  uint32_t useBlend;
  uint32_t objectPickingValue;
  uint32_t flags;
  remixapi_RetainedBlend blend;
} remixapi_RetainedInstance;

typedef struct remixapi_RetainedStatus {
  uint64_t epoch;
  uint64_t commitCount;
  uint64_t submittedCount;
  uint32_t instanceCount;
  // Sticky asynchronous error; only a successful Clear resets it.
  uint32_t lastError;
  uint32_t lastCommitFrame;
  uint32_t reserved;
} remixapi_RetainedStatus;
#pragma pack(pop)

// Single submission thread. All mutations and CommitFrame are ordered and asynchronous.
// Clear must start each session with a nonzero, increasing epoch. Older epochs are rejected.
// ORDERED records are replayed only by DrawOrdered, in the supplied ID order at its call position.
// DrawOrdered validates the whole group before submission; asynchronous errors are sticky in GetStatus.
// CommitFrame explicitly replays the unordered set once for the current runtime frame, using
// its current camera. There is no automatic replay. Call after camera acquisition and before RTX injection.
// Version 2 hosts remove unseen instances. Version 3 hosts may retain them with HIDDEN until shown or unloaded.
// Clear on mode/world/device transitions. The returned table echoes the requested version.
// GetStatus synchronizes; use at initialization, reset, and diagnostics, not every frame.
// Mesh destruction removes its retained instances. Material destruction fails replay closed.
// Transient scene-cache clears preserve the catalog and the external API mesh/material handles.
typedef struct remixapi_RetainedInterface {
  uint32_t version;
  uint32_t structSize;
  remixapi_ErrorCode (REMIXAPI_PTR* BatchUpsert)(uint64_t epoch, uint32_t count, const remixapi_RetainedInstance* instances);
  remixapi_ErrorCode (REMIXAPI_PTR* BatchRemove)(uint64_t epoch, uint32_t count, const uint64_t* ids);
  remixapi_ErrorCode (REMIXAPI_PTR* Clear)(uint64_t epoch);
  remixapi_ErrorCode (REMIXAPI_PTR* CommitFrame)(uint64_t epoch);
  remixapi_ErrorCode (REMIXAPI_PTR* GetStatus)(remixapi_RetainedStatus* status);
  remixapi_ErrorCode (REMIXAPI_PTR* DrawOrdered)(uint64_t epoch, uint32_t count, const uint64_t* ids);
} remixapi_RetainedInterface;

typedef remixapi_ErrorCode (REMIXAPI_PTR* PFN_remixapi_GetRetainedInterface)(
  uint32_t version, uint32_t structSize, remixapi_RetainedInterface* outInterface);

#ifdef __cplusplus
static_assert(sizeof(remixapi_RetainedBlend) == 80, "Retained blend wire ABI");
static_assert(sizeof(remixapi_RetainedInstance) == 168, "Retained instance wire ABI");
static_assert(sizeof(remixapi_RetainedStatus) == 40, "Retained status wire ABI");
#endif
