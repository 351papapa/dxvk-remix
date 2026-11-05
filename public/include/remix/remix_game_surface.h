#pragma once
#include <remix/remix_c.h>

// Fork extension. Discover support with GetGameValue("vc.api.gameSurfaceVersion").
#define REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_GAME_SURFACE_EXT ((remixapi_StructType)0x56430001)
typedef struct remixapi_InstanceInfoGameSurfaceEXT {
  remixapi_StructType sType;
  void* pNext;
  // Matches VC RS 210: class in bits 24..31, surface flags in 8..11, strength in 0..7.
  uint32_t wetTag;
} remixapi_InstanceInfoGameSurfaceEXT;
