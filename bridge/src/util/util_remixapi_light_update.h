#pragma once

#include <cstring>

#include <remix/remix_c.h>

namespace remixapi::light_update {
  inline constexpr char versionKey[] = "__bridge.lightDefinitions.version";
  inline constexpr char serverVersionKey[] = "__bridge.lightDefinitions.serverVersion";
  inline constexpr char statusKey[] = "__bridge.lightDefinitions.status";
  inline constexpr char serverStatusKey[] = "__bridge.lightDefinitions.serverStatus";
  inline constexpr char runtimeVersionKey[] = "__remix.lightDefinitions.version";

  struct Header {
    remixapi_StructType sType;
    const void* pNext;
  };

  // A single shape keeps validation bounded and rejects cycles before writing
  // any command bytes. Optional USD fields are owned by its serializer.
  inline bool validDefinition(const remixapi_LightInfo* info) {
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_LIGHT_INFO || !info->pNext) {
      return false;
    }
    const auto* shape = static_cast<const Header*>(info->pNext);
    if (shape->pNext) {
      return false;
    }
    switch (shape->sType) {
      case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_SPHERE_EXT:
      case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_RECT_EXT:
      case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISK_EXT:
      case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_CYLINDER_EXT:
      case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISTANT_EXT:
      case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DOME_EXT:
      case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_USD_EXT:
        return true;
      default:
        return false;
    }
  }

  inline const char* serverQueryKey(const char* key) {
    if (std::strcmp(key, versionKey) == 0) {
      return serverVersionKey;
    }
    if (std::strcmp(key, statusKey) == 0) {
      return serverStatusKey;
    }
    return key;
  }
}
