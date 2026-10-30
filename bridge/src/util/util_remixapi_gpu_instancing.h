#pragma once
#include <cstdint>
#include <cstring>
#include <remix/remix_c.h>

namespace remixapi::gpu_instancing {
// Bound a single bridge payload to 3 MiB. No per-transform draw commands are emitted.
constexpr uint32_t kMaxTransforms = 65536;
constexpr uint32_t kHeaderBytes = 3 * sizeof(uint32_t); // serializer size, sType, count
static_assert(sizeof(remixapi_Transform) == 48, "GPU transform wire stride");

inline bool valid(const remixapi_InstanceInfoGpuInstancingEXT& info) {
  return info.instanceTransforms_count <= kMaxTransforms &&
    (!info.instanceTransforms_count || info.instanceTransforms_values != nullptr);
}

inline bool validPayload(const void* data, uint32_t size) {
  if (!data || size < kHeaderBytes) {
    return false;
  }
  uint32_t header[3];
  std::memcpy(header, data, sizeof(header));
  return header[0] == size && header[1] == REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_GPU_INSTANCING_EXT &&
    header[2] <= kMaxTransforms && size == kHeaderBytes + header[2] * sizeof(remixapi_Transform);
}
}
