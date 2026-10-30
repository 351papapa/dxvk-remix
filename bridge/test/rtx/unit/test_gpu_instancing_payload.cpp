#include "util_remixapi_gpu_instancing.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>
using namespace remixapi::gpu_instancing;
static uint32_t checks = 0;
static void check(bool result) { ++checks; if (!result) { std::printf("GPU wire check %u failed\n", checks); std::exit(1); } }
int main() {
  remixapi_Transform transform {};
  remixapi_InstanceInfoGpuInstancingEXT info {};
  check(valid(info));
  info.instanceTransforms_count = 1;
  check(!valid(info));
  info.instanceTransforms_values = &transform;
  check(valid(info));
  for (uint32_t count : { 0u, 1u, 128u, kMaxTransforms }) {
    info.instanceTransforms_count = count;
    check(valid(info));
    const uint32_t size = kHeaderBytes + count * sizeof(remixapi_Transform);
    std::vector<uint8_t> blob(size + 1);
    const uint32_t header[] = { size, REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_GPU_INSTANCING_EXT, count };
    std::memcpy(blob.data() + 1, header, sizeof(header)); // deliberately unaligned queue view
    check(validPayload(blob.data() + 1, size));
    check(!validPayload(blob.data() + 1, size - 1));
    check(!validPayload(blob.data() + 1, size + 1));
    const uint32_t wrongType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
    std::memcpy(blob.data() + 1 + sizeof(uint32_t), &wrongType, sizeof(wrongType));
    check(!validPayload(blob.data() + 1, size));
  }
  info.instanceTransforms_count = kMaxTransforms + 1;
  check(!valid(info));
  info.instanceTransforms_count = UINT32_MAX;
  check(!valid(info));
  uint32_t header[] = { kHeaderBytes, REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_GPU_INSTANCING_EXT, UINT32_MAX };
  check(!validPayload(header, sizeof(header)));
  header[2] = kMaxTransforms + 1;
  check(!validPayload(header, sizeof(header)));
  check(!validPayload(nullptr, sizeof(header)));
  for (uint32_t size = 0; size < kHeaderBytes; ++size) { check(!validPayload(header, size)); }
  std::printf("GPU instancing wire: %u bounds/stride/count checks passed\n", checks);
}
