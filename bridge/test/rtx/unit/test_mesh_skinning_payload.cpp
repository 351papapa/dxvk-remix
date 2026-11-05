#include "util_remixapi.h"
#include <cstdio>
#include <cstring>
#include <vector>

using remixapi::util::serialize::MeshInfo;

class GuardedArray {
public:
  explicit GuardedArray(size_t size) {
    SYSTEM_INFO info {};
    GetSystemInfo(&info);
    const size_t committed = (size + info.dwPageSize - 1) / info.dwPageSize * info.dwPageSize;
    m_base = VirtualAlloc(nullptr, committed + info.dwPageSize, MEM_RESERVE, PAGE_NOACCESS);
    if (!m_base || !VirtualAlloc(m_base, committed, MEM_COMMIT, PAGE_READWRITE)) {
      std::abort();
    }
    data = static_cast<unsigned char*>(m_base) + committed - size;
  }
  ~GuardedArray() { VirtualFree(m_base, 0, MEM_RELEASE); }
  unsigned char* data;
private:
  void* m_base;
};

static bool serializeGuarded(const MeshInfo& mesh, void* destination) {
  __try {
    mesh.serialize(destination);
    return true;
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return false;
  }
}

int main() {
  unsigned cases = 0;
  for (const uint32_t count : {3u, 16u, 256u}) {
    for (const uint32_t bones : {1u, 2u, 4u}) {
      const uint32_t elements = count * bones;
      GuardedArray weights(elements * sizeof(float));
      GuardedArray indices(elements * sizeof(uint32_t));
      auto* weightData = reinterpret_cast<float*>(weights.data);
      auto* indexData = reinterpret_cast<uint32_t*>(indices.data);
      for (uint32_t i = 0; i < elements; ++i) {
        weightData[i] = 1.0f / bones;
        indexData[i] = i % 256;
      }
      std::vector<remixapi_HardcodedVertex> vertices(count);
      const uint32_t triangle[] = {0, 1, 2};
      remixapi_MeshInfoSurfaceTriangles surface {};
      surface.vertices_values = vertices.data();
      surface.vertices_count = count;
      surface.indices_values = triangle;
      surface.indices_count = 3;
      remixapi_MeshInfo info {};
      info.sType = REMIXAPI_STRUCT_TYPE_MESH_INFO;
      info.hash = 0x5741544500000100ull;
      info.surfaces_count = 1;
      info.surfaces_values = &surface;
      const uint32_t unskinnedSize = MeshInfo(info).size();
      surface.skinning_hasvalue = 1;
      surface.skinning_value = {bones, weightData, elements, indexData, elements};
      MeshInfo encoded(info);
      std::vector<unsigned char> bytes(encoded.size() + 16, 0xcd);
      if (!serializeGuarded(encoded, bytes.data())) {
        std::printf("FAIL: source overread for %u vertices and %u bones\n", count, bones);
        return 1;
      }
      if (encoded.size() != unskinnedSize + 12 + elements * 8) {
        std::printf("FAIL: incorrect skinning payload length\n");
        return 2;
      }
      for (size_t i = encoded.size(); i < bytes.size(); ++i) {
        if (bytes[i] != 0xcd) { return 3; }
      }
      MeshInfo decoded(bytes.data());
      decoded.deserialize();
      const auto& actual = decoded.surfaces_values[0].skinning_value;
      if (actual.bonesPerVertex != bones || actual.blendWeights_count != elements ||
          actual.blendIndices_count != elements ||
          std::memcmp(actual.blendWeights_values, weightData, elements * sizeof(float)) ||
          std::memcmp(actual.blendIndices_values, indexData, elements * sizeof(uint32_t))) {
        return 4;
      }
      MeshInfo reencoded(static_cast<const remixapi_MeshInfo&>(decoded));
      std::vector<unsigned char> second(reencoded.size());
      reencoded.serialize(second.data());
      if (second.size() != encoded.size() || std::memcmp(second.data(), bytes.data(), second.size())) {
        return 5;
      }
      ++cases;
    }
  }
  std::printf("PASS: %u protected skinning payload roundtrips\n", cases);
  return 0;
}
