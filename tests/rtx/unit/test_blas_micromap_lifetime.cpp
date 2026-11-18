// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>
#include "../../../src/util/rc/util_rc.h"
#include "../../../src/util/rc/util_rc_ptr.h"

// Exercise the exact production ownership code with intrusive DXVK Rc objects.
// The modeled queue retains each submission until its own fence is signaled.
namespace test {
  using dxvk::Rc;
  using dxvk::RcObject;
  void require(bool condition, const char* expression, int line) {
    if (!condition) {
      std::cerr << "Line " << line << ": " << expression << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
#define REQUIRE(value) require((value), #value, __LINE__)
  class DxvkResource : public RcObject {
  public:
    virtual ~DxvkResource() = default;
  };
  struct DxvkBufferCreateInfo {};
  struct MockCommon { int memoryManager() const { return 0; } };
  struct DxvkDevice {
    MockCommon common;
    MockCommon* getCommon() { return &common; }
  };
  constexpr int VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1;
  constexpr int VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR = 1;
  class DxvkAccelStructure : public DxvkResource {
  public:
    DxvkAccelStructure(DxvkDevice*,const DxvkBufferCreateInfo&,int,int,int,const char*) {}
    std::function<void()> beforeDestruction;
    ~DxvkAccelStructure() { if (beforeDestruction) beforeDestruction(); }
  };
  enum class DxvkAccess { None };
  class MockCommandList {
  public:
    std::vector<Rc<DxvkResource>> retained;
    template<DxvkAccess> void trackResource(Rc<DxvkResource> object) { retained.push_back(std::move(object)); }
  };
  class DxvkContext : public RcObject {
  public:
    MockCommandList* getCommandList() { return &list; }
    MockCommandList list;
  };
  struct PooledBlas {
    struct BoundOmm { uint64_t hash; Rc<DxvkResource> buffers; };
    Rc<DxvkAccelStructure> accelStructure;
    std::vector<BoundOmm> boundOmms;
  };
#include "../../../src/dxvk/rtx_render/rtx_blas_storage.inl"
  class TrackedInput final : public DxvkResource {
    unsigned& m_releases;
  public:
    explicit TrackedInput(unsigned& releases) : m_releases(releases) {}
    ~TrackedInput() { ++m_releases; }
  };
  void test() {
    DxvkDevice device;
    DxvkBufferCreateInfo info;
    Rc<DxvkContext> building = new DxvkContext, tracing = new DxvkContext;
    unsigned releasedA = 0, releasedB = 0;
    Rc<DxvkResource> a = new TrackedInput(releasedA), b = new TrackedInput(releasedB);
    PooledBlas pooled;
    pooled.accelStructure = new BlasStorage(&device,info,"dynamic");
    replaceBlasMicromaps(building,storage(pooled),a);
    replaceBlasMicromaps(building,storage(pooled),a);
    REQUIRE(building->list.retained.empty());
    tracing->list.trackResource<DxvkAccess::None>(pooled.accelStructure);
    a = nullptr;
    replaceBlasMicromaps(building,storage(pooled),b);
    REQUIRE(!releasedA && building->list.retained.size() == 1);
    b = nullptr;
    pooled.accelStructure = nullptr;
    REQUIRE(!releasedB);
    tracing->list.retained.clear();
    REQUIRE(releasedB == 1 && !releasedA);
    building->list.retained.clear();
    REQUIRE(releasedA == 1);

    // Merged snapshots, repeated bindings and physical clone/compaction.
    a = new TrackedInput(releasedA);
    b = new TrackedInput(releasedB);
    pooled.accelStructure = new BlasStorage(&device,info,"merged");
    std::vector<PooledBlas::BoundOmm> inputs{{1,a}, {2,b}, {3,a}};
    replaceBucketMicromaps(building,pooled,inputs);
    pooled.boundOmms = inputs;
    auto held = storage(pooled).micromaps.ptr();
    replaceBucketMicromaps(building,pooled,inputs);
    REQUIRE(building->list.retained.empty() && held == storage(pooled).micromaps.ptr());
    Rc<BlasStorage> copy = new BlasStorage(&device,info,"relocated");
    copy->micromaps = storage(pooled).micromaps;
    tracing->list.trackResource<DxvkAccess::None>(pooled.accelStructure);
    pooled.accelStructure = std::move(copy);
    replaceBucketMicromaps(building,pooled,{});
    REQUIRE(building->list.retained.size() == 1);
    inputs.clear();
    pooled.boundOmms.clear();
    a = nullptr;
    b = nullptr;
    building->list.retained.clear();
    REQUIRE(releasedA == 1 && releasedB == 1);
    tracing->list.retained.clear();
    REQUIRE(releasedA == 2 && releasedB == 2);

    // A historical TLAS holds only a device address. On retirement, protect
    // the previous physical BLAS to a later fence that dominates its last use.
    Rc<DxvkAccelStructure> retired = new BlasStorage(&device,info,"historical");
    a = new TrackedInput(releasedA);
    static_cast<BlasStorage*>(retired.ptr())->micromaps = a;
    bool destroyedPhysical = false;
    retired->beforeDestruction = [&] { REQUIRE(releasedA == 2); destroyedPhysical = true; };
    a = nullptr;
    tracing->list.trackResource<DxvkAccess::None>(retired);
    retired = nullptr;
    building->list.retained.clear();
    REQUIRE(!destroyedPhysical && releasedA == 2);
    tracing->list.retained.clear();
    REQUIRE(destroyedPhysical && releasedA == 3);
  }
}
int main() {
  test::test();
  std::cout << "Verified AS retirement, in-place rebinding, copy, unchanged generation, group lifetime and independent submission fences.\n";
}
