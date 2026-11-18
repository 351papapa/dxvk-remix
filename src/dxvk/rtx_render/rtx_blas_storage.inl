// SPDX-License-Identifier: MIT
  namespace {
    // Command lists retain the physical BLAS independently of its CPU pool entry.
    // Keep micromaps under that owner so retiring or reusing the PooledBlas
    // cannot release an in-flight build dependency.
    struct BlasMicromapOwner {
      Rc<DxvkResource> micromaps;
    };

    // Destroy the Vulkan AS before the owner releases its last micromap.
    class BlasStorage final : private BlasMicromapOwner, public DxvkAccelStructure {
    public:
      BlasStorage(DxvkDevice* device, const DxvkBufferCreateInfo& info, const char* name)
        : DxvkAccelStructure(device, info, device->getCommon()->memoryManager(),
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, name) { }

      using BlasMicromapOwner::micromaps;
    };

    class BlasMicromapGroup final : public DxvkResource {
    public:
      explicit BlasMicromapGroup(const std::vector<PooledBlas::BoundOmm>& bindings) {
        std::unordered_set<DxvkResource*> seen;
        for (const auto& binding : bindings) {
          if (binding.buffers != nullptr && seen.insert(binding.buffers.ptr()).second)
            m_resources.push_back(binding.buffers);
        }
      }
    private:
      std::vector<Rc<DxvkResource>> m_resources;
    };

    BlasStorage& storage(const PooledBlas& blas) {
      // Initial allocation, compaction and relocation all allocate BlasStorage.
      return *static_cast<BlasStorage*>(blas.accelStructure.ptr());
    }

    void replaceBlasMicromaps(const Rc<DxvkContext>& ctx, BlasStorage& blas, Rc<DxvkResource> next) {
      if (next.ptr() == blas.micromaps.ptr()) return;
      if (blas.micromaps != nullptr) {
        // A BUILD recorded now replaces the previous physical generation only
        // when it executes. Its former inputs survive to this command list's fence.
        ctx->getCommandList()->trackResource<DxvkAccess::None>(blas.micromaps);
      }
      blas.micromaps = std::move(next);
    }

    void replaceBucketMicromaps(const Rc<DxvkContext>& ctx, PooledBlas& blas,
                                const std::vector<PooledBlas::BoundOmm>& next) {
      if (next.size() < 2) {
        replaceBlasMicromaps(ctx, storage(blas), next.empty() ? nullptr : next.front().buffers);
        return;
      }
      const auto sameBuffer = [](const auto& a, const auto& b) { return a.buffers.ptr() == b.buffers.ptr(); };
      if (next.size() == blas.boundOmms.size() && std::equal(next.begin(), next.end(), blas.boundOmms.begin(), sameBuffer))
        return;
      replaceBlasMicromaps(ctx, storage(blas), new BlasMicromapGroup(next));
    }
  }
