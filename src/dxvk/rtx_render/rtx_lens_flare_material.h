// SPDX-License-Identifier: MIT
// Adapted local runtime feature. See CREDITS.md.
#pragma once
#include <filesystem>
#include <string>
#include <unordered_set>
#include "rtx_common_object.h"
#include "rtx_materials.h"
#include "rtx_option.h"

namespace dxvk {
  class DxvkContext;
  class ImGUI;
  class LensFlareMaterialSystem : public CommonDeviceObject {
  public:
    explicit LensFlareMaterialSystem(DxvkDevice* device) : CommonDeviceObject(device) {}
    uint64_t refresh(uint32_t frame);
    bool matches(uint64_t textureHash) const;
    bool takeTranslationInvalidation(uint32_t frame);
    void apply(DxvkContext& ctx, MaterialData& material);

    RTX_OPTION_FLAG("rtx.fork.lensFlare", bool, enabled, false, RtxOptionFlags::InvalidatesDrawcallTranslation, "Enable emissive flare cards identified by a reference texture.");
    RTX_OPTION_FLAG("rtx.fork.lensFlare", std::string, identityDds, "", RtxOptionFlags::InvalidatesDrawcallTranslation, "Uncompressed 32-bit RGBA or BGRA DDS identifying the submitted flare texture.");
    RTX_OPTION_FLAG("rtx.fork.lensFlare", std::string, albedoPath, "", RtxOptionFlags::InvalidatesDrawcallTranslation, "Optional replacement albedo mask. Empty keeps the submitted texture.");
    RTX_OPTION_FLAG("rtx.fork.lensFlare", std::string, emissivePath, "", RtxOptionFlags::InvalidatesDrawcallTranslation, "Optional emission mask. Empty uses the albedo mask.");
    RTX_OPTION_FLAG("rtx.fork.lensFlare", float, emissiveLevel, 1.0f, RtxOptionFlags::InvalidatesDrawcallTranslation, "Emission before the global emissive intensity multiplier.");

  private:
    friend class ImGUI;
    bool readIdentity();
    TextureRef loadTexture(DxvkContext& ctx, const std::string& path);
    uint32_t m_checkedFrame = ~0u;
    uint64_t m_generation = 0;
    uint64_t m_textureGeneration = ~0ull;
    std::filesystem::file_time_type m_stamp {};
    std::string m_identityPath, m_albedoPath, m_emissivePath;
    std::unordered_set<uint64_t> m_hashes;
    TextureRef m_albedo, m_emissive;
    float m_level = 0.0f;
    bool m_enabled = false;
    bool m_initialized = false;
    bool m_translationInvalid = false;
    bool m_texturesLoaded = false;
    bool m_reportedMatch = false;
  };
}
