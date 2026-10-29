// SPDX-License-Identifier: MIT
// Adapted local runtime feature. See CREDITS.md.
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include "rtx_lens_flare_material.h"
#include "rtx_asset_data_manager.h"
#include "rtx_texture_manager.h"
#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../dxvk_objects.h"

namespace dxvk {
  uint64_t LensFlareMaterialSystem::refresh(uint32_t frame) {
    if (m_checkedFrame == frame) { return m_generation; }
    m_checkedFrame = frame;
    const bool active = enabled();
    const std::string identity = identityDds(), albedo = albedoPath(), emissive = emissivePath();
    const float level = std::isfinite(emissiveLevel()) ? std::clamp(emissiveLevel(), 0.0f, 65504.0f) : 0.0f;
    const uint64_t textureGeneration = m_device->getCommon()->getTextureManager().getTextureCacheGeneration();
    std::filesystem::file_time_type stamp {};
    if (active && !identity.empty()) {
      std::error_code error;
      stamp = std::filesystem::last_write_time(identity, error);
      if (error) { stamp = {}; }
    }
    if (m_initialized && active == m_enabled && identity == m_identityPath && albedo == m_albedoPath
        && emissive == m_emissivePath && level == m_level && stamp == m_stamp
        && textureGeneration == m_textureGeneration) { return m_generation; }
    m_initialized = true;
    m_enabled = active;
    m_identityPath = identity; m_albedoPath = albedo; m_emissivePath = emissive;
    m_level = level; m_stamp = stamp; m_textureGeneration = textureGeneration;
    m_hashes.clear(); m_albedo = {}; m_emissive = {};
    m_texturesLoaded = false; m_reportedMatch = false;
    ++m_generation;
    m_translationInvalid = true;
    if (active && !identity.empty()) {
      if (readIdentity()) {
        Logger::info(str::format("[LensFlare] Derived ", m_hashes.size(), " texture identities from ", identity));
      } else {
        Logger::warn(str::format("[LensFlare] Invalid or missing uncompressed 32-bit reference DDS: ", identity));
      }
    }
    return m_generation;
  }

  bool LensFlareMaterialSystem::takeTranslationInvalidation(uint32_t frame) {
    refresh(frame);
    const bool changed = m_translationInvalid;
    m_translationInvalid = false;
    return changed;
  }

  bool LensFlareMaterialSystem::readIdentity() {
    std::ifstream file(m_identityPath, std::ios::binary);
    std::array<uint32_t,32> h {};
    if (!file.read(reinterpret_cast<char*>(h.data()), sizeof(h)) || h[0] != 0x20534444 || h[1] != 124 || h[19] != 32) { return false; }
    const uint32_t height = h[3], width = h[4];
    if (!width || !height || width > 4096 || height > 4096 || (h[20] & 0x40) == 0
        || (h[20] & 4) || h[21] || h[22] != 32 || h[24] != 0xff00 || h[26] != 0xff000000
        || !((h[23] == 0xff0000 && h[25] == 0xff) || (h[23] == 0xff && h[25] == 0xff0000))) { return false; }
    const size_t rowBytes = size_t(width) * 4;
    const size_t pitch = (h[2] & 8) ? h[5] : rowBytes;
    if (pitch < rowBytes || pitch > rowBytes + 4096) { return false; }
    std::vector<unsigned char> pixels(rowBytes * height), row(pitch);
    for (uint32_t y = 0; y < height; ++y) {
      if (!file.read(reinterpret_cast<char*>(row.data()), row.size())) { return false; }
      std::memcpy(pixels.data() + size_t(y) * rowBytes, row.data(), rowBytes);
    }
    std::vector<unsigned char> swapped(pixels), flipped(pixels.size()), flippedSwapped(pixels.size());
    for (size_t i = 0; i < pixels.size(); i += 4) { std::swap(swapped[i], swapped[i+2]); }
    for (uint32_t y = 0; y < height; ++y) {
      std::memcpy(flipped.data() + size_t(y) * rowBytes, pixels.data() + size_t(height-1-y) * rowBytes, rowBytes);
      std::memcpy(flippedSwapped.data() + size_t(y) * rowBytes, swapped.data() + size_t(height-1-y) * rowBytes, rowBytes);
    }
    for (const auto* variant : { &pixels, &swapped, &flipped, &flippedSwapped }) {
      m_hashes.insert(XXH3_64bits(variant->data(), variant->size()));
    }
    return true;
  }

  bool LensFlareMaterialSystem::matches(uint64_t textureHash) const {
    return m_enabled && textureHash && textureHash != kEmptyHash && m_hashes.count(textureHash) != 0;
  }

  TextureRef LensFlareMaterialSystem::loadTexture(DxvkContext& ctx, const std::string& path) {
    if (path.empty()) { return {}; }
    auto asset = AssetDataManager::get().findAsset(path);
    if (asset == nullptr) {
      Logger::warn(str::format("[LensFlare] Replacement texture not found: ", path));
      return {};
    }
    return TextureRef { ctx.getCommonObjects()->getTextureManager().preloadTextureAsset(asset, ColorSpace::AUTO, false) };
  }

  void LensFlareMaterialSystem::apply(DxvkContext& ctx, MaterialData& material) {
    if (material.getType() != MaterialDataType::Opaque) { return; }
    auto& opaque = material.getOpaqueMaterialData();
    const auto& source = opaque.getAlbedoOpacityTexture();
    if (!source.isValid() || source.isImageEmpty() || !matches(source.getImageHash())) { return; }
    if (!m_texturesLoaded) {
      m_albedo = loadTexture(ctx, m_albedoPath);
      m_emissive = loadTexture(ctx, m_emissivePath);
      m_texturesLoaded = true;
    }
    if (m_albedo.isValid() && !m_albedo.isImageEmpty()) { opaque.setAlbedoOpacityTexture(m_albedo); }
    const TextureRef emission = m_emissive.isValid() && !m_emissive.isImageEmpty() ? m_emissive : opaque.getAlbedoOpacityTexture();
    opaque.setEnableEmission(true);
    opaque.setEmissiveColorTexture(emission);
    opaque.setEmissiveIntensity(m_level);
    opaque.setEmissiveColorConstant(Vector3(1.0f));
    opaque.setUseLegacyAlphaState(false);
    opaque.setBlendEnabled(true);
    opaque.setBlendType(BlendType::kEmissive);
    opaque.setInvertedBlend(false);
    opaque.setAlphaTestType(AlphaTestType::kAlways);
    // This is an authored material. Preserve its metadata without inheriting
    // the v2 shader flag for legacy-only specular and Fresnel controls.
    material = MaterialData { opaque, material.m_particleSystem, material.getIgnored() };
    if (!m_reportedMatch) {
      Logger::info(str::format("[LensFlare] Applied explicit emissive material level=", m_level, " generation=", m_generation));
      m_reportedMatch = true;
    }
  }
}
