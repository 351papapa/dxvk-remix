#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "../dxvk_include.h"

namespace dxvk {

  // One decoded image, packed the way DxvkContext::updateImage expects,
  // so the upload is a plain copy into the staging buffer.
  // Only the top mip level is kept, because RemixFX has no 'mips' directive
  // and a DDS mip chain would never be sampled.
  struct RtxExternalEffectImage {
    VkExtent3D extent = { 0, 0, 1 };
    VkFormat format = VK_FORMAT_UNDEFINED;
    // Bytes per row and per slice of 'data'.
    // Rows are block rows for a compressed format, as updateImage counts them.
    VkDeviceSize rowPitch = 0;
    VkDeviceSize layerPitch = 0;
    std::vector<uint8_t> data;
  };

  // Decodes an image file into memory.
  // PNG, JPEG, TGA and BMP go through stb_image, and DDS goes through gli.
  // The decoder is chosen from the extension, so a mislabelled file reports
  // the error of the decoder it was handed to.
  // 'srgb' selects the sRGB variant of the file's format, so sampling returns linear values.
  // A DDS that already declares an sRGB format keeps it whatever the flag says.
  // A format with no sRGB variant is an error rather than a silent fallback.
  // Returns false with 'error' set to the reason. The caller prefixes the file name.
  bool loadRtxExternalEffectImage(
    const std::filesystem::path& path,
    bool srgb,
    RtxExternalEffectImage& image,
    std::string& error);

} // namespace dxvk
