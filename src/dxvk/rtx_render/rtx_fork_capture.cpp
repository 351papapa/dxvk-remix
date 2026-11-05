// The fork_hooks functions of the GameCapturer capture path (see rtx_fork_hooks.h).
// captureMaterialApiPath accesses GameCapturer's private members, so GameCapturer names it a friend.

// Replicate the BASE_DIR macro so the texture export paths resolve the same
// way they do in rtx_game_capturer.cpp.
#include "../../util/util_filesys.h"
#define BASE_DIR (util::RtxFileSys::path(util::RtxFileSys::Captures).string())

#include "rtx_fork_hooks.h"

#include "rtx_game_capturer.h"       // GameCapturer (full definition for friend access)
#include "rtx_instance_manager.h"    // RtInstance
#include "rtx_materials.h"           // LegacyMaterialData, kSurfaceMaterialInvalidTextureIndex
#include "rtx_texture_manager.h"     // RtxTextureManager, TextureRef
#include "rtx_constants.h"           // kEmptyHash
#include "rtx_options.h"             // RtxOptions::leftHandedCoordinateSystem()

#include "../../lssusd/game_exporter_types.h"   // lss::Export, lss::Material, lss::ext, lss::commonDirName
#include "../../lssusd/game_exporter_paths.h"
#include "../../lssusd/usd_include_begin.h"
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include "../../lssusd/usd_include_end.h"

#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {
namespace fork_hooks {

  // Implements GameCapturer::captureMaterial. A D3D9 material exports its color texture directly. An API material
  // has none, so its albedo is found in the texture table by hash. The result is cached in
  // GameCapturer::m_pCap->materials under runtimeMaterialHash.
  void captureMaterialApiPath(
      GameCapturer& capturer,
      const Rc<DxvkContext> ctx,
      const RtInstance& rtInstance,
      XXH64_hash_t runtimeMaterialHash,
      const LegacyMaterialData& materialData,
      bool bEnableOpacity) {
    lss::Material lssMat; // to be populated

    // Use runtime material hash so replacements resolve to the same material at runtime.
    const std::string matName = dxvk::hashToString(runtimeMaterialHash);
    lssMat.matName = matName;

    const auto& colorTexture = materialData.getColorTexture();
    XXH64_hash_t textureHash = colorTexture.getImageHash();

    // For API-submitted materials the LegacyMaterialData has no direct color texture; look
    // up the albedo texture via the instance's surface-material texture index instead.
    if (textureHash == 0 || textureHash == kEmptyHash) {
      const uint32_t albedoTexIndex = rtInstance.getAlbedoOpacityTextureIndex();
      if (albedoTexIndex != kSurfaceMaterialInvalidTextureIndex) {
        const RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();
        const auto& textureTable = textureManager.getTextureTable();
        if (albedoTexIndex < textureTable.size() && textureTable[albedoTexIndex].isValid()) {
          textureHash = textureTable[albedoTexIndex].getImageHash();
        }
      }
    }

    if (colorTexture.isValid() && colorTexture.getImageView()) {
      // Standard D3D9 material path: export the color texture directly.
      auto* imageView = colorTexture.getImageView();
      if (imageView && imageView->image().ptr()) {
        try {
          const std::string albedoTexFilename(matName + lss::ext::dds);
          capturer.m_exporter.dumpImageToFile(ctx, BASE_DIR + lss::commonDirName::texDir,
                                              albedoTexFilename,
                                              imageView->image());
          const std::string albedoTexPath = str::format(BASE_DIR + lss::commonDirName::texDir, albedoTexFilename);
          lssMat.albedoTexPath = albedoTexPath;
        } catch (const std::exception& e) {
          Logger::err(str::format("[GameCapturer] Failed to export D3D9 texture for material '", matName, "': ", e.what()));
        }
      } else {
        Logger::warn(str::format("[GameCapturer] D3D9 texture has invalid image for material: ", matName));
      }
    } else if (textureHash != 0 && textureHash != kEmptyHash) {
      // API-submitted material: locate the image via the texture-manager table by hash and export it.
      RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();
      const auto& textureTable = textureManager.getTextureTable();

      const TextureRef* pFoundTexture = nullptr;
      for (const auto& textureRef : textureTable) {
        if (textureRef.isValid() && textureRef.getImageHash() == textureHash) {
          pFoundTexture = &textureRef;
          break;
        }
      }

      if (pFoundTexture && pFoundTexture->getImageView()) {
        auto* apiImageView = pFoundTexture->getImageView();
        if (apiImageView && apiImageView->image().ptr()) {
          const auto& imageInfo = apiImageView->image()->info();

          // Validate image has valid dimensions
          if (imageInfo.extent.width > 0 && imageInfo.extent.height > 0) {
            const std::string albedoTexFilename(matName + lss::ext::dds);
            try {
              capturer.m_exporter.dumpImageToFile(ctx, BASE_DIR + lss::commonDirName::texDir,
                                                  albedoTexFilename,
                                                  apiImageView->image());
              const std::string albedoTexPath = str::format(BASE_DIR + lss::commonDirName::texDir, albedoTexFilename);
              lssMat.albedoTexPath = albedoTexPath;
            } catch (const std::exception& e) {
              Logger::warn(str::format("[GameCapturer] Failed to export API texture for material ", matName, ": ", e.what()));
            }
          } else {
            Logger::warn(str::format("[GameCapturer] API texture has invalid dimensions for material: ", matName,
                                     " (", imageInfo.extent.width, "x", imageInfo.extent.height, ")"));
          }
        } else {
          Logger::warn(str::format("[GameCapturer] API texture has null image for material: ", matName,
                                   " (hash: 0x", std::hex, textureHash, std::dec, ")"));
        }
      } else {
        Logger::warn(str::format("[GameCapturer] Could not resolve API texture hash for material: ", matName,
                                 " (hash: 0x", std::hex, textureHash, std::dec, ")"));
      }
    }

    lssMat.enableOpacity = bEnableOpacity;

    // Sampler info is only available on the D3D9 path; API-submitted materials may have no sampler.
    const auto& sampler = materialData.getSampler();
    if (sampler != nullptr) {
      const auto& samplerCreateInfo = sampler->info();
      lssMat.sampler.addrModeU = samplerCreateInfo.addressModeU;
      lssMat.sampler.addrModeV = samplerCreateInfo.addressModeV;
      lssMat.sampler.filter = samplerCreateInfo.magFilter;
      lssMat.sampler.borderColor = samplerCreateInfo.borderColor;
    }

    // Cache by runtime hash for proper replacement lookup.
    capturer.m_pCap->materials[runtimeMaterialHash].lssData = lssMat;
    Logger::debug("[GameCapturer][" + capturer.m_pCap->idStr + "][Mat:" + matName + "] New");
  }

  // Applies the global coordinate-system transform for the USD export stage. With rtx.leftHandedCoordinateSystem
  // the handedness inversion is skipped, because external API content is already in consistent Y-up space.
  void captureCoordSystemSkip(lss::Export& exportPrep) {
    // For external API content with explicit left-handed coordinate system setting,
    // skip the coordinate transformation - it's already in consistent Y-up space.
    // External API cameras are always LHS, so if the game is configured as LHS,
    // assume external content.
    if (!RtxOptions::leftHandedCoordinateSystem()) {
      const bool bInvX = (!exportPrep.camera.view.bInv) && (exportPrep.camera.proj.bInv || exportPrep.camera.isLHS());
      const bool bInvY = (!exportPrep.camera.view.bInv) && exportPrep.camera.proj.bInv;
      const pxr::GfVec3d scale{ bInvX ? -1.0 : 1.0, bInvY ? -1.0 : 1.0, 1.0 };
      exportPrep.globalXform.SetScale(scale);
    }
  }

} // namespace fork_hooks
} // namespace dxvk
