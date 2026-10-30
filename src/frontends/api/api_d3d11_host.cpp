/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// The Remix API's D3D11 host: a D3D11 game registers its device, Remix records into the game's immediate context and
// renders into the game's textures, and the game imports its textures and passes its UI as screen overlay layers.
// The backend's API entry points (rtx_remix_api.cpp) reach the host through RemixApiDevice.

#include <exception>
#include <memory>

#include <remix/remix_c.h>

#include "../../d3d11/d3d11_context_imm.h"
#include "../../d3d11/d3d11_device.h"
#include "../../d3d11/d3d11_texture.h"
#include "../../dxgi/dxgi_interfaces.h"
#include "../../dxvk/dxvk_perf_counters.h"
#include "../../dxvk/rtx_render/rtx_fork_api_recorder.h"
#include "../../dxvk/rtx_render/rtx_options.h"
#include "../../dxvk/rtx_render/rtx_remix_api_device.h"

namespace {
  using namespace dxvk;

  // A D3D11 game's immediate context, registered by remixapi_RegisterD3D11Device.
  class ApiD3D11Host final : public RemixApiDevice {
  public:
    ApiD3D11Host(D3D11ImmediateContext* context, const Rc<DxvkDevice>& device)
      : RemixApiDevice(device), m_context(context) { }

    D3D11ImmediateContext* context() const {
      return m_context;
    }

    RemixApiDeviceLock LockDevice() override {
      return RemixApiDeviceLock { m_context->LockContext() };
    }

    void shutdown() override {
      m_context->Flush();
      m_context->SynchronizeCsThread(DxvkCsThread::SynchronizeAll);
      m_context->DisableRemixHost();
      m_context->Release();
    }

    void synchronizeCommands() override {
      m_context->SynchronizeCsThread(DxvkCsThread::SynchronizeAll);
    }

    void flushDeferredUploads() override {
      m_context->FlushRemixInitUploads();
    }

  protected:
    void emitCs(RemixApiCsCommand& command, bool) override {
      m_context->EmitRemixCsWith([&command](DxvkCsChunkRef& chunk) { return command.pushTo(chunk); });
    }

  private:
    D3D11ImmediateContext* m_context;
  };

  ApiD3D11Host* registeredD3D11() {
    return dynamic_cast<ApiD3D11Host*>(remix_api::registeredDevice());
  }

  // A remixapi_PrepareD3D11Texture token.
  class PreparedD3D11Texture final : public RemixApiImportSource {
  public:
    PreparedD3D11Texture() = default;

    ~PreparedD3D11Texture() {
      remix_api::untrackImportSource(*this);
    }

    void markShared() const override {
      GetCommonTexture(texture.ptr())->MarkRemixShared();
    }

    void markImported() const override {
      GetCommonTexture(texture.ptr())->MarkRemixImported();
    }

    // The game's texture, held as long as the token.
    Com<ID3D11Texture2D> texture;
  };

  // The registered device's texture behind texture2D.
  // Returns null when texture2D is not a texture of that device.
  D3D11Texture2D* registeredTexture(ApiD3D11Host& host, void* texture2D, D3D11_TEXTURE2D_DESC& desc) {
    auto* texture = texture2D ? dynamic_cast<D3D11Texture2D*>(static_cast<ID3D11Texture2D*>(texture2D)) : nullptr;
    if (!texture) {
      return nullptr;
    }
    Com<ID3D11Device> owner;
    texture->GetDevice(&owner);
    Com<ID3D11Device> registered;
    host.context()->GetDevice(&registered);
    if (owner != registered) {
      return nullptr;
    }
    texture->GetDesc(&desc);
    return texture;
  }

  remixapi_ErrorCode drawScreenOverlayD3D11(
      const remixapi_ScreenOverlayLayerD3D11* layers, uint32_t count, const float* alphaCutoffs) {
    ApiD3D11Host* host = registeredD3D11();
    if (!host) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    return remix_api::drawScreenOverlay(*host, layers, count, alphaCutoffs, [host](void* texture2D) {
      D3D11_TEXTURE2D_DESC desc;
      D3D11Texture2D* texture = registeredTexture(*host, texture2D, desc);
      if (!texture || desc.SampleDesc.Count != 1 || desc.ArraySize != 1 || desc.Usage == D3D11_USAGE_STAGING) {
        return Rc<DxvkImage>();
      }
      return texture->GetCommonTexture()->GetImage();
    });
  }

  remixapi_ErrorCode prepareD3D11Texture(
      void* texture2D, uint32_t dxgiFormat, uint32_t maxDimension, remixapi_D3D11TextureToken* out_token) {
    if (!out_token) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    *out_token = nullptr;
    RemixApiDevice* remixDevice = remix_api::registeredDevice();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!texture2D) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    try {
      Com<IDXGIRemixTexture> source;
      if (FAILED(static_cast<IUnknown*>(texture2D)->QueryInterface(
          __uuidof(IDXGIRemixTexture), reinterpret_cast<void**>(&source)))) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      auto token = std::make_unique<PreparedD3D11Texture>();
      token->device = remixDevice->GetDXVKDevice();
      ApiD3D11Host* host = registeredD3D11();
      void* hostContext = host ? static_cast<ID3D11DeviceContext*>(host->context()) : nullptr;
      BOOL deferInitFlush = RtxOptions::HostImport::deferPrepareFlush() && hostContext != nullptr;
      if (FAILED(source->PrepareRemixView(token->device->handle(),
          static_cast<DXGI_FORMAT>(dxgiFormat), hostContext, &token->view, &deferInitFlush)) || token->view == nullptr) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      token->initFlushDeferred = deferInitFlush != FALSE;
      Com<ID3D11Texture2D> texture;
      if (SUCCEEDED(static_cast<IUnknown*>(texture2D)->QueryInterface(
          __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&texture)))) {
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);
        token->immutable = desc.Usage == D3D11_USAGE_IMMUTABLE;
        token->texture = std::move(texture);
        token->hasHostTexture = true;
      }
      token->maxDimension = maxDimension;
      remix_api::trackImportSource(*token, token->texture.ptr());
      *out_token = token.release();
      return REMIXAPI_ERROR_CODE_SUCCESS;
    } catch (const std::exception& e) {
      Logger::err(str::format("PrepareD3D11Texture: ", e.what()));
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
  }

  remixapi_ErrorCode importD3D11Texture(
      remixapi_D3D11TextureToken token, uint64_t hash, uint32_t encoding, bool replaceOnly,
      remixapi_TextureHandle* out_handle) {
    return remix_api::importTexture(static_cast<const PreparedD3D11Texture*>(token), hash, encoding, replaceOnly,
                                    out_handle);
  }

  // The size and format of a RenderD3D11 destination, which a recording keeps.
  ApiRecorder::TargetDesc describeTarget(void* destination) {
    ApiRecorder::TargetDesc target;
    ID3D11Texture2D* texture = nullptr;
    if (destination && SUCCEEDED(static_cast<IUnknown*>(destination)->QueryInterface(
        __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&texture)))) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      texture->Release();
      target.width = desc.Width;
      target.height = desc.Height;
      target.format = uint32_t(desc.Format);
    }
    return target;
  }

} // anonymous namespace

extern "C" {

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_RegisterD3D11Device(void* device) {
    auto* native = device ? dynamic_cast<dxvk::D3D11Device*>(static_cast<ID3D11Device*>(device)) : nullptr;
    if (!native) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    const remixapi_ErrorCode status = dxvk::remix_api::registerHostDevice([native]() {
      ID3D11DeviceContext* context = nullptr;
      native->GetImmediateContext(&context);
      auto* immediate = static_cast<dxvk::D3D11ImmediateContext*>(context);
      immediate->EnableRemixHost();
      return std::make_unique<ApiD3D11Host>(immediate, native->GetDXVKDevice());
    });
    if (status == REMIXAPI_ERROR_CODE_SUCCESS) {
      dxvk::Logger::info("[Remix.D3D11] Registered native D3D11 API host; game command stream and swapchain");
    }
    return status;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_RenderD3D11(void* destination, remixapi_Bool render) {
    if (dxvk::ApiRecorder::shouldRecord()) {
      return dxvk::ApiRecorder::RenderD3D11(&remixapi_RenderD3D11, destination, render, describeTarget(destination));
    }
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::RenderD3D11, 1 }; // rtx.profile.detailedLogging
    ApiD3D11Host* host = registeredD3D11();
    if (!host) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    D3D11_TEXTURE2D_DESC desc;
    dxvk::D3D11Texture2D* texture = registeredTexture(*host, destination, desc);
    if (!texture || desc.SampleDesc.Count != 1 || desc.ArraySize != 1 || !(desc.BindFlags & D3D11_BIND_RENDER_TARGET)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    return dxvk::remix_api::renderFrame([host, texture, render]() {
      host->context()->RenderRemix(texture->GetCommonTexture()->GetImage(), render != 0);
    });
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawScreenOverlayD3D11(
      const remixapi_ScreenOverlayLayerD3D11* layers, uint32_t count) {
    REMIX_FORK_RECORD(DrawScreenOverlayD3D11, layers, count);
    return drawScreenOverlayD3D11(layers, count, nullptr);
  }

  // Per-layer alpha cutoff. An optional named export, whose presence is the
  // capability check (see remix_c.h).
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawScreenOverlayD3D11Ex(
      const remixapi_ScreenOverlayLayerD3D11* layers, uint32_t count, const float* alphaCutoffs) {
    REMIX_FORK_RECORD(DrawScreenOverlayD3D11Ex, layers, count, alphaCutoffs);
    return drawScreenOverlayD3D11(layers, count, alphaCutoffs);
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_PrepareD3D11Texture(
      void* texture2D, uint32_t dxgiFormat, remixapi_D3D11TextureToken* out_token) {
    REMIX_FORK_RECORD(PrepareD3D11Texture, texture2D, dxgiFormat, out_token);
    return prepareD3D11Texture(texture2D, dxgiFormat, 0, out_token);
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_PrepareD3D11TextureEx(
      void* texture2D, uint32_t dxgiFormat, uint32_t maxDimension, remixapi_D3D11TextureToken* out_token) {
    REMIX_FORK_RECORD(PrepareD3D11TextureEx, texture2D, dxgiFormat, maxDimension, out_token);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::PrepareD3D11Texture, 1 }; // rtx.profile.detailedLogging
    return prepareD3D11Texture(texture2D, dxgiFormat, maxDimension, out_token);
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_ImportD3D11TextureEx(
      remixapi_D3D11TextureToken token, uint64_t hash, uint32_t encoding, remixapi_TextureHandle* out_handle) {
    REMIX_FORK_RECORD(ImportD3D11TextureEx, token, hash, encoding, out_handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::ImportD3D11Texture, 1 }; // rtx.profile.detailedLogging
    return importD3D11Texture(token, hash, encoding, false, out_handle);
  }

  // Same rules as ImportD3D11TextureEx (share / copy-on-write / copy, mip
  // cap, barriers); the result is applied to the existing entry in place.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_ReplaceD3D11TextureEx(
      remixapi_D3D11TextureToken token, remixapi_TextureHandle handle, uint32_t encoding) {
    REMIX_FORK_RECORD(ReplaceD3D11TextureEx, token, handle, encoding);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::ReplaceD3D11Texture, 1 }; // rtx.profile.detailedLogging
    if (!dxvk::RtxOptions::HostImport::enableInPlaceTextureReplace()) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    remixapi_TextureHandle out = nullptr;
    return importD3D11Texture(token, reinterpret_cast<uint64_t>(handle), encoding, true, &out);
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_ImportD3D11Texture(
      remixapi_D3D11TextureToken token, uint64_t hash, remixapi_TextureHandle* out_handle) {
    REMIX_FORK_RECORD(ImportD3D11Texture, token, hash, out_handle);
    return remixapi_ImportD3D11TextureEx(token, hash, REMIXAPI_TEXTURE_ENCODING_NONE, out_handle);
  }

  REMIXAPI void REMIXAPI_CALL remixapi_ReleaseD3D11Texture(remixapi_D3D11TextureToken token) {
    delete static_cast<PreparedD3D11Texture*>(token);
  }

}
