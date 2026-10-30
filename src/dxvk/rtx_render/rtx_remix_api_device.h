#pragma once

// The device a Remix API host registered, and the calls between the backend's API entry points (rtx_remix_api.cpp)
// and the API frontend (src/frontends/api/), which owns the API's D3D9 device and the D3D11 host.

#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <remix/remix_c.h>

#include "../dxvk_cs.h"
#include "../dxvk_device.h"

namespace dxvk {

  // A lock on the host device, held until it is destroyed. It keeps the device's own lock object, which is one pointer.
  class RemixApiDeviceLock {
  public:
    template<typename Lock>
    explicit RemixApiDeviceLock(Lock&& lock) {
      using Stored = std::decay_t<Lock>;
      static_assert(sizeof(Stored) <= sizeof(m_storage) && alignof(Stored) <= alignof(void*));
      new (m_storage) Stored(std::move(lock));
      m_release = [](void* storage) { static_cast<Stored*>(storage)->~Stored(); };
    }

    RemixApiDeviceLock(const RemixApiDeviceLock&) = delete;
    RemixApiDeviceLock& operator = (const RemixApiDeviceLock&) = delete;

    ~RemixApiDeviceLock() {
      m_release(m_storage);
    }

  private:
    alignas(void*) unsigned char m_storage[sizeof(void*)];
    void (*m_release)(void*) = nullptr;
  };

  // A command for the host device's command stream, with its type erased so that any host device can push it.
  class RemixApiCsCommand {
  public:
    // Moves the command into the chunk.
    // Returns false when the chunk is full, and then leaves the command as it is.
    virtual bool pushTo(DxvkCsChunkRef& chunk) = 0;

  protected:
    ~RemixApiCsCommand() = default;
  };

  // The device whose command stream the API's scene commands use: the API's own D3D9 device or a D3D11 game's
  // immediate context. Both are in src/frontends/api/.
  class RemixApiDevice {
  public:
    explicit RemixApiDevice(Rc<DxvkDevice> device)
      : m_device(std::move(device)) { }

    virtual ~RemixApiDevice() = default;

    const Rc<DxvkDevice>& GetDXVKDevice() const { return m_device; }

    virtual RemixApiDeviceLock LockDevice() = 0;
    // Waits for preceding CPU command-stream work, without waiting for GPU execution.
    virtual void synchronizeCommands() = 0;

    // AllowFlush lets the device consider a GPU flush when the command fills a chunk.
    template<bool AllowFlush = true, typename Cmd>
    void EmitCs(Cmd&& command) {
      TypedCommand<std::decay_t<Cmd>> typed { command };
      emitCs(typed, AllowFlush);
    }

    // Whether remixapi_Present presents this device's swapchain.
    virtual bool presentsThroughApi() const { return false; }

    // Presents the swapchain, into the override window when it is set.
    virtual remixapi_ErrorCode present(HWND) { return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED; }

    // Resizes the swapchain to the client area of the window it presented into.
    virtual remixapi_ErrorCode fitToWindow(HWND) { return REMIXAPI_ERROR_CODE_SUCCESS; }

    // Releases the host's device at remixapi_Shutdown.
    virtual void shutdown() = 0;

    // Submits the host's initial texture uploads that an import source left unsubmitted (initFlushDeferred).
    // Runs on the CS thread.
    virtual void flushDeferredUploads() { }

  protected:
    virtual void emitCs(RemixApiCsCommand& command, bool allowFlush) = 0;

  private:
    template<typename Cmd>
    class TypedCommand final : public RemixApiCsCommand {
    public:
      explicit TypedCommand(Cmd& command)
        : m_command(command) { }

      bool pushTo(DxvkCsChunkRef& chunk) override {
        return chunk->push(m_command);
      }

    private:
      Cmd& m_command;
    };

    Rc<DxvkDevice> m_device;
  };

  // A host texture prepared for import: a remixapi_PrepareD3D11Texture token. The frontend that prepared it fills the
  // fields and marks its own texture object.
  class RemixApiImportSource {
  public:
    RemixApiImportSource(const RemixApiImportSource&) = delete;
    RemixApiImportSource& operator = (const RemixApiImportSource&) = delete;

    Rc<DxvkDevice> device;
    Rc<DxvkImageView> view;
    // The host can never write the image again, so an import may sample it in place.
    bool immutable = false;
    // The longest side, in texels, of the largest mip to import. 0 means that the host sets no limit.
    uint32_t maxDimension = 0;
    // The host's initial uploads of the image are not submitted yet.
    // The import's command submits them through RemixApiDevice::flushDeferredUploads before it uses the image.
    bool initFlushDeferred = false;
    // The source holds the host's texture object, so an import can mark it with markShared and markImported.
    bool hasHostTexture = false;

    // Makes the next write to the host's texture first give every Remix texture that samples its image a copy
    // (remixHostImportCopyOnWrite).
    virtual void markShared() const = 0;

    // Marks the host's texture as imported in place, for the image statistics.
    virtual void markImported() const = 0;

  protected:
    RemixApiImportSource() = default;
    ~RemixApiImportSource() = default;
  };

  namespace remix_api {

    // Returns the registered host device.
    // Returns null before a device registers and after remixapi_Shutdown.
    RemixApiDevice* registeredDevice();

    // Registers the API's D3D9 device. outputWidth and outputHeight are its swapchain's size, which a recording keeps.
    void registerDevice(std::unique_ptr<RemixApiDevice> device, uint32_t outputWidth, uint32_t outputHeight);

    // Registers the device of a host that keeps its own swapchain and has Remix render into its textures (the D3D11
    // host). create runs under the API mutex, and only when no device is registered.
    // Returns GENERAL_FAILURE when a device is registered already.
    remixapi_ErrorCode registerHostDevice(const std::function<std::unique_ptr<RemixApiDevice>()>& create);

    // Ends the host's Remix frame: instances the persistent lights, runs the bridge's end-scene callback,
    // runs render under the API mutex, then runs the present callback.
    remixapi_ErrorCode renderFrame(const std::function<void()>& render);

    // Replaces the GPU screen overlay's layers with layers. resolve returns the image of a layer's texture2D.
    // It returns null when the texture cannot be an overlay layer, and the call then returns INVALID_ARGUMENTS.
    remixapi_ErrorCode drawScreenOverlay(RemixApiDevice& device, const remixapi_ScreenOverlayLayerD3D11* layers,
                                         uint32_t count, const float* alphaCutoffs,
                                         const std::function<Rc<DxvkImage>(void*)>& resolve);

    // Imports source under hash, or with replaceOnly re-points the live texture hash at it.
    remixapi_ErrorCode importTexture(const RemixApiImportSource* source, uint64_t hash, uint32_t encoding,
                                     bool replaceOnly, remixapi_TextureHandle* out_handle);

    // Queues game-texture registration for hash-bound materials on the host command stream.
    void trackGameTexture(RemixApiDevice& device, Rc<DxvkImageView> view);

    // Starts and stops the VRAM budget's tracking of an import source. hostTexture is the source's reference on the
    // host's texture, or null.
    void trackImportSource(const RemixApiImportSource& source, IUnknown* hostTexture);
    void untrackImportSource(const RemixApiImportSource& source);

    // Applies the startup settings of a host that has the API create its D3D9 device.
    void applyStartupInfo(const remixapi_StartupInfo& info);

    // Queues the copy of a rendering output into dest on the device's command stream.
    // backbuffer runs under the API mutex and returns the image the final colour is copied from.
    remixapi_ErrorCode copyRenderingOutput(RemixApiDevice& device, remixapi_dxvk_CopyRenderingOutputType type,
                                           Rc<DxvkImage> dest, Rc<DxvkImageView> destView,
                                           const std::function<Rc<DxvkImage>()>& backbuffer);

    // Fills the interface slots of the API frontend's entry points: Startup and the D3D9 device's dxvk_ calls.
    // The API frontend (src/frontends/api/) implements it.
    void fillDeviceEntries(remixapi_Interface& interf);

  } // namespace remix_api

} // namespace dxvk
