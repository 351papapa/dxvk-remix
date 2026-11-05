// The fork_hooks functions of the overlay subsystem (see rtx_fork_hooks.h).
// overlayInputForward reads GameOverlay::m_hwnd and dispatchScreenOverlay reads RtxContext's
// screen-overlay state, so both are friends of those classes (rtx_overlay_window.h, rtx_context.h).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>                 // GET_X_LPARAM / GET_Y_LPARAM for mouse coord translation
#include "rtx_fork_hooks.h"

#include "rtx_overlay_window.h"       // GameOverlay
#include "imgui/imgui_impl_win32.h"   // ImGui_ImplWin32_WndProcHandler
#include "imgui/imgui.h"              // ImGui::SetCurrentContext (imguiContextPin)
#include "imgui/implot.h"             // ImPlot::SetCurrentContext (imguiContextPin)
#include "imgui/imgui_remix_exports.h" // remixapi_imgui_InvokeDrawCallback (wrapperTabDraw)

#include "rtx_context.h"              // RtxContext
#include "rtx_resources.h"            // Resources::RaytracingOutput
#include "rtx_shader_manager.h"       // ManagedShader, SHADER_SOURCE, PUSH_CONSTANTS macros
#include "rtx/pass/screen_overlay/screen_overlay.h"
#include <rtx_shaders/screen_overlay.h>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace dxvk {

  namespace {
    // Alpha-composites the plugin-uploaded RGBA overlay over the final tone-mapped output image.
    class ScreenOverlayShader : public ManagedShader {
      SHADER_SOURCE(ScreenOverlayShader, VK_SHADER_STAGE_COMPUTE_BIT, screen_overlay)

      PUSH_CONSTANTS(ScreenOverlayArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE2D(SCREEN_OVERLAY_INPUT_OUTPUT)
        SAMPLER2D(SCREEN_OVERLAY_TEXTURE)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ScreenOverlayShader);
  } // anonymous namespace

namespace fork_hooks {

  // Forwards keyboard and mouse messages from the game window procedure (Path A) to ImGui's Win32 backend.
  // It covers the cases where overlayWndProc stops receiving raw input (Path B), such as a game menu
  // capturing raw input or a plugin HUD pulling focus. Every message is dropped while raw input is
  // recent, so the two paths never feed ImGui twice.
  // Mouse button and move messages carry game-client coordinates, which are translated to overlay-client
  // coordinates when the overlay has its own HWND. Wheel messages carry screen coordinates.
  void overlayInputForward(
      GameOverlay& overlay, HWND gameHwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    HWND overlayHwnd = overlay.m_hwnd.load();
    HWND targetHwnd = overlayHwnd ? overlayHwnd : gameHwnd;

    switch (msg) {
    // Keyboard, gated like the mouse below. Path B also delivers keyboard input: the ImGui Win32 backend
    // handles WM_INPUT for RIM_TYPEKEYBOARD (keys and text), and RIDEV_NOLEGACY only suppresses the
    // overlay's legacy WM_KEY* messages. Forwarding while raw input is recent would double every key.
    case WM_KEYDOWN:
    case WM_KEYUP:
    case WM_SYSKEYDOWN:
    case WM_SYSKEYUP:
    case WM_CHAR:
    case WM_SYSCHAR:
      if (overlay.isRawInputRecent()) {
        break;
      }
      // lParam has no coordinates, pass through as-is.
      ImGui_ImplWin32_WndProcHandler(targetHwnd, msg, wParam, lParam);
      break;

    // Mouse motion and buttons. Path B uses ImGui display-space coordinates and Path A unscaled
    // overlay-client ones, so both at once would make the cursor jitter between two positions.
    // Path B stamps its recency on every WM_INPUT, and Path A forwards only after kRawInputRecencyMs
    // (100 ms) without one.
    case WM_MOUSEMOVE:
    case WM_LBUTTONDOWN: case WM_LBUTTONUP: case WM_LBUTTONDBLCLK:
    case WM_RBUTTONDOWN: case WM_RBUTTONUP: case WM_RBUTTONDBLCLK:
    case WM_MBUTTONDOWN: case WM_MBUTTONUP: case WM_MBUTTONDBLCLK:
    case WM_XBUTTONDOWN: case WM_XBUTTONUP: case WM_XBUTTONDBLCLK:
    {
      if (overlay.isRawInputRecent()) {
        break;
      }
      LPARAM translated = lParam;
      if (overlayHwnd && overlayHwnd != gameHwnd) {
        POINT pt = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
        ClientToScreen(gameHwnd, &pt);
        ScreenToClient(overlayHwnd, &pt);
        translated = MAKELPARAM(pt.x, pt.y);
      }
      ImGui_ImplWin32_WndProcHandler(targetHwnd, msg, wParam, translated);
      break;
    }

    // Wheel: gated together with mouse motion/buttons (same Path A/B race).
    case WM_MOUSEWHEEL:
    case WM_MOUSEHWHEEL:
      if (overlay.isRawInputRecent()) {
        break;
      }
      ImGui_ImplWin32_WndProcHandler(targetHwnd, msg, wParam, lParam);
      break;

    default:
      break;
    }
  }

  // Alpha-composites the host overlays over the final tone-mapped output image: first the CPU pixel buffer
  // from remixapi_DrawScreenOverlay (if one was queued this frame), then the GPU layers from
  // remixapi_DrawScreenOverlayD3D11, bottom-first. Runs after tone mapping and before screenshot capture.
  // The CPU overlay image is recreated when its size or format changes. The GPU layers were copied into
  // their slot images when they were set, and persist until the host replaces or clears them.
  void dispatchScreenOverlay(RtxContext& ctx, Resources::RaytracingOutput& rtOutput) {
    if (!ctx.m_pendingScreenOverlay.has_value() && ctx.m_gpuScreenOverlayCount == 0) {
      return;
    }

    ScopedGpuProfileZone(&ctx, "Screen Overlay");

    if (ctx.m_pendingScreenOverlay.has_value()) {
      auto& overlay = *ctx.m_pendingScreenOverlay;

      // Recreate overlay image if dimensions or format changed.
      if (ctx.m_screenOverlayWidth  != overlay.width
       || ctx.m_screenOverlayHeight != overlay.height
       || ctx.m_screenOverlayFormat != overlay.format
       || !ctx.m_screenOverlayImage.ptr()) {
        DxvkImageCreateInfo imageInfo = {};
        imageInfo.type        = VK_IMAGE_TYPE_2D;
        imageInfo.format      = overlay.format;
        imageInfo.flags       = 0;
        imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.extent      = { overlay.width, overlay.height, 1 };
        imageInfo.numLayers   = 1;
        imageInfo.mipLevels   = 1;
        imageInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        imageInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.layout      = VK_IMAGE_LAYOUT_UNDEFINED;

        ctx.m_screenOverlayImage = ctx.m_device->createImage(
          imageInfo,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          DxvkMemoryStats::Category::RTXRenderTarget,
          "Screen overlay image");

        DxvkImageViewCreateInfo viewInfo = {};
        viewInfo.type      = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format    = overlay.format;
        viewInfo.usage     = VK_IMAGE_USAGE_SAMPLED_BIT;
        viewInfo.aspect    = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.minLevel  = 0;
        viewInfo.numLevels = 1;
        viewInfo.minLayer  = 0;
        viewInfo.numLayers = 1;

        ctx.m_screenOverlayView = ctx.m_device->createImageView(ctx.m_screenOverlayImage, viewInfo);

        ctx.m_screenOverlayWidth  = overlay.width;
        ctx.m_screenOverlayHeight = overlay.height;
        ctx.m_screenOverlayFormat = overlay.format;
      }

      // Copy staging buffer to overlay image
      VkImageSubresourceLayers subresource = {};
      subresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.mipLevel       = 0;
      subresource.baseArrayLayer = 0;
      subresource.layerCount     = 1;

      VkOffset3D offset = { 0, 0, 0 };
      VkExtent3D extent = { overlay.width, overlay.height, 1 };

      ctx.copyBufferToImage(ctx.m_screenOverlayImage, subresource, offset, extent,
                            overlay.stagingBuffer, 0, 0, 0);
    }

    // Dispatch overlay blend compute shader, once per layer.
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);

    auto& finalOutput         = rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite);
    const VkExtent3D outputSize = finalOutput.image->info().extent;
    const VkExtent3D workgroups = util::computeBlockCount(
      outputSize, VkExtent3D { SCREEN_OVERLAY_TILE_SIZE, SCREEN_OVERLAY_TILE_SIZE, 1 });
    const Rc<DxvkSampler> sampler = ctx.getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    auto blend = [&](const Rc<DxvkImageView>& view, float opacity, uint32_t flags, const float* rect, float alphaCutoff) {
      ScreenOverlayArgs pushArgs = {};
      pushArgs.imageSize = { outputSize.width, outputSize.height };
      pushArgs.opacity   = opacity;
      pushArgs.flags     = flags;
      pushArgs.rectX     = rect[0];
      pushArgs.rectY     = rect[1];
      pushArgs.rectW     = rect[2];
      pushArgs.rectH     = rect[3];
      pushArgs.alphaCutoff = alphaCutoff;
      ctx.pushConstants(0, sizeof(pushArgs), &pushArgs);

      ctx.bindResourceView(SCREEN_OVERLAY_INPUT_OUTPUT, finalOutput.view, nullptr);
      ctx.bindResourceView(SCREEN_OVERLAY_TEXTURE, view, nullptr);
      ctx.bindResourceSampler(SCREEN_OVERLAY_TEXTURE, sampler);

      ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ScreenOverlayShader::getShader());
      ctx.dispatch(workgroups.width, workgroups.height, workgroups.depth);
    };

    if (ctx.m_pendingScreenOverlay.has_value()) {
      static const float kFullRect[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
      blend(ctx.m_screenOverlayView, ctx.m_pendingScreenOverlay->opacity, 0, kFullRect, 0.0f);
      // Clear pending overlay after dispatch
      ctx.m_pendingScreenOverlay.reset();
    }

    for (uint32_t i = 0; i < ctx.m_gpuScreenOverlayCount; ++i) {
      const auto& slot = ctx.m_gpuScreenOverlay[i];
      if (slot.view != nullptr) {
        blend(slot.view, slot.opacity, slot.flags, slot.rect, slot.alphaCutoff);
      }
    }
  }

  // Pins the ImGui and ImPlot context pointers to the dev menu's contexts at wndProcHandler entry.
  // Plugin-side rendering on other threads can move GImGui off the dev menu's context between frames,
  // and keyboard input would then go to the wrong (or null) context.
  void imguiContextPin(ImGuiContext* ctx, ImPlotContext* plotCtx) {
    ImGui::SetCurrentContext(ctx);
    ImPlot::SetCurrentContext(plotCtx);
  }

  // Invokes the registered plugin ImGui draw callback for the dev menu's Plugin tab (kTab_Wrapper).
  void wrapperTabDraw() {
    remixapi_imgui_InvokeDrawCallback();
  }

} // namespace fork_hooks

  // Runs on the command-stream thread (queued by remixapi_DrawScreenOverlayD3D11), so each copy is recorded
  // after every draw the host issued to the source before the call. A slot's image is only recreated when
  // the source's size or format changes.
  void RtxContext::setGpuScreenOverlayLayers(std::vector<GpuScreenOverlaySource>&& layers) {
    const uint32_t count = std::min<uint32_t>(
      static_cast<uint32_t>(layers.size()), kMaxGpuScreenOverlayLayers);

    if (count == 0) {
      for (auto& slot : m_gpuScreenOverlay) {
        slot = GpuScreenOverlaySlot {};
      }
      m_gpuScreenOverlayCount = 0;
      return;
    }

    for (uint32_t i = 0; i < count; ++i) {
      const auto& source = layers[i];
      auto& slot = m_gpuScreenOverlay[i];
      const VkFormat format = source.image->info().format;
      const VkExtent3D extent = source.image->mipLevelExtent(0);

      if (slot.image == nullptr
       || slot.image->info().format != format
       || slot.image->info().extent.width != extent.width
       || slot.image->info().extent.height != extent.height) {
        DxvkImageCreateInfo imageInfo = {};
        imageInfo.type        = VK_IMAGE_TYPE_2D;
        imageInfo.format      = format;
        imageInfo.flags       = 0;
        imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.extent      = { extent.width, extent.height, 1 };
        imageInfo.numLayers   = 1;
        imageInfo.mipLevels   = 1;
        imageInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        imageInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        slot.image = m_device->createImage(
          imageInfo,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          DxvkMemoryStats::Category::RTXRenderTarget,
          "Screen overlay GPU layer");

        DxvkImageViewCreateInfo viewInfo = {};
        viewInfo.type      = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format    = format;
        viewInfo.usage     = VK_IMAGE_USAGE_SAMPLED_BIT;
        viewInfo.aspect    = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.minLevel  = 0;
        viewInfo.numLevels = 1;
        viewInfo.minLayer  = 0;
        viewInfo.numLayers = 1;

        slot.view = m_device->createImageView(slot.image, viewInfo);
      }

      const VkImageSubresourceLayers mip0 = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      copyImage(slot.image, mip0, VkOffset3D { 0, 0, 0 },
                source.image, mip0, VkOffset3D { 0, 0, 0 },
                VkExtent3D { extent.width, extent.height, 1 });

      slot.flags   = source.flags;
      slot.opacity = source.opacity;
      slot.alphaCutoff = source.alphaCutoff;
      std::copy(std::begin(source.rect), std::end(source.rect), std::begin(slot.rect));
    }
    m_gpuScreenOverlayCount = count;
  }

} // namespace dxvk
