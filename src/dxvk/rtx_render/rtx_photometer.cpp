#include "rtx_photometer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "dxvk_device.h"
#include "rtx_auto_exposure.h"
#include "rtx_context.h"
#include "rtx_host_calibration.h"
#include "rtx_imgui.h"
#include "rtx_pre_exposure.h"

namespace dxvk {

  namespace {
    constexpr uint32_t kSlots = 3;
    constexpr int32_t kWindow = 5;                   // texels averaged around the centre, per axis
    constexpr VkDeviceSize kAlbedoOffset = 256;      // final output: 5 x 5 x 8 bytes, then the fp16 albedo (5 x 5 x 8 bytes)
    constexpr VkDeviceSize kBufferSize = 512;

    struct Slot {
      Rc<DxvkBuffer> buffer;
      bool pending = false;
      float preExposure = 1.0f;  // P when the frame was rendered
    };

    struct Reading {
      bool valid = false;
      float radiance = 0.0f;  // centre luminance in Remix radiance units, pre-exposure divided out
      float albedo = 0.0f;    // centre albedo luminance
    };

    struct State {
      Slot slots[kSlots];
      uint32_t next = 0;
      Reading reading;
    };

    State& state() {
      static State s;
      return s;
    }

    float halfToFloat(uint16_t h) {
      const uint32_t sign = uint32_t(h >> 15) << 31;
      const uint32_t exponent = (h >> 10) & 0x1Fu;
      const uint32_t mantissa = h & 0x3FFu;
      if (exponent == 0u) {
        const float v = std::ldexp(float(mantissa), -24);
        return sign ? -v : v;
      }
      if (exponent == 31u) {
        return mantissa ? NAN : (sign ? -INFINITY : INFINITY);
      }
      const uint32_t bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      return f;
    }

    float luminance(float r, float g, float b) {
      return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }

    void copyCentre(DxvkContext* ctx, const Rc<DxvkBuffer>& buffer, VkDeviceSize offset, const Rc<DxvkImage>& image,
                    const VkExtent3D& extent) {
      VkImageSubresourceLayers subresource = {};
      subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.layerCount = 1;
      const int32_t x = std::max(int32_t(extent.width / 2) - kWindow / 2, 0);
      const int32_t y = std::max(int32_t(extent.height / 2) - kWindow / 2, 0);
      ctx->copyImageToBuffer(buffer, offset, 0, 0, image, subresource, VkOffset3D { x, y, 0 },
                             VkExtent3D { uint32_t(kWindow), uint32_t(kWindow), 1 });
    }

    void collect() {
      State& s = state();
      for (Slot& slot : s.slots) {
        if (!slot.pending || slot.buffer == nullptr || slot.buffer->isInUse(DxvkAccess::Read)) {
          continue;
        }
        slot.pending = false;
        const uint8_t* data = static_cast<const uint8_t*>(slot.buffer->mapPtr(0));
        double radiance = 0.0, albedo = 0.0;
        for (int32_t i = 0; i < kWindow * kWindow; ++i) {
          uint16_t h[4];
          std::memcpy(h, data + i * 8, sizeof(h));
          radiance += luminance(halfToFloat(h[0]), halfToFloat(h[1]), halfToFloat(h[2]));
          uint16_t a[4];
          std::memcpy(a, data + kAlbedoOffset + i * 8, sizeof(a));
          albedo += luminance(halfToFloat(a[0]), halfToFloat(a[1]), halfToFloat(a[2]));
        }
        const double n = double(kWindow * kWindow);
        s.reading.valid = std::isfinite(radiance);
        s.reading.radiance = float(radiance / n) / std::max(slot.preExposure, 1e-30f);
        s.reading.albedo = float(albedo / n);
      }
    }
  }

  void Photometer::queueReadback(DxvkContext* ctx, const Resources::RaytracingOutput& rtOutput) {
    if (!enable()) {
      return;
    }
    State& s = state();
    Slot& slot = s.slots[s.next];
    if (slot.pending) {
      return;  // all slots in flight: skip a frame rather than wait
    }
    if (slot.buffer == nullptr) {
      DxvkBufferCreateInfo info;
      info.size = kBufferSize;
      info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
      slot.buffer = ctx->getDevice()->createBuffer(info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "Photometer readback");
    }
    const Rc<DxvkImage> finalOutput = rtOutput.m_finalOutput.image(Resources::AccessType::Read);
    if (finalOutput == nullptr || !rtOutput.m_primaryAlbedo.isValid()) {
      return;
    }
    copyCentre(ctx, slot.buffer, 0, finalOutput, rtOutput.m_finalOutputExtent);
    copyCentre(ctx, slot.buffer, kAlbedoOffset, rtOutput.m_primaryAlbedo.image, rtOutput.m_primaryAlbedo.image->info().extent);
    slot.preExposure = PreExposure::get();
    slot.pending = true;
    s.next = (s.next + 1) % kSlots;
  }

  void Photometer::showImguiSettings(float luxPerUnit) {
    collect();
    RemixGui::Checkbox("Photometer (screen centre)", &enableObject());
    if (enable()) {
      const Reading& r = state().reading;
      if (!r.valid) {
        ImGui::TextDisabled("waiting for a reading");
      } else if (luxPerUnit > 0.0f) {
        // NUMOS's illuminance unit in lux is also its radiance unit in cd/m^2 (E = pi L for a diffuse surface
        // on both scales).
        const float candela = r.radiance * luxPerUnit;
        const float lux = 3.14159265f * candela / std::max(r.albedo, 0.05f);
        ImGui::Text("Luminance   %.4g cd/m^2", double(candela));
        ImGui::Text("Illuminance %.4g lux (diffuse, albedo %.2f)", double(lux), double(r.albedo));
        ImGui::Text("Scene EV100 %.2f", double(std::log2(std::max(candela, 1e-12f) * 100.0f / 12.5f)));
      } else {
        ImGui::Text("Luminance %.4g Remix units (no sky provider calibration)", double(r.radiance));
      }
      ImGui::Text("Exposure    %.2f stops (pre-exposure 2^%.0f)",
                  double(std::log2(std::max(PreExposure::trackedExposure(), 1e-30f))),
                  double(std::log2(PreExposure::get())));
    }
    RemixGui::DragFloat("Host lights (stops)", &HostCalibration::lightsEvObject(), 0.05f, -24.0f, 24.0f, "%.2f");
    RemixGui::SetTooltipToLastWidgetOnHover("rtx.hostCalibration.lightsEv: converts the game's light units to physical.");
    RemixGui::DragFloat("Host emissives vs lights (stops)", &HostCalibration::emissiveOffsetEvObject(), 0.05f, -24.0f, 24.0f, "%.2f");
    RemixGui::SetTooltipToLastWidgetOnHover("rtx.hostCalibration.emissiveOffsetEv: emissives are scaled by 2^(lightsEv + this).");
  }

}
