// Options, shader constants and dev-menu controls for the cheaper visibility rays.
// See rtx_fork_shadow_rays.h.

#include "rtx_fork_shadow_rays.h"

#include <algorithm>

#include "rtx_imgui.h"
#include "rtx/pass/raytrace_args.h"

namespace dxvk {

  void ShadowRays::setRaytraceArgs(RaytraceArgs& args, bool nrdDenoises) {
    // NRD is the only reader of the primary direct hit distance, and it is the only thing first-hit changes.
    args.shadowRayFirstHitPrimaryNee = (firstHitPrimaryNee() && !nrdDenoises) ? 1u : 0u;

    const uint32_t targetResolution = std::clamp(alphaMipTargetResolution(), 4u, 4096u);
    args.shadowRayAlphaGradient = coarseAlphaMip() ? 1.0f / static_cast<float>(targetResolution) : 0.0f;
  }

  void ShadowRays::showImguiSettings() {
    if (RemixGui::CollapsingHeader("Shadow / Visibility Rays", ImGuiTreeNodeFlags_CollapsingHeader)) {
      ImGui::Indent();
      ImGui::PushID("ShadowRays");

      RemixGui::Checkbox("Primary NEE Shadow Ray Stops at First Hit", &firstHitPrimaryNeeObject());
      RemixGui::Checkbox("Coarse Mip for Visibility Alpha Tests", &coarseAlphaMipObject());
      ImGui::BeginDisabled(!coarseAlphaMip());
      RemixGui::DragInt("Alpha Test Mip Resolution (texels)", &alphaMipTargetResolutionObject(), 1.f, 4, 4096);
      ImGui::EndDisabled();

      ImGui::PopID();
      ImGui::Unindent();
    }
  }

} // namespace dxvk
