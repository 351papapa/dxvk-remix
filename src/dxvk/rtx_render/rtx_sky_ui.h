#pragma once

// The Sky page of the Remix developer menu (Game Setup > Parameter Tuning > Sky). Owns the sky
// mode switch, the rasterized skybox's controls and sky detection; the sky provider adds its own pages
// while rtx.skyMode selects it.

namespace dxvk {
  class DxvkContext;

  void showSkyImguiSettings(DxvkContext* ctx);
}
