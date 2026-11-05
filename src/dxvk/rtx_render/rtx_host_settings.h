#pragma once

// Host settings registry.
//
// A host (game plugin) registers its options through the remixapi_*HostSetting* exports, and the dev menu's
// "Host Settings" tab draws them as live controls. The values live here. The host polls a generation counter
// and re-reads only when the menu changed something.
//
// Every entry point takes the registry mutex. The host calls from its game and render threads, and the menu
// draws on the runtime's thread.

#include <cstdint>

struct remixapi_HostSettingInfo;

namespace dxvk::host_settings {

  // Returns the handle (1-based) or 0 on invalid input. Registering a key that
  // already exists updates its metadata and value and keeps its handle.
  uint32_t registerSetting(const remixapi_HostSettingInfo& info);

  bool getValue(uint32_t handle, double& out);
  bool getValueByKey(const char* key, double& out);
  // Host-side write. Does not bump the generation (the host already knows).
  bool setValue(uint32_t handle, double value);
  void clear();

  // Bumped each time the menu changes a value.
  uint64_t generation();

  bool empty();

  // Draws the tab contents. Called from ImGUI::showMainMenu.
  void showImguiTab();

  // A setting carries REMIXAPI_HOST_SETTING_FLAG_USER_MENU.
  bool hasUserMenuSettings();

  // Draws the settings that carry REMIXAPI_HOST_SETTING_FLAG_USER_MENU, in registration order. Called from the
  // user menu.
  void showUserMenuSettings();

}
