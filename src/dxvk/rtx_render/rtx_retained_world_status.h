#pragma once

namespace dxvk {
  class DxvkDevice;

  void publishRetainedWorldStatus(DxvkDevice* device, bool hardwareReady);
}
