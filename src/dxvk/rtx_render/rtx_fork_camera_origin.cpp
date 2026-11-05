#include "rtx_fork_camera_origin.h"

// Ported from BetaRT (sambow23/dxvk-remix-gmod, commit 11761453a). See the header.

#include "rtx_fork_game_state.h"
#include "rtx_imgui.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <string>

namespace dxvk {

  void WorldOrigin::showImguiSettings() {
    RemixGui::Checkbox("Host World Origin", &enableObject());
    RemixGui::Checkbox("Precise API Camera (double)", &preciseCameraObject());
  }

  namespace fork_camera_origin {

    bool tryParseWorldOriginComponent(const std::string& raw, float& outValue) {
      const char* const begin = raw.c_str();
      char* end = nullptr;
      errno = 0;
      const float parsed = std::strtof(begin, &end);
      if (end == begin || errno == ERANGE) {
        return false;
      }

      while (*end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) {
          return false;
        }
        ++end;
      }

      outValue = parsed;
      return true;
    }

    bool tryReadWorldOriginOffsetFromGameState(Vector3& outOrigin) {
      std::string rawX;
      std::string rawY;
      std::string rawZ;
      auto& gameState = fork_game_state::GameStateStore::get();
      if (!gameState.tryGet(kWorldOriginGameValueX, rawX)
          || !gameState.tryGet(kWorldOriginGameValueY, rawY)
          || !gameState.tryGet(kWorldOriginGameValueZ, rawZ)) {
        return false;
      }

      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      if (!tryParseWorldOriginComponent(rawX, x)
          || !tryParseWorldOriginComponent(rawY, y)
          || !tryParseWorldOriginComponent(rawZ, z)) {
        return false;
      }

      outOrigin = Vector3(x, y, z);
      return true;
    }

    Vector3 calculatePreviousCameraHistoryOffset(
      bool hasPreviousOrigin,
      const Vector3& previousOrigin,
      const Vector3& currentOrigin) {
      if (!hasPreviousOrigin) {
        return Vector3(0.0f);
      }

      return previousOrigin - currentOrigin;
    }

  } // namespace fork_camera_origin
} // namespace dxvk
