#pragma once

// A key/value store for plugin-injected runtime state that graph components read by name. It is a header-only
// singleton guarded by an internal mutex. Plugins write it through remixapi_SetGameValue on their own thread, and
// GameValueRead* graph components read it on the render thread.
//
// The store is a function-local static, so it survives Remix Shutdown and re-init, and plugins do not have to
// re-populate their state across device resets.

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace dxvk {
  namespace fork_game_state {

    class GameStateStore {
    public:
      static GameStateStore& get() {
        static GameStateStore s_instance;
        return s_instance;
      }

      // Overwrites the value under key. Thread-safe.
      void set(const std::string& key, std::string value) {
        std::lock_guard<std::mutex> lock{ m_lock };
        m_values[key] = std::move(value);
      }

      // Copies the current value under key into out. Returns false if the
      // key is not present. Thread-safe.
      bool tryGet(const std::string& key, std::string& out) const {
        std::lock_guard<std::mutex> lock{ m_lock };
        auto it = m_values.find(key);
        if (it == m_values.end()) {
          return false;
        }
        out = it->second;
        return true;
      }

    private:
      GameStateStore() = default;
      GameStateStore(const GameStateStore&) = delete;
      GameStateStore& operator=(const GameStateStore&) = delete;

      mutable std::mutex m_lock;
      std::unordered_map<std::string, std::string> m_values;
    };

  }  // namespace fork_game_state
}  // namespace dxvk
