/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "rtx_frontend_interface.h"
#include "rtx_types.h"

namespace dxvk {
  struct RtLight;
  class SceneManager;
  struct ReplacementBucket;

  // The game lights frontends send each frame the game sets them (frontend::SceneInterface::addGameLights), with the
  // GameMatched lifetime. A light without a USD light replacement goes to LightManager::addGameLight, which matches it
  // to last frame's lights and ages it out. A light with one is keyed here by its hash and position. Its replacement
  // lights are created once and updated at every emission, and they age out like a scene object the frontend stopped
  // drawing (translator::LifetimePolicy, light anti-culling included). A scene clear removes them all.
  class GameLights {
  public:
    explicit GameLights(SceneManager& scene);
    ~GameLights();

    void add(const frontend::FixedFunctionLight* lights, uint32_t count);
    // Ages out the replaced lights the game stopped setting. Later calls in the same frame do nothing.
    void collect();
    // Returns the counters since the last call.
    frontend::GameLightStats takeStats();

  private:
    struct Entry {
      std::shared_ptr<const ReplacementBucket> bucket;
      std::vector<RtLight*> lights;     // one per light replacement, in bucket order (null = not created)
      Matrix4 transform;                // LightUtils::getLightTransform of the game light
      AxisAlignedBoundingBox lightBounds;
      uint32_t frameCreated = 0;
      uint32_t frameLastSeen = 0;
    };

    static void destroy(Entry& entry);

    SceneManager& m_scene;
    std::unordered_map<XXH64_hash_t, Entry> m_entries;
    uint32_t m_collectFrame = ~0u;
    uint64_t m_clearEpoch = ~0ull;
    frontend::GameLightStats m_stats;
  };

} // namespace dxvk
