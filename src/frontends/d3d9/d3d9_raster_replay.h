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

// The D3D9 frontend's raster replays. A D3D9 draw is replayed through
// frontend::RasterService into the sky matte and probe or into the terrain cascades. For each pass the callback
// overrides the draw's D3D9 constants, specialization constants and textures, and issues the draw on the CS thread's
// context. The draw's state is restored after the replay.

#include <cstdint>

namespace dxvk {
  class RtxContext;
  struct DrawCallState;
  struct MaterialData;

  // A D3D9 draw's counts, which the CS thread issues again for each replay pass.
  struct DrawParameters {
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t instanceCount = 0;
    uint32_t firstIndex = 0;
    uint32_t vertexOffset = 0;
  };

  namespace frontends {
    namespace d3d9 {

      // Replays a sky-camera draw into the sky matte and probe. Does nothing while the backend renders its own sky.
      void replaySky(RtxContext& ctx, const DrawParameters& params, const DrawCallState& state);

      struct TerrainReplay {
        // The baked material the draw is shaded with, or null when nothing was baked.
        const MaterialData* material = nullptr;
        // The cascade set changed this frame, so the material is described again.
        bool cascadesChanged = false;
        uint32_t channelPasses = 0;  // passes of a material channel other than albedo
      };

      // Replays a Terrain-category draw into the bake cascades. The draw then samples the cascades: its texture
      // transform, texture coordinate generation and legacy material are the bake's.
      TerrainReplay replayTerrain(RtxContext& ctx, const DrawParameters& params, DrawCallState& state);

    } // namespace d3d9
  } // namespace frontends
} // namespace dxvk
