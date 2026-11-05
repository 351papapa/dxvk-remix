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
#include "material_translator.h"

#include <utility>

namespace dxvk {
  namespace translator {

    AlphaStateDesc MaterialTranslator::translateAlphaState(const FixedFunctionState& state, const MaterialData& material) {
      FixedFunctionAlphaInput input;
      input.alphaTestCompareOp = state.alphaTestCompareOp;
      input.alphaTestReferenceValue = state.alphaTestReferenceValue;
      input.blendMode = state.blendMode;
      input.textureAlphaOperation = state.textureAlphaOperation;
      input.textureAlphaArg1Source = state.textureAlphaArg1Source;
      input.textureAlphaArg2Source = state.textureAlphaArg2Source;
      input.tFactor = state.tFactor;
      return deriveAlphaState(input, material, state.categories);
    }

    LegacySurfaceDesc MaterialTranslator::translateLegacySurface(const FixedFunctionState& state) {
      LegacySurfaceDesc desc;
      desc.textureColorArg1Source = state.textureColorArg1Source;
      desc.textureColorArg2Source = state.textureColorArg2Source;
      desc.textureColorOperation = state.textureColorOperation;
      desc.textureAlphaArg1Source = state.textureAlphaArg1Source;
      desc.textureAlphaArg2Source = state.textureAlphaArg2Source;
      desc.textureAlphaOperation = state.textureAlphaOperation;
      desc.tFactor = state.tFactor;
      desc.texgenMode = state.texgenMode;
      desc.isTextureFactorBlend = state.textureFactorBlend;
      desc.isVertexColorBakedLighting = state.vertexColorBakedLighting;
      desc.textureTransform = state.textureTransform;
      desc.enableClipPlane = state.clipPlaneEnabled;
      desc.clipPlane = state.clipPlane;
      return desc;
    }

    MaterialTranslation MaterialTranslator::translate(MaterialData&& material, const FixedFunctionState& state) {
      MaterialTranslation out { std::move(material) };
      out.alphaState = translateAlphaState(state, out.material);
      out.legacySurface = translateLegacySurface(state);
      return out;
    }

  } // namespace translator
} // namespace dxvk
