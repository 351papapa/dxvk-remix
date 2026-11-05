#pragma once

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace dxvk {

  // Parameters live in a uniform buffer, so this cap is not bound by the push constant limit.
  // The parser enforces it and the runtime sizes its buffer from it.
  constexpr uint32_t kMaxRtxExternalEffectParameterValues = 1024;

  // Matches the read binding range 32..47 and the write binding range 48..63.
  // A texture keeps the same index in both ranges.
  constexpr uint32_t kMaxRtxExternalEffectTextures = 16;

  // A 'size' texture is allocated as declared, so this cap guards against huge allocations.
  constexpr uint32_t kMaxRtxExternalEffectTextureExtent = 16384;

  // Stands for the effect's output image wherever a texture index is expected.
  // It has no texture slot, its extent comes from the render target, and the runtime owns it.
  constexpr uint32_t kRtxExternalEffectOutputTexture = 0xFFFFFFFFu;

  enum class RtxExternalEffectDomain {
    HDR,
    Display,
  };

  enum class RtxExternalEffectParameterType {
    Bool,
    Int,
    Float,
    Float2,
    Float3,
    Float4,
    Color3,
    Color4,
  };

  struct RtxExternalEffectParameter {
    std::string id;
    // Display label. Defaults to the title-cased id and is overridden by 'param.<id>.label'.
    std::string name;
    std::string tooltip;
    // Group set by the last 'param.category' before the declaration. Empty when ungrouped.
    std::string category;
    // Combo labels from 'param.<id>.items', valid only on an int parameter.
    // The parser guarantees one entry per representable value.
    std::vector<std::string> items;
    RtxExternalEffectParameterType type = RtxExternalEffectParameterType::Float;
    uint32_t valueOffset = 0;
    uint32_t valueCount = 1;
    std::vector<float> defaultValues;
    std::vector<float> minValues;
    std::vector<float> maxValues;
    float step = 0.01f;
  };

  // Kept separate from VkFormat so the parser and its test need no Vulkan headers.
  // The runtime maps it to a VkFormat.
  enum class RtxExternalEffectTextureFormat {
    R8,
    RG8,
    RGBA8,
    R16F,
    RG16F,
    RGBA16F,
    R32F,
    RG32F,
    RGBA32F,
    R32U,
    RG32U,
    RGBA32U,
    R11G11B10F,
  };

  struct RtxExternalEffectTexture {
    std::string id;
    // Used only by a scratch texture. A file texture takes its format from the image.
    RtxExternalEffectTextureFormat format = RtxExternalEffectTextureFormat::RGBA16F;
    // A non-zero divisor ceil-divides the effect's output extent.
    // Otherwise width and height are the literal extent.
    // Integer divisors are used because nested ceil-divides compose exactly,
    // so a 'div 32' tile texture lines up with a 'div 2' pass's group grid.
    // All three are zero for a file texture, whose extent comes from the image.
    uint32_t divisor = 1;
    uint32_t width = 0;
    uint32_t height = 0;
    // Non-empty makes this a read-only file texture, decoded from disk and uploaded once.
    // The path is stored as written and resolved against the effect file's directory.
    std::string file;
    // Samples the file through the sRGB variant of its format. Defaults off,
    // because the runtime cannot tell colour images from lookup tables.
    bool srgb = false;
    // Samples with VK_SAMPLER_ADDRESS_MODE_REPEAT instead of clamp-to-edge.
    bool repeat = false;
    // Contents are kept across frames, including while the effect is switched off.
    // The shader sees 'historyInvalid' on every frame where they are not valid.
    bool persist = false;
    // Set by the parser from the pass lists, to warn about unused textures.
    bool read = false;
    bool written = false;
  };

  struct RtxExternalEffectPass {
    std::string id;
    std::string entryPoint;
    // Texture index, or kRtxExternalEffectOutputTexture, whose extent sets the dispatch size.
    // Ignored when 'once' is set, which dispatches one workgroup.
    uint32_t over = kRtxExternalEffectOutputTexture;
    bool once = false;
    // Texture indices, possibly including kRtxExternalEffectOutputTexture.
    // They drive the descriptor layout, the barrier access flags, and the
    // check against the module's reflected bindings.
    std::vector<uint32_t> reads;
    std::vector<uint32_t> writes;
  };

  struct RtxExternalEffectManifest {
    std::string id;
    std::string name;
    RtxExternalEffectDomain domain = RtxExternalEffectDomain::Display;
    bool enabledByDefault = false;
    uint32_t parameterValueCount = 0;
    std::vector<RtxExternalEffectParameter> parameters;
    std::vector<RtxExternalEffectTexture> textures;
    // Never empty. For an effect with no pass directives the parser
    // synthesizes 'entry main over output write output'.
    std::vector<RtxExternalEffectPass> passes;
    // Unknown directives are recorded here rather than rejected, so newer files still load.
    // The caller logs them, which keeps the parser free of a Logger.
    std::vector<std::string> warnings;
  };

  // Parses metadata directives from a .remixfx.slang source stream. Directives
  // use the form "//! remixfx dotted.key = value", one per line, with
  // everything after the first '=' taken verbatim as the value. They are
  // intentionally independent from the Slang parser, so discovery and UI state
  // remain available even if shader compilation fails.
  bool parseRtxExternalEffectManifest(
    std::istream& stream,
    const std::string& fallbackId,
    RtxExternalEffectManifest& manifest,
    std::string& error);

  bool isValidRtxExternalEffectId(const std::string& value);

  // Accepts C identifiers only. The parser also uses it for texture, pass and entry point names.
  // Parameter ids are dotted key prefixes ('param.<id>.label'), so they cannot contain '.'.
  // Texture ids become -DREMIXFX_TEX_<id> defines on the slangc command line.
  bool isValidRtxExternalEffectParameterId(const std::string& value);

} // namespace dxvk
