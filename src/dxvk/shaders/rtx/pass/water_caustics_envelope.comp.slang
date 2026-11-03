// Shapes a source texture once per activation/source change. The full mip chain
// supplies the area mean used by the shared normalized envelope sampler.
layout(binding = 0) Sampler2D Source;
layout(r16f, binding = 1) RWTexture2D<float> Shaped;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID) {
  uint width;
  uint height;
  Shaped.GetDimensions(width, height);
  if (any(pixel >= uint2(width, height))) {
    return;
  }
  const float source = Source.SampleLevel((float2(pixel) + 0.5f) / float2(width, height), 0.0f).r;
  const float value = isfinite(source) ? saturate(source) : 0.0f;
  const float squared = value * value;
  Shaped[pixel] = squared * squared;
}
