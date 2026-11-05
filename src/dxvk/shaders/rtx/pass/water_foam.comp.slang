// Dedicated optional pass: no foam code is added to the general material variants.
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/water_foam.h"
#include "rtx/concept/camera/camera.slangh"
#include "rtx/algorithm/sky_provider.slangh"
#include "rtx/concept/light/light.slangh"
layout(push_constant) ConstantBuffer<WaterFoamArgs> args;
layout(binding = WATER_FOAM_COLOR) RWTexture2D<float4> Color;
layout(binding = WATER_FOAM_FIELD) Texture2D<float4> Field;
layout(binding = WATER_FOAM_PATTERN) Sampler2D Pattern;
layout(binding = BINDING_WATER_FOAM_SURFACE_HEIGHT) Texture2D<float4> CarrierHeight;
layout(binding = WATER_FOAM_SKY) SamplerCube<float3> FoamSkyProbe;
layout(binding = WATER_FOAM_ENVIRONMENT_SAMPLER) SamplerState FoamEnvironmentSampler;

#include "rtx/pass/water_foam_common.slangh"
#include "rtx/pass/water_foam_sentinel.slangh"

bool foamLightVisible(float3 origin, float3 direction) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = direction;
  ray.TMin = 0.02f;
  ray.TMax = 1000.0f;
  const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
    | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE, ray);
  while (query.Proceed()) {
    if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) { continue; }
    uint index = (query.CandidateInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CandidateGeometryIndex();
    // The light receiver is above the water. Water triangles must not turn a
    // numerical self-intersection into a completely black environment sample.
    if (!surfaces[index].isAnimatedWater) { query.CommitNonOpaqueTriangleHit(); }
  }
  return query.CommittedStatus() != COMMITTED_TRIANGLE_HIT;
}

float3 foamRasterSkyLighting(float3 position, float3 normal, float3 up) {
  float3 origin = position + normal * 0.06f + up * 0.02f;
  float3 tangent, bitangent;
  calcOrthonormalBasis(normal, tangent, bitangent);
  float3 ambient = float3(0.0f);
  // Deterministic cosine-weighted hemisphere quadrature: Lambertian outgoing
  // radiance is the average incident radiance, with no additional division by pi.
  // Sample the same environment and brightness convention as indirect paths.
  [unroll] for (uint i = 0u; i < 8u; ++i) {
    float radius = sqrt((float(i) + 0.5f) / 8.0f);
    float angle = float(i) * 2.39996323f;
    float3 direction = tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle))
      + normal * sqrt(1.0f - radius * radius);
    float3 radiance = cb.domeLightArgs.active
      ? cb.domeLightArgs.radiance * sampleDomeLightTexture(FoamEnvironmentSampler, direction,
          cb.domeLightArgs.textureIndex, cb.domeLightArgs.worldToLightTransform)
      : cb.skyBrightness * FoamSkyProbe.SampleLevel(direction, 0.0f);
    if (all(isfinite(radiance)) && any(radiance > 0.0f) && foamLightVisible(origin, direction)) {
      ambient += max(radiance, float3(0.0f)) * (1.0f / 8.0f);
    }
  }
  float3 direct = float3(0.0f);
  // The renderer already groups lights by type. A beach pixel only scans its
  // directional range, never the hundreds of local lights in the whole scene.
  const LightRangeInfo range = cb.lightRanges[lightTypeDistant];
  const uint end = min(uint(range.offset) + uint(range.count), uint(cb.lightCount));
  [loop] for (uint index = uint(range.offset); index < end; ++index) {
    DecodedPolymorphicLight decoded = decodePolymorphicLight(lights[index]);
    if (uint(decoded.polymorphicType) != lightTypeDistant || !all(isfinite(decoded.radiance))) { continue; }
    DistantLight light = distantLightCreate(decoded);
    float directionLength = length(light.direction);
    if (directionLength <= 0.0001f || !isfinite(directionLength) || !isfinite(light.cosHalfAngle)) { continue; }
    float3 toLight = -light.direction / directionLength;
    float cosine = saturate(dot(normal, toLight));
    if (cosine <= 0.0f || !foamLightVisible(origin, toLight)) { continue; }
    // Match distantLightSampleArea: radiance/sin^2(theta), integrated over
    // 2*pi*(1-cos(theta)), then the Lambertian 1/pi. Small-angle limit is 1.
    float angularNormalization = 2.0f / max(1.0f + light.cosHalfAngle, 0.0001f);
    direct += max(light.radiance, float3(0.0f)) * (angularNormalization * cosine);
  }
  return direct + ambient * args.skyAmbient;
}

// decides how bright "normally lit" foam is.
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID) {
  if (pixel.x >= args.width || pixel.y >= args.height) { return; }
  float3 up = cb.isZUp != 0 ? float3(0, 0, 1) : float3(0, 1, 0);
  float2 uv = (float2(pixel) + 0.5f) / float2(args.width, args.height);
  CameraDirections direction = cameraScreenUVToDirection(cb.camera, uv, false);
  RayDesc ray;
  ray.Origin = cameraGetWorldPosition(cb.camera); ray.Direction = direction.worldDirection;
  ray.TMin = 0.02f; ray.TMax = args.maxDistance;
  const uint flags = RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<flags> query;
  query.TraceRayInline(topLevelAS, flags, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, ray);
  while (query.Proceed()) { }
  if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return; }
  uint id = (query.CommittedInstanceID() & CUSTOM_INDEX_SURFACE_MASK) + query.CommittedGeometryIndex();
  const Surface surface = surfaces[id];
  const bool hitWater = surface.isAnimatedWater;
  if (!hitWater || dot(direction.worldDirection, up) >= -0.005f
      || (args.oceanOnly != 0 && !surface.isOceanWater)) { return; }
  float waterDistance = query.CommittedRayT();
  float3 p = ray.Origin + waterDistance * ray.Direction;
  float2 world = cb.isZUp != 0 ? p.xy : p.xz;
  float2 grid = (world - float2(args.originX, args.originY)) / args.spacing;
  if (any(grid < 1) || any(grid >= float(args.fieldSize - 2))) { return; }
  int2 cell = int2(floor(grid)); float2 f = frac(grid);
  const int2 offsets[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
  const float weights[4] = { (1.0f-f.x)*(1.0f-f.y), f.x*(1.0f-f.y), (1.0f-f.x)*f.y, f.x*f.y };
  FoamCarrier carriers[4];
  float support = 0.0f, storedDensity = 0.0f, storedHeight = 0.0f, heightSupport = 0.0f;
  float2 flow = float2(0);
  [unroll] for (uint i = 0; i < 4; ++i) {
    const int3 tap = int3(cell + offsets[i], 0);
    const float4 state = Field.Load(tap);
    const float4 heights = CarrierHeight.Load(tap);
    carriers[i] = foamSelectCarrier(heights, state);
    // Interpolate the underlying surface height across the waterline as well:
    // discarding wet terrain corners would shift the matched shoreline uphill.
    const float2 matchedHeight = heights.xy;
    if (matchedHeight.y >= 0.5f && isfinite(matchedHeight.x)) {
      storedHeight += matchedHeight.x * weights[i];
      heightSupport += weights[i];
    }
    if (!carriers[i].valid || carriers[i].water != hitWater) { continue; }
    support += weights[i];
    storedDensity += carriers[i].density * weights[i];
    flow += state.yz * weights[i];
  }
  if (support <= 0.0001f || heightSupport <= 0.0001f || storedDensity <= 0.0f) { return; }
  storedDensity /= support;
  storedHeight /= heightSupport;
  flow /= support;
  // Height matching prevents foam stored below a pier or on a different water
  // level from painting the first visible surface above it.
  if (abs(dot(p, up) - storedHeight) > max(0.03f, args.spacing * 0.25f)) { return; }
  float3 carrierNormal = up;
  if (carriers[0].valid && carriers[1].valid && carriers[2].valid && carriers[3].valid
      && foamCarrierContinuous(carriers[0].height, carriers[1].height, args.spacing)
      && foamCarrierContinuous(carriers[0].height, carriers[2].height, args.spacing)
      && foamCarrierContinuous(carriers[1].height, carriers[3].height, args.spacing)
      && foamCarrierContinuous(carriers[2].height, carriers[3].height, args.spacing)) {
    const float dx = lerp(carriers[1].height - carriers[0].height, carriers[3].height - carriers[2].height, f.y) / args.spacing;
    const float dy = lerp(carriers[2].height - carriers[0].height, carriers[3].height - carriers[1].height, f.x) / args.spacing;
    carrierNormal = normalize(cb.isZUp != 0 ? float3(-dx, -dy, 1) : float3(-dx, 1, -dy));
  }
  float time = args.time;
  float scale = max(args.textureScale, 0.1f);
  float detailScale = 5.0f / scale;
  float noise = foamNoise(world * (0.42f * detailScale) + float2(time * 0.13f, -time * 0.09f)).x;
  float fine = foamNoise(world * (1.1f * detailScale) + float2(-time * 0.21f, time * 0.16f)).x;
  float density = 1.0f - exp(-storedDensity * 2.8f);
  float erosion = saturate((density - (noise * 0.38f + fine * 0.18f)) / 0.5f);
  float2 patternUV = world / scale + float2(-0.045f, 0.035f) * time
    - flow * 0.14f + (float2(noise, fine) - 0.5f) * 0.22f;
  float lod = max(0.0f, log2(max(1.0f, waterDistance * 512.0f / (float(args.height) * scale))));
  float pattern = Pattern.SampleLevel(patternUV, lod).a;
  // Support fades incompatible water layers and the shallow shoreline edge.
  float edge = min(min(grid.x, grid.y), min(float(args.fieldSize - 1) - grid.x, float(args.fieldSize - 1) - grid.y));
  float alpha = saturate(density * erosion * pattern * args.strength) * support * smoothstep(1.0f, 9.0f, edge);
  alpha *= 1.0f - smoothstep(args.maxDistance * 0.8f, args.maxDistance, waterDistance);
  if (alpha <= 0.001f) { return; }
  // Diffuse illumination: sun plus sky ambient. Foam previously carried its own
  // light (a fixed float3(0.006f) ambient and a visibility-independent sun floor),
  // so it did not follow the environment at all.
  //
  // Numos limitation: its ambient below is sky *radiance*
  // looked up by direction, which is position-independent. It follows time of day,
  // weather and cloud cover, but it does NOT know about local occlusion, so a foam
  // patch under a bridge or indoors is not automatically darker than one in the
  // open. Doing that properly needs either a sky-visibility ray per pixel (extra
  // cost on every foam pixel) or routing the foam through the material shading
  // path as real geometry (the WaterLab plugin's own foam mesh works that way and
  // therefore does get full ambient). Until then this is an approximation, not GI.
  float3 light = float3(0.0f);
  if (cb.skyMode == 1) {
    const vec3 zenithWorld = up;
    const vec3 horizonWorld = normalize(cb.isZUp != 0 ? vec3(0, 1, 0.05f) : vec3(0, 0.05f, 1));
    const vec3 skyZenith = skyProviderAmbient(zenithWorld) * cb.preExposure;
    const vec3 skyHorizon = skyProviderAmbient(horizonWorld) * cb.preExposure;
    // args.skyAmbient multiplies the ambient directly, so 0 genuinely removes it
    // and restores the pre-change look rather than leaving a hidden floor.
    float3 ambient = (skyZenith + skyHorizon) * 0.5f * args.skyAmbient;
    float3 sun = skyProviderSunDirection();
    float elevation = saturate(dot(sun, carrierNormal));
    float3 sunlight = skyProviderSunIrradiance(p) * cb.preExposure;
    RayDesc shadow;
    shadow.Origin = p + carrierNormal * 0.06f + up * 0.02f; shadow.Direction = sun; shadow.TMin = 0.02f; shadow.TMax = 1000;
    const uint shadowFlags = flags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    RayQuery<shadowFlags> sq;
    sq.TraceRayInline(topLevelAS, shadowFlags, OBJECT_MASK_OPAQUE, shadow);
    while (sq.Proceed()) { }
    float visibility = sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0f : 1.0f;
    // No visibility-independent sun floor: in full shadow the foam is lit by the
    // sky term alone, which is what "follows the environment" has to mean. Raise
    // rtx.water.shoreFoamSkyAmbient if shadowed foam reads too dark.
    light = sunlight * (elevation * visibility / 3.14159265f) + ambient;
  } else {
    // AtmosphereArgs is only populated in Numos mode. Rasterized skies must
    // use their real environment and scene lights, not the inactive sun fields.
    light = foamRasterSkyLighting(p, carrierNormal, up);
  }
  float4 input = Color[pixel];
  Color[pixel] = float4(lerp(input.xyz, light * float3(0.8f, 0.85f, 0.83f), alpha), input.w);
}
