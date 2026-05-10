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

/*
 * Cloud shadow map compute pass. Each texel ray-marches upward along sunDir
 * through the cloud slab from a ground (X, 0, Z) position, accumulating
 * optical depth along 4 deterministic taps and writing Beer-Lambert
 * transmittance to a 2D R8 UNORM texture.
 *
 * Spec: docs/superpowers/specs/2026-05-09-cloud-shadow-map-design.md
 *
 * Math reuse: this shader includes atmosphere_common.slangh purely for the
 * pure-math helpers (cloudPlanetRadius / smoothNoise2D / cloudTypeProfile /
 * computeCloudHeightFraction / sampleCloudDensityTextured / intersectSphere).
 * It does NOT include atmosphere_bindings.slangh -- the dispatch's binding
 * surface is the minimal set in cloud_shadow_map_bindings.slangh:
 *   - the R8 output texture
 *   - the cloud-noise 3D SRV/sampler (passed as parameters into
 *     sampleCloudDensityTextured so atmosphere_common doesn't have to see
 *     them as globals)
 *   - a push constant with the cloud-march subset of AtmosphereArgs.
 *
 * The helpers in atmosphere_common all consume an `AtmosphereArgs` struct.
 * To call them without dragging the full ~1KB AtmosphereArgs CBV into a
 * push constant, main() synthesizes a partial AtmosphereArgs from the push
 * fields, populating only the members the called helpers reach into:
 *   cloudPlanetRadius:           planetRadius, cloudCurvature
 *   computeCloudHeightFraction:  + cloudAltitude, cloudThickness
 *   sampleCloudDensityTextured:  + cloudWindOffset, cloudNoiseTileKm,
 *                                  cloudAnvilBias
 * All other AtmosphereArgs members stay default-initialized (0); the helpers
 * never touch them on this code path.
 */

#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/atmosphere/cloud_shadow_map_bindings.slangh"

// Slab entry/exit for an upward-going ray that starts at a ground position
// (Y = 0) and points along the sun direction. The cloud slab is bounded by
// two concentric spheres centered at (0, -cloudR, 0) -- the curvature-
// adjusted cloud planet center -- with radii (cloudR + cloudAltitude) for
// the base and (cloudR + cloudAltitude + cloudThickness) for the top.
//
// Generalizes intersectCloudLayer (which assumes ray-origin = world origin)
// for the shadow-map case where each texel marches from its own (X, 0, Z)
// ground point. Reuses atmosphere_common::intersectSphere so the algebra
// matches the existing rendering path bit-for-bit.
//
// Returns false (and leaves t-outputs undefined) when:
//   - sun is at or below horizon (no upward path through the slab)
//   - either bounding sphere is missed (numerically degenerate cases)
//   - the slab is degenerate (entry >= exit, or exit behind ray origin)
//
// On success: tEntry is clamped to >= 0 so the caller can march the
// [tEntry, tExit] range without worrying about negative-t pre-march space.
bool intersectCloudSlabFromGround(vec3 groundPos, vec3 sunDir,
                                  AtmosphereArgs args,
                                  out float tEntry, out float tExit) {
  tEntry = 0.0f;
  tExit  = 0.0f;

  // Looking horizontal-or-below: cloud-march is meaningless. Same epsilon
  // intersectCloudLayer uses for the same reason.
  if (sunDir.y <= 0.001f) {
    return false;
  }

  float cloudR     = cloudPlanetRadius(args);
  vec3  planetC    = vec3(0.0f, -cloudR, 0.0f);
  float baseRadius = cloudR + args.cloudAltitude;
  float topRadius  = cloudR + args.cloudAltitude + args.cloudThickness;

  float t0b, t1b, t0t, t1t;
  if (!intersectSphere(groundPos, sunDir, planetC, baseRadius, t0b, t1b)) {
    return false;
  }
  if (!intersectSphere(groundPos, sunDir, planetC, topRadius, t0t, t1t)) {
    return false;
  }

  // Ground position (Y = 0) lies above both planet centers and below both
  // cloud spheres -- so we are *inside* both bounding spheres. The far
  // intersection (t1) is the upward exit through each shell:
  //   t1b = where the upward ray exits the base sphere -> slab entry
  //   t1t = where the upward ray exits the top  sphere -> slab exit
  tEntry = t1b;
  tExit  = t1t;

  if (tEntry >= tExit || tExit <= 0.0f) {
    return false;
  }
  tEntry = max(tEntry, 0.0f);
  return true;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadID : SV_DispatchThreadID) {
  if (threadID.x >= cb.resolution || threadID.y >= cb.resolution) {
    return;
  }

  vec3 sunDir = normalize(cb.sunDirection);

  // Sun below horizon: no shadow, full transmittance. The shadow map's
  // consumers blend cloudShadowStrength on top of this, so a uniform 1.0
  // here is the correct "no occlusion" value at night.
  if (sunDir.y < 0.0f) {
    CloudShadowMapOutput[threadID] = 1.0f;
    return;
  }

  // World position of this texel's ground point. +0.5 centers the sample
  // on the texel; mapOriginXZ is the bottom-left corner pre-snapped on CPU.
  vec2 worldXZ   = cb.mapOriginXZ + (vec2(threadID) + 0.5f) * cb.texelSize;
  vec3 groundPos = vec3(worldXZ.x, 0.0f, worldXZ.y);

  // Synthesize the partial AtmosphereArgs that the called helpers need.
  // All other members default-initialize to 0 and are never touched on the
  // cloud-density path (verified by audit of cloudPlanetRadius /
  // computeCloudHeightFraction / sampleCloudDensityTextured).
  AtmosphereArgs args = (AtmosphereArgs)0;
  args.planetRadius       = cb.planetRadius;
  args.cloudAltitude      = cb.cloudAltitude;
  args.cloudThickness     = cb.cloudThickness;
  args.cloudCurvature     = cb.cloudCurvature;
  args.cloudWindOffset    = cb.cloudWindOffset;
  args.cloudNoiseTileKm   = cb.cloudNoiseTileKm;
  args.cloudAnvilBias     = cb.cloudAnvilBias;

  float tEntry, tExit;
  if (!intersectCloudSlabFromGround(groundPos, sunDir, args, tEntry, tExit)) {
    CloudShadowMapOutput[threadID] = 1.0f;
    return;
  }

  // 4 deterministic taps. No jitter -- 4 taps is enough at 15.6m/texel
  // (matching the natural penumbra) that integration noise is dominated by
  // the natural cloud-density variation rather than tap pattern. Adding
  // per-pixel jitter would create temporal noise on a per-frame baked
  // texture that has no temporal accumulator behind it.
  const int kTaps = 4;
  const float dt  = (tExit - tEntry) / float(kTaps);

  float opticalDepth = 0.0f;

  [unroll]
  for (int i = 0; i < kTaps; ++i) {
    float t  = tEntry + dt * (float(i) + 0.5f);
    vec3 sp  = groundPos + sunDir * t;
    vec2 spXZ = vec2(sp.x, sp.z);

    // Same Nubis weather-noise scheme as evalCloudGroundShadow /
    // evalSkyRadiance: independent smoothNoise2D evals with separate seeds
    // produce spatially-decorrelated type and coverage fields.
    float typeNoise     = smoothNoise2D(spXZ * cb.cloudTypeNoiseScale,     200.0f);
    float coverageNoise = smoothNoise2D(spXZ * cb.cloudCoverageNoiseScale, 250.0f);
    float typeLocal     = saturate(cb.cloudTypeMean
                                 + (typeNoise - 0.5f) * cb.cloudTypeSpread);
    float coverageLocal = saturate(cb.cloudCoverageMean
                                 + (coverageNoise - 0.5f) * cb.cloudCoverageSpread);

    float hf      = computeCloudHeightFraction(sp, args);
    float density = sampleCloudDensityTextured(sp, hf, typeLocal, coverageLocal,
                                               args,
                                               AtmosphereCloudNoise3D,
                                               AtmosphereCloudNoise3DSampler);
    opticalDepth += density;
  }

  // Beer-Lambert. The 2.0 factor matches evalCloudGroundShadow's existing
  // calibration so cloud-shadow darkness is consistent between the legacy
  // single-point path and the new per-texel map (transmittance = exp(-density
  // * cloudDensity * 2.0f) at the same density value).
  float transmittance = exp(-opticalDepth * dt * 2.0f * cb.cloudDensity);
  CloudShadowMapOutput[threadID] = transmittance;
}
