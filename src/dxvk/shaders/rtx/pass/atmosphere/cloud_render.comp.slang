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

// Cloud Render Compute Pass — Nubis Cubed 2023 (fork — 2026-05-12, C4).
//
// Per-pixel view-direction raymarch through the cloud slab using the Nubis
// Cubed 2023 lighting equations (paper pp. 137, 142). Writes premultiplied
// RGB + view-ray cloud transmittance alpha to AtmosphereCloudRender at the
// downscale render extent.
//
// The view-march replaces the analytical evalClouds() rendering — the
// Wrenninge octave sum + isotropic deep-scatter floor — with the page-137
// two-HG-lobe direct term (sigma_ms remap on sun_dot + sdf depth) and the
// page-142 ambient pow(1 - dim_profile, 0.5) * exp(-D_ambient).
//
// Composited into sky-miss in C5 (gated by default-off RTX_OPTION). C4
// only ships the RT and a debug view (enum 876) for standalone A/B against
// the existing analytical look.
//
// Coordinate frame: ALL math operates in Y-up world space with the camera
// at the origin (the established cloud-math convention — see
// `intersectCloudLayer`'s "Ray origin is at the camera (0,0,0) in Y-up
// space" comment). Per-pixel view directions are reconstructed CPU-side
// into the cloudRenderForwardYUp / RightYUp / UpYUp basis vectors in
// AtmosphereArgs; the shader builds viewDir per pixel from those basis
// vectors. Sun direction comes from args.sunDirection (already Y-up).

// Note: define ATMOSPHERE_AVAILABLE so the sampleDSun / sampleDAmbient
// helpers in atmosphere_common.slangh resolve to texture lookups (rather
// than the binding-free fallback stubs used by the LUT bakers). The
// global names the helpers reach for (AtmosphereCloudDSun, AtmosphereCloudDAmbient,
// AtmosphereCloudNoise3D, AtmosphereCloudNoise3DSampler, AtmosphereFastNoise)
// are declared at our local slots BELOW so they're in scope before the
// include's helper bodies are parsed.
#define ATMOSPHERE_AVAILABLE

// Bring in shader_types (for vec3/uint/etc.) and AtmosphereArgs ahead of the
// global texture declarations so AtmosphereArgs is defined before slot 0's CB.
#include "rtx/utility/shader_types.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

// Cloud 3D Perlin FBM noise volume — same resource that sampleCloudDensityTextured
// consumes elsewhere. Linear/REPEAT sampler matches the frac()-tile-wrap
// convention. NOTE: the binding *name* `AtmosphereCloudNoise3D` matches the
// global declared in common_bindings.slangh; we re-declare it here at our own
// binding slot since this compute pass owns its descriptor set independently
// from the COMMON_RAYTRACING_BINDINGS macro.
layout(binding = 1)
Texture3D<float> AtmosphereCloudNoise3D;

layout(binding = 2)
SamplerState AtmosphereCloudNoise3DSampler;

// Cloud voxel grids (D_sun and D_ambient), sampled via the same linear/REPEAT
// sampler. Names match the common_bindings.slangh globals so the helpers in
// atmosphere_common.slangh (sampleDSun / sampleDAmbient) resolve them.
layout(binding = 3)
Texture3D<float> AtmosphereCloudDSun;

layout(binding = 4)
Texture3D<float> AtmosphereCloudDAmbient;

// EA Importance-Sampled FAST noise (128x128x32 RG8 Texture2DArray) — same
// resource consumed by fastJitter() in atmosphere_common.slangh.
layout(binding = 5)
Texture2DArray<float2> AtmosphereFastNoise;

layout(binding = 6, rgba16f)
RWTexture2D<float4> cloudRenderOutput;

// Sky-view LUT + cloud-sky-transmittance LUT for the page-142 ambient
// term. sampleSkyAmbientForVolume in atmosphere_common.slangh consumes
// these as explicit parameters (so the helper compiles in any TU).
// Linear/CLAMP sampler — sky-view LUT is keyed by (azimuth, elevation);
// REPEAT would alias the south pole onto the north.
layout(binding = 7)
Texture2D<float4> AtmosphereSkyViewLut;

layout(binding = 8)
Texture2D<float> AtmosphereCloudSkyTransmittanceLut;

layout(binding = 9)
SamplerState AtmosphereSkyViewSampler;

// Include the helper library AFTER the texture globals are declared so the
// ATMOSPHERE_AVAILABLE-gated helper bodies (sampleDSun / sampleDAmbient /
// fastJitter / etc.) resolve to our local declarations.
#include "rtx/pass/atmosphere/atmosphere_common.slangh"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
  uint2 pixelCoord = dispatchId.xy;
  uint2 outputDim;
  cloudRenderOutput.GetDimensions(outputDim.x, outputDim.y);
  if (any(pixelCoord >= outputDim)) return;

  AtmosphereArgs args = atmosphereArgs;

  // Trivially-clear sky: no clouds means full transmittance, zero radiance.
  if (args.cloudEnabled < 0.5f || args.cloudCoverageMean < 0.01f || args.cloudViewSamples == 0u) {
    cloudRenderOutput[pixelCoord] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  // ===== Reconstruct per-pixel view ray in Y-up world space =====
  // NDC in [-1, 1] with top-left pixel mapping to (-1, +1) Vulkan convention.
  vec2 uv = (vec2(pixelCoord) + 0.5f) / vec2(outputDim);
  vec2 ndc = uv * 2.0f - 1.0f;
  ndc.y = -ndc.y;  // y-down pixels -> y-up NDC

  // Basis vectors are pre-multiplied by tan(halfFovX/Y) and aspect ratio
  // CPU-side, so the math is just a weighted sum.
  vec3 viewDirYUp = normalize(args.cloudRenderForwardYUp
                              + ndc.x * args.cloudRenderRightYUp
                              + ndc.y * args.cloudRenderUpYUp);

  // Below-horizon view direction: no clouds above this pixel.
  if (viewDirYUp.y <= 0.001f) {
    cloudRenderOutput[pixelCoord] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  // ===== Slab intersection (mirrors evalClouds at atmosphere_sky.slangh:679) =====
  // Two ray-sphere tests against the base + top shells of the curvature-
  // adjusted cloud planet so the dome tightens as cloudCurvature increases.
  // Camera is implicitly at world origin in Y-up.
  float cloudR = cloudPlanetRadius(args);
  vec3 planetCenter = vec3(0.0f, -cloudR, 0.0f);
  float baseRadius = cloudR + args.cloudAltitude;
  float topRadius  = cloudR + args.cloudAltitude + args.cloudThickness;

  float t0base, t1base, t0top, t1top;
  if (!intersectSphere(vec3(0.0f), viewDirYUp, planetCenter, baseRadius, t0base, t1base)) {
    cloudRenderOutput[pixelCoord] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }
  if (!intersectSphere(vec3(0.0f), viewDirYUp, planetCenter, topRadius, t0top, t1top)) {
    cloudRenderOutput[pixelCoord] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  float tEntry = t1base;
  float tExit  = t1top;
  if (tEntry >= tExit || tExit <= 0.0f) {
    cloudRenderOutput[pixelCoord] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }
  tEntry = max(tEntry, 0.0f);  // camera may sit above slab base

  // ===== Per-pixel FAST-noise jitter on the fractional step offset =====
  // Mirrors the view-march jitter pattern from evalClouds. Channel x is the
  // view-march; we don't take more than one tap so channel y is unused.
  float jitter = fastJitter(pixelCoord, args.cloudRenderFrameIdx).x;

  uint sampleCount = args.cloudViewSamples;
  float stepLen = (tExit - tEntry) / float(sampleCount);

  // ===== Per-ray lighting source radiances =====
  // sunRadiance: args.sunIlluminance is raw user-tuned color (near-white in
  // FNV defaults) -- atmospheric extinction along the sun ray is NOT applied
  // CPU-side. We mirror the legacy evalClouds path's `airTrans` step here so
  // the direct term picks up the warm-orange Rayleigh shift at low sun
  // elevation. Without this, result.direct stays white while result.ambient
  // (from the sky-view LUT) reads orange, producing a per-pixel hue mismatch
  // that varies with cloudPhaseG2 -- the bug the C6.4 diagnosis caught.
  //
  // skyRadiance: sampled from the sky-view LUT at the sun direction (clamped
  // to a hair above horizon so we don't return zero at exactly-on-horizon sun).
  // Deliberate deviation from page-142's "zenith ambient" -- zenith at sunset
  // is bluer than horizon, so sampling there would tint sunset cumulus bulk
  // blue. Sampling at the sun direction gives the actual warm-orange forward-
  // scatter atmosphere color matching the paper's pp. 122/132/171-173 renders.
  vec3 sunDirYUp = normalize(args.sunDirection);
  vec3 airTrans = getAtmosphericTransmittanceForDir(sunDirYUp, args);
  vec3 sunRadiance = args.sunIlluminance * airTrans;

  vec3 sunDirForAmbient = sunDirYUp;
  sunDirForAmbient.y = max(sunDirForAmbient.y, 0.05f);
  sunDirForAmbient = normalize(sunDirForAmbient);
  vec3 skyRadiance = sampleSkyAmbientForVolume(sunDirForAmbient, args,
                                                AtmosphereSkyViewLut,
                                                AtmosphereCloudSkyTransmittanceLut,
                                                AtmosphereSkyViewSampler);

  // ===== View-ray raymarch =====
  // evalNubisCubedSample internally uses sigma = args.cloudDensity (no doubling
  // — the Nubis Cubed equations use raw cloudDensity as the extinction
  // coefficient; the existing Wrenninge path's baseSigma = 2*cloudDensity is
  // intentionally NOT mirrored here so the new look isn't double-dimmed).
  vec3 accumColor = vec3(0.0f);
  float viewTransmittance = 1.0f;

  for (uint i = 0u; i < sampleCount; ++i) {
    if (viewTransmittance < 0.01f) break;  // early-exit (free perf win)

    float t = tEntry + stepLen * (float(i) + jitter);
    vec3 samplePos = viewDirYUp * t;  // camera at origin

    // Spatial coverage / type fields, sampled per-step (same convention as
    // evalClouds at atmosphere_sky.slangh:913).
    vec2 sampleXZ = vec2(samplePos.x, samplePos.z);
    float typeNoise     = smoothNoise2D(sampleXZ * args.cloudTypeNoiseScale, 200.0f);
    float coverageNoise = smoothNoise2D(sampleXZ * args.cloudCoverageNoiseScale, 250.0f);
    float typeLocal     = saturate(args.cloudTypeMean     + (typeNoise     - 0.5f) * args.cloudTypeSpread);
    float coverageLocal = saturate(args.cloudCoverageMean + (coverageNoise - 0.5f) * args.cloudCoverageSpread);

    float hf = computeCloudHeightFraction(samplePos, args);
    float density = sampleCloudDensityTextured(samplePos, hf, typeLocal, coverageLocal, args,
                                                AtmosphereCloudNoise3D, AtmosphereCloudNoise3DSampler);
    if (density < 0.001f) continue;

    // Nubis Cubed lighting at this sample. samplePos is already Y-up, so
    // isZUp = false for the D_sun / D_ambient lookups.
    NubisCubedLighting lighting = evalNubisCubedSample(
        samplePos, hf, density,
        viewDirYUp, sunDirYUp, sunRadiance, skyRadiance,
        args, /*isZUp=*/false);

    vec3 sampleColor = args.cloudColor * (lighting.direct + lighting.ambient);

    // Beer-Lambert step (analytical integration of constant emission across
    // the step). Mirrors evalClouds' integration form.
    float effectiveDensity = density * args.cloudDensity;
    float stepTransmittance = exp(-effectiveDensity * stepLen);
    vec3 stepRadiance = sampleColor * (1.0f - stepTransmittance);
    accumColor += stepRadiance * viewTransmittance;
    viewTransmittance *= stepTransmittance;
  }

  // Output is premultiplied: rgb = accumulated cloud radiance,
  // a = view-ray cloud transmittance.
  cloudRenderOutput[pixelCoord] = vec4(accumColor, viewTransmittance);
}
