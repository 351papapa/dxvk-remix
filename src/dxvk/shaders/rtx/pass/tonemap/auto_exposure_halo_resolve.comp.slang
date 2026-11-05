/*
* Halo Meter auto exposure, pass 2 of 2: fold the tiles, turn them into an exposure and
* move the stored one towards it.
*
* Provenance: a Slang port of the `MeterResolve` compute entry point in
* shaders/composite.hlsl of the "halocedlaart" Halo CE (MCC) path-tracer ReShade addon
* (archive halocedlaart5.rar). See auto_exposure_halo_meter.comp.slang for the
* attribution note.
*
* The wanted exposure is clamp(target / geometricMean * compensation, 2^minEv, 2^maxEv).
* The blend is linear in the exposure multiplier, with the fraction computed on the CPU.
* The first reading after a reset snaps instead of fading up.
* The result goes to the shared 1x1 exposure texture that all downstream consumers read.
*/
#include "rtx/pass/tonemap/tonemapping.h"

// x = sum(w * log), y = sum(w), z = in-scene surface centre weight, w = total centre weight.
layout(rgba32f, binding = AUTO_EXPOSURE_HALO_PARTIALS_INPUT_OUTPUT)
RWTexture2D<float4> InPartials;

layout(rg32f, binding = AUTO_EXPOSURE_HALO_STATE_INPUT_OUTPUT)
RWTexture1D<float2> InOutState;

layout(r32f, binding = AUTO_EXPOSURE_HALO_EXPOSURE_OUTPUT)
RWTexture1D<float> OutExposure;

layout(push_constant)
ConstantBuffer<ToneMappingHaloMeterArgs> cb;

static const float kMeterFloor = 1e-4f;
static const uint kResolveThreads = 256u;

groupshared float g_meterLogSum[kResolveThreads];
groupshared float g_meterWeightSum[kResolveThreads];
groupshared float g_meterSurfaceSum[kResolveThreads];
groupshared float g_meterCentreSum[kResolveThreads];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint index : SV_GroupIndex)
{
  const uint groups = cb.groupsX * cb.groupsY;

  float4 sum = float4(0.0f);
  for (uint i = index; i < groups; i += kResolveThreads)
  {
    sum += InPartials[uint2(i % cb.groupsX, i / cb.groupsX)];
  }

  g_meterLogSum[index] = sum.x;
  g_meterWeightSum[index] = sum.y;
  g_meterSurfaceSum[index] = sum.z;
  g_meterCentreSum[index] = sum.w;

  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = kResolveThreads / 2u; stride > 0u; stride >>= 1u)
  {
    if (index < stride)
    {
      g_meterLogSum[index] += g_meterLogSum[index + stride];
      g_meterWeightSum[index] += g_meterWeightSum[index + stride];
      g_meterSurfaceSum[index] += g_meterSurfaceSum[index + stride];
      g_meterCentreSum[index] += g_meterCentreSum[index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (index != 0u)
  {
    return;
  }

  // A non-finite or non-positive stored exposure would never recover through the blend,
  // so it counts as unmetered and the next reading snaps.
  float2 previous = InOutState[0];
  if (!isfinite(previous.x) || !(previous.x > 0.0f))
  {
    previous = float2(1.0f, 0.0f);
  }

  // Hold when there is no weight, or when scene surface covers less than minCoverage
  // of the centre-weighted frame.
  const float coverage = g_meterCentreSum[0] > 0.0f ? g_meterSurfaceSum[0] / g_meterCentreSum[0] : 0.0f;
  if (g_meterWeightSum[0] <= 1e-6f || coverage < cb.minCoverage)
  {
    if (previous.y > 0.5f)
    {
      OutExposure[0] = previous.x;  // Keep the adapted exposure; state unchanged.
    }
    else
    {
      // Nothing has been metered since the reset, so stay neutral and unmetered.
      OutExposure[0] = 1.0f;
    }
    return;
  }

  // The mean floor stops a near-black frame from running the exposure away.
  const float mean = max(exp(g_meterLogSum[0] / max(g_meterWeightSum[0], 1e-6f)), cb.meanFloor);
  if (!isfinite(mean))
  {
    // Nothing valid was metered this frame, so hold.
    OutExposure[0] = previous.y > 0.5f ? previous.x : 1.0f;
    InOutState[0] = previous;
    return;
  }
  // The scene's EV100 sets a compensation bias, so night can sit a few stops under mid grey.
  const float sceneEv = log2(max(mean, kMeterFloor)) + cb.evOffset;
  const float nightWeight = cb.compensationDayEv > cb.compensationNightEv
    ? saturate((cb.compensationDayEv - sceneEv) / (cb.compensationDayEv - cb.compensationNightEv))
    : (sceneEv <= cb.compensationNightEv ? 1.0f : 0.0f);
  const float compensation = exp2(cb.compensationNightStops * nightWeight);
  const float wanted = clamp(cb.target / max(mean, kMeterFloor) * compensation,
                             exp2(cb.minEv), exp2(cb.maxEv));

  // State .y is 1 once anything has been metered since the last reset. Until then the blend snaps.
  const float blend = previous.y > 0.5f ? saturate(cb.adaptation) : 1.0f;
  const float exposure = lerp(previous.x, wanted, blend);

  InOutState[0] = float2(exposure, 1.0f);
  OutExposure[0] = exposure;
}
