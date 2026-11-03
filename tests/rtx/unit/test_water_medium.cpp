// SPDX-License-Identifier: MIT
#include <cmath>
#include <cstdio>
#include <limits>
#include "glm/gtc/packing.hpp"

#include "../../../src/dxvk/rtx_render/rtx_water_medium.h"

int main() {
  using namespace dxvk;
  int failures = 0;
  const auto check = [&](bool passed, const char* message) {
    if (!passed) {
      ++failures;
      std::printf("FAIL: %s\n", message);
    }
  };
  const auto closeTo = [](float actual, float expected) {
    return std::abs(actual - expected) <= 0.00001f * (1.0f + std::abs(expected));
  };

  const auto optics = waterOpticalProperties(Vector3(0.25f, 0.5f, 0.8f), 4.0f);
  for (uint32_t channel = 0; channel < 3; ++channel) {
    check(closeTo(std::exp(-optics.extinction[channel] * optics.distance), optics.transmittance[channel]),
      "Beer-Lambert reconstructs authored linear transmission at its measurement distance");
    check(closeTo(std::exp(-optics.extinction[channel] * optics.distance * 2.0f),
      optics.transmittance[channel] * optics.transmittance[channel]), "twice the path length squares transmission");
    check(std::exp(-optics.extinction[channel] * 0.0f) == 1.0f, "a zero-length water segment does not absorb");
  }
  check(closeTo(optics.extinction.x, std::log(4.0f) / 4.0f), "linear transmission is not gamma converted");
  const auto white = waterOpticalProperties(Vector3(1.0f), 1.0f);
  check(white.extinction == Vector3(0.0f), "white transmission has exactly zero extinction");

  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const auto invalid = waterOpticalProperties(Vector3(nan, infinity, -infinity), nan);
  check(invalid.transmittance == Vector3(1.0f) && invalid.extinction == Vector3(0.0f) && invalid.distance == 1.0f,
    "nonfinite optical inputs fall back to a finite clear medium");
  const auto clamped = waterOpticalProperties(Vector3(-10.0f, 0.0f, 2.0f), -10.0f);
  check(clamped.transmittance == Vector3(0.0001f, 0.0001f, 1.0f) && clamped.distance == 0.1f,
    "finite out-of-range inputs saturate to the optical limits");
  check(std::isfinite(clamped.extinction.x) && clamped.extinction.x >= 0.0f && clamped.extinction.z == 0.0f,
    "saturated optical inputs produce bounded nonnegative extinction");
  check(waterOpticalProperties(Vector3(0.5f), 1000.0f).distance == 100.0f,
    "measurement distance has a finite upper limit");
  check(waterOpticalProperties(Vector3(0.5f), infinity).distance == 1.0f,
    "infinite distance uses the finite default");

  // Adding neutral particles must retain the authored baseline when disabled, without
  // binding their density to the least-absorbed (nearly transparent blue) channel.
  const Vector3 authoredTransmission(0.0f, 0.86248f, 0.988417f);
  const auto authored = waterOpticalProperties(authoredTransmission, 17.9f);
  for (const float particles : {0.0f, -1.0f, nan, infinity}) {
    const auto unchanged = waterOpticalProperties(authoredTransmission, 17.9f, particles);
    check(unchanged.transmittance == authored.transmittance && unchanged.distance == authored.distance
      && unchanged.materialDistance == authored.distance && unchanged.extinction == authored.extinction,
      "zero, negative or nonfinite particle density preserves the exact existing baseline");
  }
  const auto turbid = waterOpticalProperties(authoredTransmission, 17.9f, 0.01f);
  for (uint32_t channel = 0; channel < 3; ++channel) {
    check(closeTo(turbid.extinction[channel], authored.extinction[channel] + 0.01f),
      "added neutral particles increase total extinction equally in every channel");
    check(closeTo(std::exp(-turbid.extinction[channel] * 17.9f),
      authored.transmittance[channel] * std::exp(-0.01f * 17.9f)),
      "total transmission combines the authored medium with independent particle extinction");
  }
  check(turbid.transmittance.x < 0.0001f,
    "the authored zero-red limit does not clip newly added particle extinction");
  const auto dense = waterOpticalProperties(authoredTransmission, 17.9f, 0.2f);
  check(dense.materialDistance < dense.distance && dense.distance == authored.distance,
    "dense material encoding shortens its reference distance while retaining the authored volume range");
  for (const auto encoded : {turbid, dense,
      waterOpticalProperties(Vector3(0.0f), 100.0f, 1.0f),
      waterOpticalProperties(Vector3(1.0f), 100.0f, 0.01f),
      waterOpticalProperties(Vector3(0.0f, 0.5f, 1.0f), 0.1f, std::numeric_limits<float>::max())}) {
    const float gpuDistance = glm::unpackHalf1x16(glm::packHalf1x16(encoded.materialDistance));
    check(encoded.materialDistance >= 0.001f && std::isfinite(gpuDistance) && gpuDistance >= 0.001f,
      "effective reference distance survives the material clamp and half storage");
    for (uint32_t channel = 0; channel < 3; ++channel) {
      const auto packedTransmission = glm::packHalf1x16(encoded.transmittance[channel]);
      const float gpuTransmission = glm::unpackHalf1x16(packedTransmission);
      const float gpuExtinction = -std::log(gpuTransmission) / gpuDistance;
      check(packedTransmission >= 0x0400 && packedTransmission <= 0x3c00,
        "effective transmission remains a normal positive half value");
      check(closeTo(-std::log(encoded.transmittance[channel]) / encoded.materialDistance, encoded.extinction[channel]),
        "material encoding reconstructs the same total extinction used by camera volumes");
      check(std::abs(gpuExtinction - encoded.extinction[channel]) <= 0.003f * encoded.extinction[channel] + 0.000001f,
        "actual half material storage reconstructs volume extinction within 0.3 percent");
    }
  }

  constexpr uint64_t waterHash = 0x123456789abcdef0ull;
  check(parseWaterMediumHash("123456789abcdef0") == waterHash
    && parseWaterMediumHash("0x123456789ABCDEF0") == waterHash
    && parseWaterMediumHash("0X123456789abcdef0") == waterHash, "hash parser accepts optional hexadecimal prefixes");
  check(parseWaterMediumHash("FFFFFFFFFFFFFFFF") == std::numeric_limits<uint64_t>::max(),
    "hash parser preserves every bit of a 64-bit material hash");
  for (const auto text : {"", "0x", "0X", "-1", "+1", " 12", "12 ", "12tail", "0x0x1", "10000000000000000"}) {
    check(parseWaterMediumHash(text) == 0, "malformed or overflowing hash does not select a medium");
  }

  const WaterMediumState clear;
  const auto entered = makeWaterMediumState(true, waterHash, waterHash, optics, 0.5f, 0.4f, clear);
  check(entered.active && entered.changed, "entering the selected water medium invalidates history");
  check(entered.extinction == optics.extinction && entered.attenuationDistance == optics.distance,
    "medium and surface share extinction and measurement distance");
  check(closeTo(entered.scattering.x, optics.extinction.z * 0.5f)
    && entered.scattering.x == entered.scattering.y && entered.scattering.y == entered.scattering.z,
    "neutral scattering follows the least-absorbed channel");
  for (const float fraction : {-2.0f, 0.0f, 0.2f, 1.0f, 2.0f, nan, infinity}) {
    const auto state = makeWaterMediumState(true, waterHash, waterHash, optics, fraction, 0.4f, clear);
    for (uint32_t channel = 0; channel < 3; ++channel) {
      check(std::isfinite(state.scattering[channel]) && state.scattering[channel] >= 0.0f
        && state.scattering[channel] <= state.extinction[channel], "scattering never exceeds extinction in any channel");
    }
  }
  const auto particlesOnly = waterOpticalProperties(Vector3(1.0f), 17.9f, 0.01f);
  const auto particleMedium = makeWaterMediumState(true, waterHash, waterHash, particlesOnly, 0.9f, 0.4f, clear);
  check(closeTo(particleMedium.scattering.x, 0.009f) && closeTo(particleMedium.extinction.x, 0.01f),
    "a transparent baseline can independently contain neutral scattering particles");
  const auto turbidMedium = makeWaterMediumState(true, waterHash, waterHash, turbid, 0.9f, 0.4f, clear);
  check(closeTo(turbidMedium.scattering.x, (authored.extinction.z + 0.01f) * 0.9f)
    && turbidMedium.scattering.x > 0.009f,
    "added particles provide substantial scattering even with near-transparent baseline blue");
  for (const float fraction : {0.0f, 0.2f, 0.9f, 1.0f}) {
    const auto state = makeWaterMediumState(true, waterHash, waterHash, turbid, fraction, 0.4f, clear);
    const float neutralExtinction = authored.extinction.z + 0.01f;
    for (uint32_t channel = 0; channel < 3; ++channel) {
      const float absorption = authored.extinction[channel] - authored.extinction.z + neutralExtinction * (1.0f - fraction);
      check(absorption >= 0.0f && closeTo(absorption + state.scattering[channel], state.extinction[channel]),
        "colored absorption plus neutral absorption and scattering conserve total extinction");
    }
  }
  const auto invalidScattering = makeWaterMediumState(true, waterHash, waterHash, optics, nan, infinity, clear);
  check(invalidScattering.scattering == Vector3(0.0f) && invalidScattering.anisotropy == 0.0f,
    "nonfinite scattering controls have finite isotropic defaults");
  check(makeWaterMediumState(true, waterHash, waterHash, optics, 2.0f, 2.0f, clear).anisotropy == 0.8f
    && makeWaterMediumState(true, waterHash, waterHash, optics, -2.0f, -2.0f, clear).anisotropy == 0.0f,
    "anisotropy saturates at both limits");
  check(makeWaterMediumState(true, waterHash, waterHash, white, 1.0f, 0.4f, clear).scattering == Vector3(0.0f),
    "a clear medium cannot create scattering energy");

  for (const auto state : {
      makeWaterMediumState(false, waterHash, waterHash, optics, 0.5f, 0.4f, clear),
      makeWaterMediumState(true, 0, 0, optics, 0.5f, 0.4f, clear),
      makeWaterMediumState(true, waterHash, 0, optics, 0.5f, 0.4f, clear),
      makeWaterMediumState(true, waterHash, waterHash + 1, optics, 0.5f, 0.4f, clear)}) {
    check(!state.active && !state.changed && state.extinction == Vector3(0.0f) && state.scattering == Vector3(0.0f),
      "disabled, unset and non-water camera media preserve the clear fallback");
  }

  const auto stable = makeWaterMediumState(true, waterHash, waterHash, optics, 0.5f, 0.4f, entered);
  check(stable.active && !stable.changed, "identical consecutive frames retain history");
  const auto recolored = makeWaterMediumState(true, waterHash, waterHash,
    waterOpticalProperties(Vector3(0.3f, 0.5f, 0.8f), 4.0f), 0.5f, 0.4f, stable);
  check(recolored.changed, "water color changes invalidate active history");
  check(makeWaterMediumState(true, waterHash, waterHash,
    waterOpticalProperties(optics.transmittance, 8.0f), 0.5f, 0.4f, stable).changed,
    "measurement distance changes invalidate active history");
  check(makeWaterMediumState(true, waterHash, waterHash, optics, 0.75f, 0.4f, stable).changed,
    "scattering changes invalidate active history");
  check(makeWaterMediumState(true, waterHash, waterHash, optics, 0.5f, 0.6f, stable).changed,
    "phase changes invalidate active history");
  check(makeWaterMediumState(true, waterHash, waterHash,
    waterOpticalProperties(Vector3(0.25f, 0.5f, 0.8f), 4.0f, 0.01f), 0.5f, 0.4f, stable).changed,
    "particle-density changes invalidate active volume history");
  const auto exited = makeWaterMediumState(true, waterHash, 0, optics, 0.5f, 0.4f, stable);
  check(!exited.active && exited.changed, "leaving water invalidates history once");
  const auto inactiveEdit = makeWaterMediumState(false, waterHash, waterHash, white, 0.0f, 0.0f, exited);
  check(!inactiveEdit.active && !inactiveEdit.changed, "inactive optical edits do not reset atmospheric history");
  check(makeWaterMediumState(true, waterHash, waterHash, optics, 0.5f, 0.4f, inactiveEdit).changed,
    "re-entering water invalidates history after inactive edits");

  std::printf("Water optical properties and medium transitions: %d failures\n", failures);
  return failures ? -1 : 0;
}
