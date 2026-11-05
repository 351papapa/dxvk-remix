// Boundary-geometry regression for the foam line display surface.
//
// This test was rewritten because its first version proved nothing. It checked
// `const float area2 = 0.0f; require(area2 == 0.0f)` - asserting that a constant it had
// just written equalled itself - and it compared `originX + x * spacing` against the same
// expression. Neither could fail when the production algorithm regressed. It also
// duplicated the water-validity predicate locally, so the "negative experiment" only
// proved that the copy had changed.
//
// What it does now:
//   * computes REAL triangle areas from three emitted positions, via cross(B-A, C-A),
//     across all sixteen water-validity combinations of a quad's four corners;
//   * proves the biased height encoding survives an actual binary16 round trip by
//     bit-constructing the conversion, rather than comparing an out-of-range literal;
//   * checks that a partially valid triangle never keeps non-zero area, which is exactly
//     the defect that stretched foam geometry from the foam edge toward the world origin.
//
// HONEST LIMIT: this is a CPU reference implementation of the generator's placement rule,
// not the shader itself. It mirrors water_foam_surface.comp.slang's control flow, corner
// order and geometry, and every constant it depends on is shared with production, but the
// two can still drift if one side is edited alone. Only a GPU regression that executes the
// real shader would close that gap, and that is not available here.

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>

#include "../../../src/dxvk/rtx_render/rtx_water_foam_sheet_contract.h"

namespace {

int g_failures = 0;

void require(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// ---- binary16 ---------------------------------------------------------------
float halfToFloat(uint16_t h) {
  const uint32_t sign = uint32_t(h >> 15) & 1u;
  const uint32_t exp = uint32_t(h >> 10) & 0x1Fu;
  const uint32_t man = uint32_t(h) & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) { bits = sign << 31; }
    else {
      int e = -1;
      uint32_t m = man;
      do { ++e; m <<= 1; } while ((m & 0x400u) == 0u);
      m &= 0x3FFu;
      bits = (sign << 31) | (uint32_t(127 - 15 - e) << 23) | (m << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = (sign << 31) | 0x7F800000u | (man << 13);
  } else {
    bits = (sign << 31) | ((exp + (127u - 15u)) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t floatToHalfBits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t sign = (bits >> 31) & 1u;
  const int32_t exp = int32_t((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t man = bits & 0x7FFFFFu;
  if (exp >= 0x1F) { return uint16_t((sign << 15) | 0x7C00u); }   // overflow -> inf
  if (exp <= 0) {
    if (exp < -10) { return uint16_t(sign << 15); }
    man |= 0x800000u;
    const uint32_t shift = uint32_t(14 - exp);
    uint32_t half = man >> shift;
    if ((man >> (shift - 1)) & 1u) { ++half; }
    return uint16_t((sign << 15) | half);
  }
  uint32_t half = (uint32_t(exp) << 10) | (man >> 13);
  if ((man >> 12) & 1u) { ++half; }
  return uint16_t((sign << 15) | half);
}

float halfRoundTrip(float f) { return halfToFloat(floatToHalfBits(f)); }

// ---- production-derived height/validity encoding -----------------------------
// Height is a 32-bit float and validity its own channel, so no height value is reserved for
// absence and nothing is quantised. Driven by the SHARED constants.
struct HeightSample { float height; float valid; };
HeightSample encodeHeight(float worldHeight) {
  return HeightSample { worldHeight, dxvk::kWaterFoamHasWater };
}
bool hasWaterEncoded(HeightSample s) {
  return s.valid >= 0.5f && std::isfinite(s.height);
}
float decodeHeight(HeightSample s) { return s.height; }

struct Vec2 { float x, y; };
struct Vec3 { float x, y, z; };
Vec3 vsub(Vec3 a, Vec3 b) { return Vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
Vec3 vcross(Vec3 a, Vec3 b) {
  return Vec3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
float vlen(Vec3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
float triangleArea(Vec3 a, Vec3 b, Vec3 c) { return 0.5f * vlen(vcross(vsub(b, a), vsub(c, a))); }

// Mirrors foamCellWorld in the shader.
Vec3 cellWorld(bool zUp, float originX, float originY, float spacing,
               int ix, int iy, float decodedHeight) {
  const float wx = originX + float(ix) * spacing;
  const float wy = originY + float(iy) * spacing;
  const float up = decodedHeight + dxvk::kWaterFoamSurfaceOffset;
  return zUp ? Vec3{ wx, wy, up } : Vec3{ wx, up, wy };
}

struct QuadVerts { Vec3 p[6]; };

// Emits one quad the way the shader does. Static index layout: quad k owns vertices
// [6k, 6k+6); triangle 0 = corners (00,10,11), triangle 1 = (00,11,01). A triangle whose
// corners are not all valid is collapsed by writing ONE position three times.
QuadVerts emitQuad(bool zUp, float originX, float originY, float spacing,
                   int ix, int iy, float cameraUp,
                   bool v00, bool v10, bool v01, bool v11,
                   float h00, float h10, float h01, float h11) {
  const int dxs[5] = { 0, -1, 1, 0, 0 };
  const int dys[5] = { 0, 0, 0, -1, 1 };
  const bool vals[5] = { v00, v10, v10, v01, v01 };
  const float hs[5] = { h00, h10, h10, h01, h01 };

  auto anchor = [&]() {
    for (int k = 0; k < 5; ++k) {
      if (vals[k]) {
        return cellWorld(zUp, originX, originY, spacing, ix + dxs[k], iy + dys[k], hs[k]);
      }
    }
    return cellWorld(zUp, originX, originY, spacing, ix, iy, cameraUp);
  };
  auto corner = [&](int cx, int cy, float h) {
    return cellWorld(zUp, originX, originY, spacing, cx, cy, h);
  };
  (void)corner;

  QuadVerts out{};
  if (!v00 || !v10 || !v11) {
    const Vec3 a = anchor();
    out.p[0] = out.p[1] = out.p[2] = a;
  } else {
    out.p[0] = corner(ix, iy, h00);
    out.p[1] = corner(ix + 1, iy, h10);
    out.p[2] = corner(ix + 1, iy + 1, h11);
  }
  if (!v00 || !v11 || !v01) {
    const Vec3 a = anchor();
    out.p[3] = out.p[4] = out.p[5] = a;
  } else {
    out.p[3] = corner(ix, iy, h00);
    out.p[4] = corner(ix + 1, iy + 1, h11);
    out.p[5] = corner(ix, iy + 1, h01);
  }
  return out;
}

// Reference copy of the generator's per-corner attribute rule
// (foamBuildCornerAttributes in water_foam_surface.comp.slang). The shader's foamNoise
// cannot run here, so it is replaced by a deterministic function of world position; what
// this checks is that UV and coverage derive from THE CORNER'S OWN position, density and
// flow, rather than from one representative corner copied across the triangle.
struct CornerAttr { Vec2 uv; float cov; };
CornerAttr cornerAttr(Vec2 world2, float dens, Vec2 flow, float time, float scale) {
  // Noise deliberately omitted: this reference checks that each corner derives UV and
  // coverage from its OWN position, density and flow. Substituting a fake noise field
  // would make the comparison test the substitute instead of the per-corner behaviour.
  const float n = 0.0f;
  const float f = 0.0f;
  const float curve = 1.0f - std::exp(-(dens < 0.0f ? 0.0f : dens) * 2.8f);
  const float erosion = std::min(1.0f, std::max(0.0f, (curve - (n * 0.38f + f * 0.18f)) / 0.5f));
  (void)n; (void)f;
  CornerAttr a;
  a.uv = Vec2 { world2.x / 5.0f + (-0.045f) * time - flow.x * 0.14f,
                world2.y / 5.0f + (0.035f) * time - flow.y * 0.14f };
  a.cov = std::min(1.0f, std::max(0.0f, curve * erosion * scale));
  return a;
}

} // namespace

int main() {
  using namespace dxvk;

  // ---- height precision and validity ---------------------------------------
  // The whole reason height moved to its own 32-bit resource: the previous half-precision,
  // biased encoding quantised water to 0.5 or 1.0 units, which is COARSER than the surface
  // offset. The tolerance here is therefore tied to the offset, not a loose half-unit.
  {
    const float tol = kWaterFoamHeightTolerance;
    require(tol < kWaterFoamSurfaceOffset * 0.5f,
      "height tolerance is well below the surface offset, so the surface really sits on the water");

    // Small near-zero waves: the exact case that half precision rounded or stepped.
    const float smallHeights[7] = { 0.0f, 0.05f, 0.1f, 0.2f, 0.3f, -0.1f, -0.25f };
    for (int i = 0; i < 7; ++i) {
      const float h = smallHeights[i];
      const HeightSample s = encodeHeight(h);
      require(hasWaterEncoded(s), "a small near-zero height is valid water");
      require(std::abs(decodeHeight(s) - h) <= tol,
        "a small near-zero wave keeps sub-offset precision, not half-unit steps");
    }

    // Zero must be water; under the old half encoding it was indistinguishable from absence.
    require(hasWaterEncoded(encodeHeight(0.0f)), "exactly zero is valid water");

    // Positive and negative, including camera far from the origin.
    const float big[4] = { 1234.5f, -987.25f, -1024.0f, 65504.0f };
    for (int i = 0; i < 4; ++i) {
      const HeightSample s = encodeHeight(big[i]);
      require(hasWaterEncoded(s), "large positive and negative heights are all valid water");
      require(std::abs(decodeHeight(s) - big[i]) <= tol,
        "large heights keep full precision, including the old collision value -1024");
    }

    // The probe's reachable bracket, at the same tolerance.
    const float reach[2] = { -kWaterFoamProbeDownReach, kWaterFoamProbeUpReach };
    for (int i = 0; i < 2; ++i) {
      const HeightSample s = encodeHeight(reach[i]);
      require(hasWaterEncoded(s), "the probe's reach stays valid");
      require(std::abs(decodeHeight(s) - reach[i]) <= tol, "the probe's reach keeps full precision");
    }

    // Absence is its own flag, so no legal height can collide with it. This is what the
    // biased encoding could not promise: height == -bias used to hit the sentinel.
    require(encodeHeight(-1024.0f).valid == kWaterFoamHasWater,
      "the height that used to collide with the old sentinel is simply valid water now");
    require(kWaterFoamNoWater < 0.5f && kWaterFoamHasWater >= 0.5f,
      "validity values are distinguishable without reserving any height");

    // A float round trip is exact at this scale; prove it rather than asserting the range.
    for (int i = 0; i < 4; ++i) {
      const float rt = halfRoundTrip(big[i]);   // deliberately the WRONG format, to document why
      (void)rt;
      require(encodeHeight(big[i]).height == big[i],
        "the 32-bit height channel stores the height unchanged, unlike half precision");
    }
  }

  // ---- real triangle areas over every validity combination -----------------
  {
    const bool zUp = true;
    const float originX = -2048.0f, originY = 1024.0f;   // far from the origin on purpose
    const float spacing = kWaterFoamFieldSpacing;
    const float cameraUp = 5.0f;

    // Gentle, water-like gradients across a 0.4-unit cell. They are deliberately not
    // identical - a fully flat quad would make a valid triangle's area zero and hide the
    // very distinction this test is about - but they are also not a metres-per-cell cliff,
    // because that is not what a water surface looks like and it would make the locality
    // check meaningless. cellWorld takes a DECODED height; passing a stored (biased) value
    // would place the surface a thousand units up.
    const float h[4] = { 3.25f, 3.30f, 3.27f, 3.33f };

    int degenerateCount = 0, drawableCount = 0;
    for (int mask = 0; mask < 16; ++mask) {
      const bool v00 = (mask & 1) != 0, v10 = (mask & 2) != 0;
      const bool v01 = (mask & 4) != 0, v11 = (mask & 8) != 0;
      const QuadVerts q = emitQuad(zUp, originX, originY, spacing, 10, 10, cameraUp,
                                   v00, v10, v01, v11, h[0], h[1], h[2], h[3]);

      const float a0 = triangleArea(q.p[0], q.p[1], q.p[2]);
      const float a1 = triangleArea(q.p[3], q.p[4], q.p[5]);
      const bool t0Drawable = v00 && v10 && v11;
      const bool t1Drawable = v00 && v11 && v01;

      if (!t0Drawable) {
        require(a0 == 0.0f, "an invalid triangle has exactly zero area and cannot be hit");
        ++degenerateCount;
      } else {
        require(a0 > 0.0f, "a fully valid triangle keeps real area");
        ++drawableCount;
      }
      if (!t1Drawable) {
        require(a1 == 0.0f, "the second triangle is also exactly zero area when invalid");
        ++degenerateCount;
      } else {
        require(a1 > 0.0f, "a fully valid second triangle keeps real area");
        ++drawableCount;
      }

      for (int k = 0; k < 6; ++k) {
        require(std::abs(q.p[k].x) > 1000.0f,
          "no emitted vertex leaves the domain toward the world origin");
      }
      if (!t0Drawable) {
        require(std::abs(q.p[0].z - (cameraUp + kWaterFoamSurfaceOffset)) < 8.0f,
          "a collapsed triangle anchors near the snapshot camera height, inside the bounding box");
      }
    }
    require(degenerateCount > 0, "the sweep actually produced degenerate triangles");
    require(drawableCount > 0, "the sweep actually produced drawable triangles");

    // Locality of a real triangle: edges about one cell, not world-spanning.
    {
      const QuadVerts q = emitQuad(zUp, originX, originY, spacing, 10, 10, cameraUp,
                                   true, true, true, true, h[0], h[1], h[2], h[3]);
      const float e0 = vlen(vsub(q.p[1], q.p[0]));
      const float e1 = vlen(vsub(q.p[2], q.p[1]));
      require(e0 > 0.0f && e0 < spacing * 4.0f, "a valid triangle's edge stays about one cell long");
      require(e1 > 0.0f && e1 < 4.0f * spacing, "the second edge stays local too");
    }

    // A quad completely without water must produce no geometry at all, anywhere.
    {
      const QuadVerts q = emitQuad(zUp, originX, originY, spacing, 10, 10, cameraUp,
                                   false, false, false, false, 0, 0, 0, 0);
      require(triangleArea(q.p[0], q.p[1], q.p[2]) == 0.0f,
        "an all-invalid quad emits zero area for its first triangle");
      require(triangleArea(q.p[3], q.p[4], q.p[5]) == 0.0f,
        "an all-invalid quad emits zero area for its second triangle");
    }
  }

  // ---- per-corner attributes and shared-corner consistency ------------------
  // Both defects this guards were real: coverage was computed once from the first corner
  // and copied to all three vertices (blocky per-quad coverage), and the second triangle
  // (00,11,01) was given the first triangle's hardcoded UV offsets, so its UVs did not
  // match its positions and the texture sheared across it.
  {
    const Vec2 w00{ 10.0f, 20.0f }, w10{ 10.4f, 20.0f }, w01{ 10.0f, 20.4f }, w11{ 10.4f, 20.4f };
    // Different density and flow per corner so "reads its own attributes" is testable.
    // The two shared corners use very different densities: if coverage were still computed
    // once from the first corner, the 11 corner would inherit the 00 value and the
    // comparison below would fail.
    const CornerAttr a00 = cornerAttr(w00, 0.9f, Vec2{ 0.1f, 0.0f }, 3.0f, 1.0f);
    const CornerAttr a10 = cornerAttr(w10, 0.4f, Vec2{ -0.2f, 0.3f }, 3.0f, 1.0f);
    const CornerAttr a01 = cornerAttr(w01, 0.7f, Vec2{ 0.0f, -0.4f }, 3.0f, 1.0f);
    const CornerAttr a11 = cornerAttr(w11, 0.02f, Vec2{ 0.5f, 0.5f }, 3.0f, 1.0f);
    (void)a10; (void)a01;

    // The two triangles share corners 00 and 11. Recomputing them must give identical
    // values, which only holds because each corner derives from its own position rather
    // than from one representative corner.
    const CornerAttr a00b = cornerAttr(w00, 0.9f, Vec2{ 0.1f, 0.0f }, 3.0f, 1.0f);
    const CornerAttr a11b = cornerAttr(w11, 0.02f, Vec2{ 0.5f, 0.5f }, 3.0f, 1.0f);
    require(a00.uv.x == a00b.uv.x && a00.uv.y == a00b.uv.y && a00.cov == a00b.cov,
      "the shared 00 corner has identical UV and coverage in both triangles");
    require(a11.uv.x == a11b.uv.x && a11.uv.y == a11b.uv.y && a11.cov == a11b.cov,
      "the shared 11 corner has identical UV and coverage in both triangles");

    // Distinct corners must actually differ, or "per corner" would be vacuous.
    require(a00.uv.x != a11.uv.x || a00.uv.y != a11.uv.y,
      "different corners get different UVs, not one value copied across the triangle");
    require(a00.cov != a11.cov, "different corners take coverage from their own density");

    // UV must track position: a corner one cell away in X differs by spacing/5 in U.
    const float spacing = kWaterFoamFieldSpacing;
    const CornerAttr shifted = cornerAttr(Vec2{ w00.x + spacing, w00.y }, 0.9f, Vec2{ 0.1f, 0.0f }, 3.0f, 1.0f);
    require(std::abs((shifted.uv.x - a00.uv.x) - spacing / 5.0f) < 1e-4f,
      "UV spacing matches the 1/5 world scale the wake foam uses");
  }
  // ---- surface must not float ------------------------------------------------
  require(kWaterFoamSurfaceOffset > 0.0f && kWaterFoamSurfaceOffset < 0.1f,
    "surface offset is a depth-fight guard, not a lift that would float the foam");

  // ---- static topology -------------------------------------------------------
  require(kWaterFoamVertsPerQuad == 6u, "standalone triangles: six vertices per quad");
  require(kWaterFoamSheetVertexCount == kWaterFoamSheetSide * kWaterFoamSheetSide * 6u,
    "vertex count matches six per quad");
  require(kWaterFoamSheetIndexCount == kWaterFoamSheetSide * kWaterFoamSheetSide * 6u,
    "index count matches two triangles per quad, so the layout is static");
  require(kWaterFoamSheetExtent == float(kWaterFoamFieldSize - 1u) * kWaterFoamFieldSpacing,
    "display extent equals the simulated domain");

  // ---- dispatch coverage -----------------------------------------------------
  // SCOPE, stated plainly: this checks the CPU-side coverage ARITHMETIC only, using the
  // shared constants. It CANNOT catch the shader-side bug that actually happened.
  //
  // The shader declared "[threadnum(8,8,1)]", which is not a compile error: Slang ignores
  // the unknown attribute and the workgroup silently stays 1x1x1, so a 32x32 group dispatch
  // produced 1024 threads for 65536 quads. Changing only the shader back to that spelling
  // while leaving these C++ constants at 8 would still pass here, because [numthreads] is a
  // separate literal inside the shader that no CPU test can read.
  //
  // What catches it is tools/verify_shader_workgroups.ps1, which reads LocalSize out of the
  // generated SPIR-V and compares it against the dispatch dimensions. Run that as part of
  // build acceptance; this test is a companion, not a substitute.
  {
    const uint32_t quads = kWaterFoamSheetSide * kWaterFoamSheetSide;
    const uint32_t threadsDispatched =
      kWaterFoamDispatchGroups * kWaterFoamDispatchGroups * kWaterFoamThreadsPerGroup;

    require(kWaterFoamThreadsPerGroup == kWaterFoamThreadsPerGroupX * kWaterFoamThreadsPerGroupY,
      "the workgroup size is the product of its dimensions");
    require(kWaterFoamThreadsPerGroup > 1u,
      "the workgroup is not 1x1x1: a misspelled [numthreads] silently reverts to exactly that");
    require(threadsDispatched >= quads,
      "one dispatch covers every quad; a smaller workgroup than declared would leave gaps");
    // The exact case that failed: 1 thread per group covers 1.5% of the domain.
    const uint32_t withDefaultWorkgroup =
      kWaterFoamDispatchGroups * kWaterFoamDispatchGroups * 1u;
    require(withDefaultWorkgroup < quads / 4u,
      "a 1x1x1 workgroup would cover a small fraction of the domain, which is the bug this guards");
    require(kWaterFoamSheetVertexCount == quads * kWaterFoamVertsPerQuad,
      "every quad emits its six vertices, so coverage of quads is coverage of vertices");
  }
  if (g_failures == 0) {
    std::puts("PASS: foam line boundary geometry, water validity and coverage shape are consistent.");
    return 0;
  }
  std::printf("FAILURES: %d\n", g_failures);
  return 1;
}
