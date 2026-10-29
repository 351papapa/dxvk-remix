// Tests the shared GPU foam ABI, world slots, scene bounds and diagnostic attributes.
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <string>
#include <limits>
#include <vector>

#include "../../../src/dxvk/rtx_render/rtx_water_foam_sheet_contract.h"

namespace {

int g_failures = 0;

void require(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// True when (x,y,z) lies inside `b`, with tolerance for float rounding.
bool inside(const dxvk::WaterFoamBox& b, float x, float y, float z) {
  const float e = 1e-3f;
  return x >= b.minX - e && x <= b.maxX + e &&
         y >= b.minY - e && y <= b.maxY + e &&
         z >= b.minZ - e && z <= b.maxZ + e;
}

} // namespace

int main() {
  using namespace dxvk;

  for (const unsigned int side : {60u, 90u, 256u}) {
    for (int origin = -2 * int(side); origin <= 2 * int(side); ++origin) {
      std::vector<bool> slots(side, false);
      for (unsigned int local = 0; local < side; ++local) {
        const int cell = origin + int(local);
        const auto slot = waterFoamWorldCellSlot(cell, side);
        const auto expected = (int64_t(cell) % side + side) % side;
        require(slot < side && slot == expected, "world-cell slot matches Euclidean modulo");
        if (slot < side) {
          require(!slots[slot], "each world cell has a unique output slot across the origin");
          slots[slot] = true;
        }
      }
    }
    for (const int cell : {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
      require(waterFoamWorldCellSlot(cell, side) == (int64_t(cell) % side + side) % side,
        "world-cell slot handles signed coordinate limits");
    }
  }

  // ---- layout contracts with water_foam_surface.comp.slang ----
  require(sizeof(WaterFoamGpuVertex) == 36, "WaterFoamGpuVertex stride is 36 bytes");
  require(offsetof(WaterFoamGpuVertex, position) == 0, "position is at offset 0");
  require(offsetof(WaterFoamGpuVertex, normal) == 12, "normal is at offset 12");
  require(offsetof(WaterFoamGpuVertex, texcoord) == 24, "texcoord is at offset 24");
  require(offsetof(WaterFoamGpuVertex, color) == 32, "color is at offset 32");

  require(offsetof(WaterFoamSurfaceArgs, worldToClip) == 0, "worldToClip is first");
  require(offsetof(WaterFoamSurfaceArgs, vertexCount) == 64, "vertexCount is at offset 64");
  require(offsetof(WaterFoamSurfaceArgs, sheetSide) == 68, "sheetSide is at offset 68");
  require(offsetof(WaterFoamSurfaceArgs, sheetExtent) == 72, "sheetExtent is at offset 72");
  require(offsetof(WaterFoamSurfaceArgs, upAxisIsZ) == 76, "upAxisIsZ is at offset 76");
  require(offsetof(WaterFoamSurfaceArgs, gridOrigin) == 80, "gridOrigin is at offset 80");
  require(offsetof(WaterFoamSurfaceArgs, gridSpacing) == 88, "gridSpacing is at offset 88");
  require(sizeof(WaterFoamSurfaceArgs) <= 128, "push constant block fits the guaranteed size");

  // ---- topology and buffer sizing ----
  require(kWaterFoamFieldSize == 257u, "density field is 257x257 cells");
  require(kWaterFoamSheetVertexCount == 393216u, "six standalone vertices per density quad");
  require(kWaterFoamSheetIndexCount == 393216u, "256x256 quads of two triangles");
  require(kWaterFoamSheetSide == 256u, "256 quads across");
  require(kWaterFoamVertsPerQuad == 6u, "standalone triangles: six vertices per quad");
  require(kWaterFoamSheetVertexCount * sizeof(WaterFoamGpuVertex) == 14155776u,
    "vertex buffer size matches vertex count");
  require(kWaterFoamSheetIndexCount * sizeof(uint32_t) == 1572864u,
    "index staging size matches index count");
  require(kWaterFoamSheetExtent == 102.4f, "display domain is the 257-cell field at 0.4 spacing");

  // ---- stable identity hashes ----
  const uint64_t sheetHashes[] = { kWaterFoamSheetHashIndices, kWaterFoamSheetHashPosition,
    kWaterFoamSheetHashTexcoord, kWaterFoamSheetHashDescriptor, kWaterFoamSheetHashLayout };
  for (int i = 0; i < 5; ++i) {
    require(sheetHashes[i] != 0, "geometry hash component is initialised, not zero");
    for (int j = i + 1; j < 5; ++j) {
      require(sheetHashes[i] != sheetHashes[j], "geometry hash components are distinct");
    }
  }

  // Bounds call the same helper as the scene geometry snapshot.
  for (bool zUp : {false, true}) {
    for (float centre : {-2048.0f, 0.0f, 1024.0f}) {
      const auto bounds = waterFoamPlacedSheetBox(zUp, -31.2f, 14.4f, centre, 3.0f);
      const auto point = [&](float x, float y, float height) {
        return zUp ? inside(bounds,x,y,height) : inside(bounds,x,height,y);
      };
      require(point(-31.2f,14.4f,centre), "domain origin is inside production bounds");
      require(point(71.2f,116.8f,centre+3.0f), "opposite domain corner and swell fit bounds");
      require(!point(-32.0f,14.4f,centre), "bounds do not drift away from the density origin");
      require(!point(0,20,centre+12.0f), "observed height range limits the vertical bound");
      const auto initial = waterFoamPlacedSheetBox(zUp,-31.2f,14.4f,centre,0.0f);
      require(zUp ? inside(initial,0,20,centre+255) : inside(initial,0,centre+255,20),
        "unobserved carrier range includes the full contact probe reach");
    }
  }
  WaterFoamGpuVertex vertex {{123,-456,789},{0,0.6f,0.8f},{17,-23},0};
  require(waterFoamVertexAttributesValid(vertex), "zero-alpha dynamic foam is valid");
  vertex.texcoord[0] = std::numeric_limits<float>::quiet_NaN();
  require(!waterFoamVertexAttributesValid(vertex), "NaN UV fails GPU sample validation");
  vertex.texcoord[0] = 17;
  vertex.position[2] = std::numeric_limits<float>::infinity();
  require(!waterFoamVertexAttributesValid(vertex), "infinite position fails GPU sample validation");
  vertex.position[2] = 789;
  vertex.normal[1] = vertex.normal[2] = 0;
  require(!waterFoamVertexAttributesValid(vertex), "zero normal fails GPU sample validation");

  if (g_failures == 0) {
    std::puts("PASS: foam sheet contract, topology, hashes and production bbox are consistent.");
    return 0;
  }
  std::printf("FAILURES: %d\n", g_failures);
  return 1;
}
