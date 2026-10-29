// SPDX-License-Identifier: MIT
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "../../../src/dxvk/rtx_render/rtx_grass_placement.h"

using namespace dxvk;
static void require(bool value, const char* message) {
  if (!value) { throw std::runtime_error(message); }
}

int main() {
  try {
    const std::vector<std::array<Vector3, 3>> ground = {
      {Vector3(0,0,0), Vector3(10,0,0), Vector3(10,10,0)},
      {Vector3(0,0,0), Vector3(10,10,0), Vector3(0,10,0)}
    };
    std::atomic<bool> cancel {false};
    auto build = [&](const auto& geometry, const Matrix4& world, bool zUp, bool inset = false, float density = 4) {
      return GrassPatchBuilder::build(geometry, world, .70710678f, density, 1.0f, inset, zUp, cancel);
    };
    const auto roots = build(ground, Matrix4(), true);
    const auto repeat = build(ground, Matrix4(), true);
    require(roots.size() == 400, "Area-weighted placement count is incorrect");
    require(roots.size() == repeat.size() && !std::memcmp(roots.data(), repeat.data(), roots.size()*sizeof(GrassRoot)), "Placement is nondeterministic");
    for (const auto& root : roots) {
      require(root.positionScale.x >= 0 && root.positionScale.x <= 10 && root.positionScale.y >= 0 && root.positionScale.y <= 10, "Root escaped its triangle union");
      require(root.positionScale.z == 0 && root.normalAngle.z > .999f, "Ground orientation incorrect");
    }
    const auto inset = build(ground, Matrix4(), true, true, 10);
    bool crossesSharedEdge = false;
    require(!inset.empty() && inset.size() < 1000, "Boundary inset had no effect");
    for (const auto& root : inset) {
      const float radius = root.positionScale.w;
      require(root.positionScale.x >= radius && root.positionScale.x <= 10-radius && root.positionScale.y >= radius && root.positionScale.y <= 10-radius, "Root overlaps an open edge");
      crossesSharedEdge |= std::abs(root.positionScale.x-root.positionScale.y) < .1f;
    }
    require(crossesSharedEdge, "Shared triangle edge was treated as a hole");
    Matrix4 yup;
    yup[1] = Vector4(0,0,-1,0); yup[2] = Vector4(0,1,0,0);
    const auto yRoots = build(ground,yup,false);
    require(yRoots.size() == 400, "Y-up slope rejected horizontal ground");
    for (const auto& root : yRoots) { require(root.positionScale.y == 0 && root.normalAngle.y > .999f, "Y-up roots have wrong orientation"); }
    Matrix4 mirror;
    mirror[0].x = -1;
    const auto mirrored = build(ground,mirror,true);
    require(mirrored.size() == 400, "Mirrored ground lost upward winding");
    Matrix4 wall;
    wall[0] = Vector4(0,0,1,0); wall[2] = Vector4(-1,0,0,0);
    require(build(ground,wall,true).empty(), "Vertical wall accepted as ground");
    auto invalid = ground;
    invalid[0][0].x = std::numeric_limits<float>::quiet_NaN();
    require(build(invalid,Matrix4(),true).size() == 200, "Invalid triangle contaminated other geometry");
    require(build(ground,Matrix4(),true,false,std::numeric_limits<float>::quiet_NaN()).empty(), "NaN density accepted");
    require(build(ground,Matrix4(),true,false,0).empty(), "Zero density generated roots");
    const auto capped = build(ground,Matrix4(),true,false,1000000);
    require(capped.size() == GrassPatchBuilder::kMaxPatchPoints, "Patch root cap failed");
    cancel = true;
    require(build(ground,Matrix4(),true).empty(), "Cancelled generation published roots");
    std::cout << "Grass placement: density, bounds, welded edges, Y/Z-up, mirrored transforms, slope, invalid inputs, cap and cancellation passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
