#pragma once

// The game assets a Remix mod replaces, read from the names of the prims under /RootNode.
// Shared by the runtime (ModManager clash resolution) and the remix_mod_owners packaging tool,
// so both name the same assets. See documentation/RemixModOrder.md.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "usd_include_begin.h"
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include "usd_include_end.h"

namespace lss {

// Indices into ModAssetHashes, in the order of ModAssetKind in rtx_mod_manager.h.
constexpr size_t kModAssetMaterial = 0;
constexpr size_t kModAssetMesh = 1;
constexpr size_t kModAssetLight = 2;
constexpr size_t kModAssetKindCount = 3;

// Sorted, unique hashes per kind.
using ModAssetHashes = std::array<std::vector<uint64_t>, kModAssetKindCount>;

// Returns the prim name prefix of a kind: "mat_", "mesh_" or "light_", as lss::prefix spells them.
inline const char* modAssetPrefix(size_t kind) {
  static const char* const kPrefixes[kModAssetKindCount] = { "mat_", "mesh_", "light_" };
  return kPrefixes[kind];
}

// Returns the hash in a prim name such as "mat_0123456789ABCDEF", parsed as UsdMod parses it.
// Returns 0 when the name does not start with the prefix.
inline uint64_t parseModAssetHash(const std::string& name, const char* namePrefix) {
  const size_t length = std::char_traits<char>::length(namePrefix);
  if (name.compare(0, length, namePrefix) != 0) {
    return 0;
  }
  return std::strtoull(name.c_str() + length, nullptr, 16);
}

// Collects the replaced asset hashes of a composed mod stage: the active children of
// /RootNode/Looks (mat_), /RootNode/meshes (mesh_) and /RootNode/lights (light_, or the
// legacy sphereLight_), the prims UsdMod turns into replacements.
inline ModAssetHashes collectModAssetHashes(const pxr::UsdStageRefPtr& stage) {
  struct Section {
    const char* path;
    size_t kind;
  };
  static const Section kSections[] = {
    { "/RootNode/Looks", kModAssetMaterial },
    { "/RootNode/meshes", kModAssetMesh },
    { "/RootNode/lights", kModAssetLight },
  };

  ModAssetHashes result;
  if (!stage) {
    return result;
  }
  for (const Section& section : kSections) {
    const pxr::UsdPrim root = stage->GetPrimAtPath(pxr::SdfPath(section.path));
    if (!root.IsValid()) {
      continue;
    }
    std::vector<uint64_t>& hashes = result[section.kind];
    for (const pxr::UsdPrim& child : root.GetFilteredChildren(pxr::UsdPrimIsActive)) {
      const std::string& name = child.GetName().GetString();
      uint64_t hash = parseModAssetHash(name, modAssetPrefix(section.kind));
      if (hash == 0 && section.kind == kModAssetLight) {
        hash = parseModAssetHash(name, "sphereLight_");
      }
      if (hash != 0) {
        hashes.push_back(hash);
      }
    }
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
  }
  return result;
}

// Returns the ownership file name of an asset: the kind's prefix and the hash as 16 upper-case hex digits,
// e.g. "mat_0123456789ABCDEF".
inline std::string modOwnerFileName(size_t kind, uint64_t hash) {
  static const char kHex[] = "0123456789ABCDEF";
  std::string name(modAssetPrefix(kind));
  for (int shift = 60; shift >= 0; shift -= 4) {
    name.push_back(kHex[(hash >> shift) & 0xF]);
  }
  return name;
}

} // namespace lss
