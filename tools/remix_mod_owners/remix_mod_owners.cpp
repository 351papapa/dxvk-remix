// remix_mod_owners: writes the ownership files of a Remix mod, one per game asset it replaces.
//
//   remix_mod_owners <mod folder> [--owners <folder>] [--name <owner>] [--check] [--prune]
//
// The mod folder holds mod.usda (or mod.usd / mod.usdc). The tool opens its composed stage (sublayers
// included), collects the mat_ / mesh_ / light_ prims the runtime replaces (src/lssusd/mod_asset_keys.h),
// and writes <owners>/<kind>_<HASH> holding the owner name, which is the mod folder's name by default.
// <owners> defaults to the owners folder beside the mod folder (rtx-remix/mods/owners).
// How the runtime uses the files: documentation/RemixModOrder.md. How to package a mod: documentation/RemixModPackaging.md.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "../../src/lssusd/mod_asset_keys.h"

namespace fs = std::filesystem;

namespace {
  struct Options {
    fs::path modDir;
    fs::path ownersDir;
    std::string owner;
    bool check = false;
    bool prune = false;
  };

  void printUsage() {
    std::printf(
      "Usage: remix_mod_owners <mod folder> [--owners <folder>] [--name <owner>] [--check] [--prune]\n"
      "\n"
      "Writes one ownership file per game asset the mod replaces, so a mod manager's file\n"
      "conflict order decides which Remix mod wins an asset that several mods replace.\n"
      "\n"
      "  <mod folder>       the folder holding mod.usda (mod.usd, mod.usdc), e.g. rtx-remix\\mods\\MyMod\n"
      "  --owners <folder>  where the files go. Default: the owners folder beside the mod folder\n"
      "  --name <owner>     the name written into the files. Default: the mod folder's name\n"
      "  --check            report what would be written and which files name another mod, write nothing\n"
      "  --prune            delete ownership files that name this mod for assets it no longer replaces\n");
  }

  std::string lowerCase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
    return text;
  }

  // Returns the first line of a file without surrounding white space, or an empty string.
  std::string readOwner(const fs::path& file) {
    std::ifstream in(file);
    std::string line;
    if (!in || !std::getline(in, line)) {
      return {};
    }
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
      line.erase(0, 3);
    }
    const size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      return {};
    }
    return line.substr(first, line.find_last_not_of(" \t\r\n") - first + 1);
  }

  // Returns the mod file in a folder, matched the way the runtime matches it, or an empty path.
  fs::path findModFile(const fs::path& modDir) {
    static const char* const kExtensions[] = { ".usda", ".usd", ".usdc" };
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(modDir, ec)) {
      const fs::path& path = entry.path();
      if (path.stem() != "mod") {
        continue;
      }
      for (const char* extension : kExtensions) {
        if (path.extension() == extension) {
          return path;
        }
      }
    }
    return {};
  }

  bool parseArguments(int argc, wchar_t** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
      const std::wstring arg = argv[i];
      if (arg == L"--check") {
        options.check = true;
      } else if (arg == L"--prune") {
        options.prune = true;
      } else if (arg == L"--owners" && i + 1 < argc) {
        options.ownersDir = argv[++i];
      } else if (arg == L"--name" && i + 1 < argc) {
        options.owner = fs::path(argv[++i]).u8string();
      } else if (!arg.empty() && arg[0] != L'-' && options.modDir.empty()) {
        options.modDir = arg;
      } else {
        return false;
      }
    }
    return !options.modDir.empty();
  }
}

int wmain(int argc, wchar_t** argv) {
  Options options;
  if (!parseArguments(argc, argv, options)) {
    printUsage();
    return 1;
  }

  std::error_code ec;
  const fs::path modDir = fs::absolute(options.modDir, ec).lexically_normal();
  const fs::path modFile = findModFile(modDir);
  if (modFile.empty()) {
    std::fprintf(stderr, "No mod.usda, mod.usd or mod.usdc in %s\n", modDir.u8string().c_str());
    return 1;
  }
  // A trailing separator leaves an empty filename, which lexically_normal turns into a trailing "/".
  const fs::path modFolder = modDir.has_filename() ? modDir : modDir.parent_path();
  if (options.owner.empty()) {
    options.owner = modFolder.filename().u8string();
  }
  const fs::path ownersDir = options.ownersDir.empty() ? modFolder.parent_path() / "owners"
                                                       : fs::absolute(options.ownersDir, ec).lexically_normal();

  const pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(modFile.u8string(), pxr::UsdStage::LoadNone);
  if (!stage) {
    std::fprintf(stderr, "Could not open %s\n", modFile.u8string().c_str());
    return 2;
  }
  const lss::ModAssetHashes assets = lss::collectModAssetHashes(stage);

  size_t total = 0;
  for (const auto& hashes : assets) {
    total += hashes.size();
  }
  std::printf("%s replaces %zu materials, %zu meshes and %zu lights.\n", options.owner.c_str(),
              assets[lss::kModAssetMaterial].size(), assets[lss::kModAssetMesh].size(), assets[lss::kModAssetLight].size());

  if (!options.check && total > 0) {
    fs::create_directories(ownersDir, ec);
    if (ec) {
      std::fprintf(stderr, "Could not create %s: %s\n", ownersDir.u8string().c_str(), ec.message().c_str());
      return 1;
    }
  }

  const std::string ownerKey = lowerCase(options.owner);
  size_t written = 0;
  size_t unchanged = 0;
  size_t taken = 0;
  for (size_t kind = 0; kind < lss::kModAssetKindCount; ++kind) {
    for (const uint64_t hash : assets[kind]) {
      const std::string fileName = lss::modOwnerFileName(kind, hash);
      const fs::path file = ownersDir / fileName;
      const std::string existing = readOwner(file);
      if (lowerCase(existing) == ownerKey) {
        ++unchanged;
        continue;
      }
      if (!existing.empty()) {
        // Another installed mod replaces this asset too. The mod manager decides which file stays.
        ++taken;
        std::printf("  %s is owned by %s%s\n", fileName.c_str(), existing.c_str(), options.check ? "" : ", overwritten");
      }
      if (options.check) {
        continue;
      }
      std::ofstream out(file, std::ios::binary | std::ios::trunc);
      out << options.owner << "\n";
      if (!out) {
        std::fprintf(stderr, "Could not write %s\n", file.u8string().c_str());
        return 1;
      }
      ++written;
    }
  }

  size_t pruned = 0;
  if (options.prune && fs::is_directory(ownersDir, ec)) {
    for (const auto& entry : fs::directory_iterator(ownersDir, ec)) {
      const std::string fileName = entry.path().filename().u8string();
      bool replaced = false;
      bool isOwnerFile = false;
      for (size_t kind = 0; kind < lss::kModAssetKindCount && !isOwnerFile; ++kind) {
        const uint64_t hash = lss::parseModAssetHash(fileName, lss::modAssetPrefix(kind));
        if (hash != 0 && lss::modOwnerFileName(kind, hash) == fileName) {
          isOwnerFile = true;
          replaced = std::binary_search(assets[kind].begin(), assets[kind].end(), hash);
        }
      }
      if (!isOwnerFile || replaced || lowerCase(readOwner(entry.path())) != ownerKey) {
        continue;
      }
      std::printf("  %s no longer replaced%s\n", fileName.c_str(), options.check ? "" : ", deleted");
      if (!options.check) {
        fs::remove(entry.path(), ec);
      }
      ++pruned;
    }
  }

  if (options.check) {
    std::printf("Checked %zu ownership files in %s: %zu would be written, %zu name another mod, %zu would be pruned.\n",
                total, ownersDir.u8string().c_str(), total - unchanged, taken, pruned);
  } else {
    std::printf("Wrote %zu ownership files to %s (%zu already named %s, %zu were another mod's, %zu pruned).\n",
                written, ownersDir.u8string().c_str(), unchanged, options.owner.c_str(), taken, pruned);
  }
  return 0;
}
