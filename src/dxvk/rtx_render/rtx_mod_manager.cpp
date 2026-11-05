/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_mod_manager.h"
#include "rtx_asset_data_manager.h"

#include "rtx_asset_replacer.h"
#include "rtx_mod_usd.h"

#include "rtx_options.h"

#include "../../util/log/log.h"
#include "../../util/util_filesys.h"
#include "../../util/util_string.h"

#include <cctype>
#include <fstream>
#include <set>
#include <unordered_set>
#include <assert.h>
#include <regex>

namespace fs = std::filesystem;

namespace dxvk {

static const std::string kDefaultModFileName = "mod";

static const ModTypeInfo* kModTypeInfos[] = {
  &UsdMod::getTypeInfo(),
};

namespace {
  constexpr size_t kAssetKindCount = static_cast<size_t>(ModAssetKind::Count);
  // Clash lines written to the log per resolve. The Remix menu lists every clash.
  constexpr size_t kMaxClashLogLines = 32;

  const char* assetPrefix(ModAssetKind kind) {
    static const char* const kPrefixes[kAssetKindCount] = { "mat_", "mesh_", "light_" };
    return kPrefixes[static_cast<size_t>(kind)];
  }

  // Returns the asset's prim and ownership file name, e.g. "mat_0123456789ABCDEF".
  std::string assetName(ModAssetKind kind, XXH64_hash_t hash) {
    static const char kHex[] = "0123456789ABCDEF";
    std::string name(assetPrefix(kind));
    for (int shift = 60; shift >= 0; shift -= 4) {
      name.push_back(kHex[(hash >> shift) & 0xF]);
    }
    return name;
  }

  std::string lowerCase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
    return text;
  }

  // Returns the mod folder name in a dependency key or an ownership file line. Either may hold
  // the bare name or a path to the folder or to its mod.usda ("OtherMod", "../OtherMod/mod.usda").
  std::string modFolderName(std::string text) {
    // A UTF-8 byte order mark, quotes and surrounding white space are not part of the name.
    if (text.compare(0, 3, "\xEF\xBB\xBF") == 0) {
      text.erase(0, 3);
    }
    const size_t first = text.find_first_not_of(" \t\r\n\"");
    if (first == std::string::npos) {
      return {};
    }
    text = text.substr(first, text.find_last_not_of(" \t\r\n\"") - first + 1);
    std::replace(text.begin(), text.end(), '\\', '/');
    while (!text.empty() && text.back() == '/') {
      text.pop_back();
    }
    const fs::path path(text);
    if (path.has_extension() && lowerCase(path.stem().string()) == kDefaultModFileName) {
      return path.parent_path().filename().string();
    }
    return path.filename().string();
  }

  const char* clashRuleName(ModClashRule rule) {
    switch (rule) {
    case ModClashRule::OwnerFile: return "ownership file";
    case ModClashRule::Dependency: return "dependency";
    case ModClashRule::Name: return "name";
    case ModClashRule::LoadOrder: return "load order";
    }
    return "load order";
  }

  // Returns true when `mod` depends on `other`, directly or through other mods.
  bool dependsOn(const Mod* mod, const Mod* other,
                 const std::unordered_map<const Mod*, std::vector<const Mod*>>& dependencies) {
    std::vector<const Mod*> stack { mod };
    std::unordered_set<const Mod*> visited;
    while (!stack.empty()) {
      const Mod* current = stack.back();
      stack.pop_back();
      if (!visited.insert(current).second) {
        continue;
      }
      const auto found = dependencies.find(current);
      if (found == dependencies.end()) {
        continue;
      }
      for (const Mod* dependency : found->second) {
        if (dependency == other) {
          return true;
        }
        stack.push_back(dependency);
      }
    }
    return false;
  }
}

ModManager::ModManager() {
  refreshMods();
}

void ModManager::refreshMods() {
  std::vector<std::unique_ptr<Mod>> found = enumerateAllMods();

  // Mods still on disk keep their objects, and with them their loaded replacements.
  std::vector<std::unique_ptr<Mod>> mods;
  mods.reserve(found.size());
  for (auto& foundMod : found) {
    const auto existing = std::find_if(m_mods.begin(), m_mods.end(),
      [&foundMod](const std::unique_ptr<Mod>& mod) { return mod->m_filePath == foundMod->m_filePath; });
    if (existing != m_mods.end()) {
      mods.push_back(std::move(m_mods.extract(existing).value()));
    } else {
      mods.push_back(std::move(foundMod));
    }
  }
  // The mods left in the set were removed from disk.
  m_mods.clear();

  // m_priority is part of the set's ordering, so it is settled before the mods go back in.
  orderMods(mods);
  for (auto& mod : mods) {
    // Mods can only ask for a republish once they know who manages them.
    mod->m_pManager = this;
    m_mods.insert(std::move(mod));
  }

  logModOrder();

  // The set just changed, so the flattened precedence order has too.
  publishSearchPaths();

  std::lock_guard<std::mutex> lock(m_conflictMutex);
  resolveClashes();
}

void ModManager::orderMods(std::vector<std::unique_ptr<Mod>>& mods) {
  // Name order first: a later name has the higher precedence.
  std::sort(mods.begin(), mods.end(), [](const std::unique_ptr<Mod>& a, const std::unique_ptr<Mod>& b) {
    const std::string nameA = lowerCase(a->m_name);
    const std::string nameB = lowerCase(b->m_name);
    if (nameA != nameB) {
      return nameA < nameB;
    }
    return a->m_filePath < b->m_filePath;
  });

  // A dependency names a mod by its folder name, a path to it, or its lightspeed_mod_name.
  const size_t count = mods.size();
  std::unordered_map<std::string, size_t> byName;
  for (size_t i = 0; i < count; ++i) {
    byName.emplace(lowerCase(mods[i]->m_name), i);
  }
  for (size_t i = 0; i < count; ++i) {
    if (!mods[i]->m_displayName.empty()) {
      byName.emplace(lowerCase(mods[i]->m_displayName), i);
    }
  }

  std::vector<std::vector<size_t>> dependents(count);
  std::vector<size_t> unplacedDependencies(count, 0);
  for (size_t i = 0; i < count; ++i) {
    Mod& mod = *mods[i];
    mod.m_resolvedDependencies.clear();
    for (const std::string& dependency : mod.m_dependencies) {
      auto found = byName.find(lowerCase(modFolderName(dependency)));
      if (found == byName.end()) {
        found = byName.find(lowerCase(dependency));
      }
      if (found == byName.end()) {
        Logger::warn(str::format("[ModOrder] ", mod.m_name, " depends on \"", dependency, "\", which is not installed."));
        continue;
      }
      const size_t j = found->second;
      const Mod* pDependency = mods[j].get();
      if (j == i || std::find(mod.m_resolvedDependencies.begin(), mod.m_resolvedDependencies.end(), pDependency) !=
                    mod.m_resolvedDependencies.end()) {
        continue;
      }
      mod.m_resolvedDependencies.push_back(pDependency);
      dependents[j].push_back(i);
      ++unplacedDependencies[i];
    }
  }

  // Each step places the earliest name among the mods whose dependencies are all placed.
  std::set<size_t> ready;
  for (size_t i = 0; i < count; ++i) {
    if (unplacedDependencies[i] == 0) {
      ready.insert(i);
    }
  }
  std::vector<bool> placed(count, false);
  std::vector<size_t> order;
  order.reserve(count);
  while (order.size() < count) {
    if (ready.empty()) {
      // Only dependency cycles are left. The earliest name is placed as if it had no dependencies.
      const size_t forced = static_cast<size_t>(std::find(placed.begin(), placed.end(), false) - placed.begin());
      Logger::warn(str::format("[ModOrder] ", mods[forced]->m_name, " is part of a dependency cycle. It is ordered by name."));
      ready.insert(forced);
    }
    const size_t i = *ready.begin();
    ready.erase(ready.begin());
    if (placed[i]) {
      continue;
    }
    placed[i] = true;
    order.push_back(i);
    for (const size_t dependent : dependents[i]) {
      if (!placed[dependent] && unplacedDependencies[dependent] > 0 && --unplacedDependencies[dependent] == 0) {
        ready.insert(dependent);
      }
    }
  }

  std::vector<std::unique_ptr<Mod>> ordered;
  ordered.reserve(count);
  for (size_t rank = 0; rank < count; ++rank) {
    mods[order[rank]]->m_priority = rank;
    ordered.push_back(std::move(mods[order[rank]]));
  }
  mods = std::move(ordered);
}

void ModManager::logModOrder() const {
  Logger::info(str::format("[ModOrder] ", m_mods.size(), " Remix mod(s), from lowest to highest precedence:"));
  for (const auto& mod : m_mods) {
    std::string line = str::format("[ModOrder]   ", mod->m_priority + 1, ". ", mod->m_name);
    if (!mod->m_displayName.empty() || !mod->m_version.empty()) {
      line += str::format(" (\"", mod->m_displayName, "\" ", mod->m_version, ")");
    }
    if (!mod->m_resolvedDependencies.empty()) {
      line += ", depends on";
      for (const Mod* dependency : mod->m_resolvedDependencies) {
        line += " " + dependency->m_name;
      }
    }
    Logger::info(line);
  }
}

void ModManager::publishSearchPaths() const {
  // Mod order is precedence order. m_mods is sorted by ComparePtrs, so concatenating
  // in iteration order gives ascending precedence and the last mod wins.
  //
  // Lock order, three subsystems deep: ModManager::m_searchPathMutex (here) ->
  // AssetDataManager::m_searchPathMutex -> FileWatch::m_mutex -> FileWatch::m_requestsMutex.
  // Keep any lock taken from within this call, or setSearchPaths, consistent with that order.
  std::vector<Mod::Path> flattened;
  std::lock_guard<std::mutex> lock(m_searchPathMutex);
  for (const auto& mod : m_mods) {
    const auto& paths = mod->searchPaths();
    flattened.insert(flattened.end(), paths.begin(), paths.end());
  }
  AssetDataManager::get().setSearchPaths(flattened);
}

void ModManager::setReplacedAssets(Mod& mod, ModAssetSet assets) {
  std::lock_guard<std::mutex> lock(m_conflictMutex);
  mod.m_replacedAssets = std::move(assets);
  resolveClashes();
}

void ModManager::resolveClashes() {
  std::vector<Mod*> order;
  order.reserve(m_mods.size());
  for (const auto& mod : m_mods) {
    order.push_back(mod.get());
  }

  ModConflictReport report;
  std::unordered_map<const Mod*, std::vector<const Mod*>> dependencies;
  for (const Mod* mod : order) {
    ModConflictReport::ModEntry entry;
    entry.name = mod->m_name;
    entry.displayName = mod->m_displayName;
    entry.version = mod->m_version;
    for (const Mod* dependency : mod->m_resolvedDependencies) {
      entry.dependencies.push_back(dependency->m_name);
    }
    for (size_t kind = 0; kind < kAssetKindCount; ++kind) {
      entry.replacedCounts[kind] = mod->m_replacedAssets[kind].size();
    }
    report.mods.push_back(std::move(entry));
    dependencies.emplace(mod, mod->m_resolvedDependencies);
  }

  auto winners = std::make_unique<ClashWinners>();
  if (order.size() > 1) {
    for (size_t kind = 0; kind < kAssetKindCount; ++kind) {
      const ModAssetKind assetKind = static_cast<ModAssetKind>(kind);

      // The first mod in precedence order that replaces each hash, and all of them for the hashes that clash.
      fast_unordered_cache<size_t> firstProvider;
      fast_unordered_cache<std::vector<size_t>> providers;
      std::vector<XXH64_hash_t> clashHashes;
      for (size_t m = 0; m < order.size(); ++m) {
        for (const XXH64_hash_t hash : order[m]->m_replacedAssets[kind]) {
          const auto [first, inserted] = firstProvider.try_emplace(hash, m);
          if (inserted) {
            continue;
          }
          std::vector<size_t>& clashProviders = providers[hash];
          if (clashProviders.empty()) {
            clashProviders.push_back(first->second);
            clashHashes.push_back(hash);
          }
          clashProviders.push_back(m);
        }
      }
      std::sort(clashHashes.begin(), clashHashes.end());

      for (const XXH64_hash_t hash : clashHashes) {
        const std::vector<size_t>& clashProviders = providers[hash];
        ModConflictReport::Clash clash;
        clash.kind = assetKind;
        clash.hash = hash;
        for (const size_t m : clashProviders) {
          clash.mods.push_back(order[m]->m_name);
        }

        // An ownership file decides when it names one of the mods replacing the asset.
        Mod* winner = nullptr;
        const std::string& owner = readOwnerFile(assetKind, hash);
        if (!owner.empty()) {
          const std::string ownerKey = lowerCase(owner);
          for (const size_t m : clashProviders) {
            if (lowerCase(order[m]->m_name) == ownerKey) {
              winner = order[m];
            }
          }
          if (winner == nullptr) {
            clash.staleOwner = owner;
          }
        }

        if (winner != nullptr) {
          clash.rule = ModClashRule::OwnerFile;
        } else {
          // Otherwise the precedence order decides, and the rule names what put the winner last.
          winner = order[clashProviders.back()];
          bool dependsOnAll = true;
          bool laterNameThanAll = true;
          for (const size_t m : clashProviders) {
            if (order[m] == winner) {
              continue;
            }
            dependsOnAll = dependsOnAll && dependsOn(winner, order[m], dependencies);
            laterNameThanAll = laterNameThanAll && lowerCase(winner->m_name) > lowerCase(order[m]->m_name);
          }
          clash.rule = dependsOnAll ? ModClashRule::Dependency
                                    : laterNameThanAll ? ModClashRule::Name : ModClashRule::LoadOrder;
        }
        clash.winner = winner->m_name;
        winners->winners[kind].emplace(hash, winner);
        report.clashes.push_back(std::move(clash));
      }
    }
  }

  // Swapped under the lock. The previous table is destroyed after it is released.
  bool changed = false;
  {
    std::lock_guard<sync::Spinlock> lock(m_clashWinnersLock);
    for (size_t kind = 0; kind < kAssetKindCount && !changed; ++kind) {
      const auto& next = winners->winners[kind];
      if (m_clashWinners == nullptr) {
        changed = !next.empty();
        continue;
      }
      const auto& previous = m_clashWinners->winners[kind];
      changed = previous.size() != next.size() ||
        std::any_of(next.begin(), next.end(), [&previous](const auto& entry) {
          const auto found = previous.find(entry.first);
          return found == previous.end() || found->second != entry.second;
        });
    }
    std::swap(m_clashWinners, winners);
    m_hasClashes.store(!report.clashes.empty(), std::memory_order_release);
  }

  if (changed) {
    m_clashesChanged.store(true, std::memory_order_release);
    if (report.clashes.empty()) {
      Logger::info("[ModOrder] No asset is replaced by more than one mod.");
    } else {
      Logger::info(str::format("[ModOrder] ", report.clashes.size(), " asset(s) replaced by more than one mod:"));
      for (size_t i = 0; i < report.clashes.size() && i < kMaxClashLogLines; ++i) {
        const ModConflictReport::Clash& clash = report.clashes[i];
        std::string losers;
        for (const std::string& mod : clash.mods) {
          if (mod != clash.winner) {
            losers += (losers.empty() ? "" : ", ") + mod;
          }
        }
        std::string line = str::format("[ModOrder]   ", assetName(clash.kind, clash.hash), ": ", clash.winner,
                                       " wins over ", losers, " (", clashRuleName(clash.rule), ")");
        if (!clash.staleOwner.empty()) {
          line += str::format(". Its ownership file names ", clash.staleOwner, ", which does not replace it");
        }
        Logger::info(line);
      }
      if (report.clashes.size() > kMaxClashLogLines) {
        Logger::info(str::format("[ModOrder]   ... and ", report.clashes.size() - kMaxClashLogLines,
                                 " more. The Remix menu lists them all (Enhancements, Mod Conflicts)."));
      }
    }
  }

  m_report = std::move(report);
}

const std::string& ModManager::readOwnerFile(ModAssetKind kind, XXH64_hash_t hash) {
  auto& cache = m_ownerFiles[static_cast<size_t>(kind)];
  const auto cached = cache.find(hash);
  if (cached != cache.end()) {
    return cached->second;
  }

  std::string owner;
  const std::string fileName = assetName(kind, hash);
  for (const Path& modsDir : m_modsDirs) {
    std::ifstream file(modsDir / "owners" / fileName);
    if (!file) {
      continue;
    }
    std::string line;
    std::getline(file, line);
    owner = modFolderName(line);
    if (!owner.empty()) {
      break;
    }
  }
  return cache.emplace(hash, std::move(owner)).first->second;
}

void ModManager::forgetOwnerFiles() {
  std::lock_guard<std::mutex> lock(m_conflictMutex);
  for (auto& cache : m_ownerFiles) {
    cache.clear();
  }
}

ModConflictReport ModManager::conflictReport() const {
  std::lock_guard<std::mutex> lock(m_conflictMutex);
  return m_report;
}

Mod* ModManager::clashWinner(ModAssetKind kind, XXH64_hash_t hash) const {
  if (!m_hasClashes.load(std::memory_order_acquire)) {
    return nullptr;
  }
  std::lock_guard<sync::Spinlock> lock(m_clashWinnersLock);
  if (m_clashWinners == nullptr) {
    return nullptr;
  }
  const auto& winners = m_clashWinners->winners[static_cast<size_t>(kind)];
  const auto found = winners.find(hash);
  return found != winners.end() ? found->second : nullptr;
}

bool Mod::isLoadingCancelled() const {
  return m_pManager != nullptr && m_pManager->isLoadingCancelled();
}

void Mod::setSearchPaths(std::vector<Path> paths) {
  if (m_pManager != nullptr) {
    {
      std::lock_guard<std::mutex> lock(m_pManager->m_searchPathMutex);
      m_searchPaths = std::move(paths);
    }
    m_pManager->publishSearchPaths();
    return;
  }
  m_searchPaths = std::move(paths);
}

void Mod::setReplacedAssets(ModAssetSet assets) {
  if (m_pManager != nullptr) {
    m_pManager->setReplacedAssets(*this, std::move(assets));
    return;
  }
  m_replacedAssets = std::move(assets);
}

std::string ModManager::getBaseGameModPath(std::string baseGameModRegexStr, std::string baseGameModPathRegexStr) {
  std::string commandLine = GetCommandLineA();
  std::smatch match;
  std::string baseGameModPath = "";

  if (baseGameModRegexStr != "" && std::regex_search(commandLine, match, std::regex(baseGameModRegexStr, std::regex::icase))) {
    std::regex baseGameModPathRegex(baseGameModPathRegexStr, std::regex::icase);
    if (regex_search(commandLine, match, baseGameModPathRegex)) {
      baseGameModPath = match.str(1);
      std::replace(baseGameModPath.begin(), baseGameModPath.end(), '\\', '/');
    }
  }
  return baseGameModPath;
}

std::vector<std::unique_ptr<Mod>> ModManager::enumerateAllMods() {
  std::vector<std::unique_ptr<Mod>> mods;
  
  const std::string modsPath = util::RtxFileSys::path(util::RtxFileSys::Mods).string();
  
  const Path defaultModsDir = Path(modsPath).lexically_normal();

  std::string baseGameModPath = getBaseGameModPath(RtxOptions::baseGameModRegex(), RtxOptions::baseGameModPathRegex());

  std::vector<Path> vecModsDirs; // [TODO] = GetAddtlModsSearchDirs();
  if (baseGameModPath != "") {
    vecModsDirs.push_back(baseGameModPath + "/rtx-remix/mods/");
  }
  vecModsDirs.push_back(defaultModsDir);
  m_modsDirs.clear();
  for (const auto& modsDirPath : vecModsDirs) {
    if (!fs::exists(modsDirPath)) {
      static std::unordered_set<std::string> warnedOnce;
      if(warnedOnce.count(modsDirPath.string()) == 0) {
        Logger::warn(str::format("Cannot find ",modsDirPath.c_str()," under current directory: ", fs::current_path().c_str()));
        warnedOnce.insert(modsDirPath.string());
      }
      continue;
    }
    m_modsDirs.push_back(modsDirPath);
    enumerateModsInDir(modsDirPath, mods);
  }
  return mods;
}

struct ModDesc {
  Mod::Path filePath;
  const ModTypeInfo* typeInfo;
};

static std::tuple<bool, ModDesc> modExists(const Mod::Path& modDirPath) {
  for (auto const& modDirEntry : fs::directory_iterator { modDirPath }) {
    const auto potentialModPath = modDirEntry.path();
    if (potentialModPath.stem().string().compare(kDefaultModFileName) == 0) {
      for (auto modTypeInfo : kModTypeInfos) {
        if (modTypeInfo->isValidMod(potentialModPath)) {
          return { true, { potentialModPath, modTypeInfo } };
        }
      }
    }
  }
  return { false, {} };
}

void ModManager::enumerateModsInDir(const Path& modsDirPath, std::vector<std::unique_ptr<Mod>>& outMods) {
  for (auto const& modsDirEntry : fs::directory_iterator{modsDirPath}) {
    if (modsDirEntry.is_directory()) {
      const auto modDirPath = modsDirEntry.path();
      const auto [bModExists, modDesc] = modExists(modDirPath);
      if (!bModExists) {
        continue;
      }
      // Both mods directories can name the same folder.
      const Mod::Path filePath = modDesc.filePath;
      const bool known = std::any_of(outMods.begin(), outMods.end(),
        [&filePath](const std::unique_ptr<Mod>& mod) { return mod->m_filePath == filePath; });
      if (!known) {
        outMods.push_back(modDesc.typeInfo->construct(filePath));
      }
    }
  }
}

Mod::Mod(const Path& filePath)
  : m_filePath(filePath),
    m_name(filePath.parent_path().filename().string()),
    m_priority(0) {
  m_replacements = std::make_unique<AssetReplacements>();
}

Mod::~Mod() {
}

}
