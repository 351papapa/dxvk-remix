#include "rtx_numos_weather.h"
#include "rtx_precipitation.h"
#include "rtx_global_volumetrics.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "numos/settings.h"
#include "numos/weather.h"

#include "rtx_numos_settings.h"
#include "../../util/util_env.h"
#include "../../util/log/log.h"

namespace dxvk {

  namespace {
    namespace fs = std::filesystem;

    constexpr const char* kSeedTarget = "(seed)";
    constexpr const char* kLegacyMarker = ".legacy-presets-imported";

    // Defaults of the in-tree atmosphere the older preset files were tuned against. Absolute values of
    // these keys keep their ratio to the default when converted.
    constexpr float kLegacyCloudDensity = 6.7f;
    constexpr float kLegacyProfileDepthKm = 0.7f;

    size_t fieldSize(numos::SettingType type) {
      switch (type) {
      case numos::SettingType::Bool:  return sizeof(bool);
      case numos::SettingType::Int:   return sizeof(int32_t);
      case numos::SettingType::Uint:  return sizeof(uint32_t);
      case numos::SettingType::Float: return sizeof(float);
      case numos::SettingType::Vec3:  return sizeof(numos::Vector3);
      }
      return 0;
    }

    char* fieldPtr(numos::Settings& s, const numos::SettingInfo& info) {
      return reinterpret_cast<char*>(&s) + info.offset;
    }

    const char* fieldPtr(const numos::Settings& s, const numos::SettingInfo& info) {
      return reinterpret_cast<const char*>(&s) + info.offset;
    }

    float& floatField(numos::Settings& s, const numos::SettingInfo& info) {
      return *reinterpret_cast<float*>(fieldPtr(s, info));
    }

    numos::Vector3& vec3Field(numos::Settings& s, const numos::SettingInfo& info) {
      return *reinterpret_cast<numos::Vector3*>(fieldPtr(s, info));
    }

    bool sameField(const numos::Settings& a, const numos::Settings& b, const numos::SettingInfo& info) {
      return std::memcmp(fieldPtr(a, info), fieldPtr(b, info), fieldSize(info.type)) == 0;
    }

    void copyField(numos::Settings& dst, const numos::Settings& src, const numos::SettingInfo& info) {
      std::memcpy(fieldPtr(dst, info), fieldPtr(src, info), fieldSize(info.type));
    }

    // dst's value times the multiplier in `factors`. Only floats and colours carry multipliers.
    void multiplyField(numos::Settings& dst, const numos::Settings& factors, const numos::SettingInfo& info) {
      if (info.type == numos::SettingType::Float) {
        floatField(dst, info) *= *reinterpret_cast<const float*>(fieldPtr(factors, info));
      } else if (info.type == numos::SettingType::Vec3) {
        numos::Vector3& v = vec3Field(dst, info);
        const numos::Vector3& m = *reinterpret_cast<const numos::Vector3*>(fieldPtr(factors, info));
        v = numos::Vector3(v.x * m.x, v.y * m.y, v.z * m.z);
      } else {
        copyField(dst, factors, info);
      }
    }

    size_t indexOf(const numos::SettingInfo& info) {
      return size_t(&info - numos::settingInfos().data());
    }

    std::string trim(const std::string& s) {
      size_t b = 0, e = s.size();
      while (b < e && std::isspace(uint8_t(s[b]))) ++b;
      while (e > b && std::isspace(uint8_t(s[e - 1]))) --e;
      return s.substr(b, e - b);
    }

    std::string lower(std::string s) {
      std::transform(s.begin(), s.end(), s.begin(), [](char c) { return char(std::tolower(uint8_t(c))); });
      return s;
    }

    bool equalsNoCase(const std::string& a, const std::string& b) {
      return a.size() == b.size() &&
             std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) { return std::tolower(uint8_t(x)) == std::tolower(uint8_t(y)); });
    }

    bool parseBool(const std::string& text) {
      return text == "1" || equalsNoCase(text, "true");
    }

    // "1.5" or "0.1, 0.2, 0.3": exactly `count` comma-separated floats.
    bool parseFloats(const std::string& text, float* out, int count) {
      const char* p = text.c_str();
      for (int i = 0; i < count; ++i) {
        char* end = nullptr;
        const float v = std::strtof(p, &end);
        if (end == p || !std::isfinite(v)) {
          return false;
        }
        out[i] = v;
        p = end;
        while (*p == ' ' || *p == '\t') ++p;
        if (i + 1 < count) {
          if (*p != ',') {
            return false;
          }
          ++p;
        }
      }
      while (std::isspace(uint8_t(*p))) ++p;
      return *p == '\0';
    }

    // A colour as three floats, or one float for all three channels.
    bool parseVec3(const std::string& text, numos::Vector3& out) {
      float v[3];
      if (parseFloats(text, v, 3)) {
        out = numos::Vector3(v[0], v[1], v[2]);
        return true;
      }
      if (parseFloats(text, v, 1)) {
        out = numos::Vector3(v[0], v[0], v[0]);
        return true;
      }
      return false;
    }

    std::string fileNameFor(const std::string& name) {
      std::string out = trim(name);
      for (char& c : out) {
        if (std::strchr("\\/:*?\"<>|", c) != nullptr) {
          c = '_';
        }
      }
      return out.empty() ? std::string("Preset") : out;
    }

    // The defaults with every weather field moved off its default value.
    numos::Settings perturbedWeather() {
      const numos::Settings defaults {};
      numos::WeatherSnapshot w = numos::weatherFromSettings(defaults);
      size_t count = 0;
      const numos::WeatherFieldInfo* fields = numos::weatherFields(count);
      for (size_t i = 0; i < count; ++i) {
        float& v = w.*(fields[i].member);
        v = fields[i].kind == numos::WeatherFieldKind::Flag ? (v != 0.0f ? 0.0f : 1.0f) : v * 1.37f + 0.731f;
      }
      numos::Settings s = defaults;
      numos::applyWeather(w, s);
      return s;
    }

    // Which settings are weather fields (moved by perturbedWeather).
    const std::vector<bool>& weatherKeys() {
      static const std::vector<bool> keys = [] {
        const numos::Settings defaults {};
        const numos::Settings moved = perturbedWeather();
        const auto& infos = numos::settingInfos();
        std::vector<bool> k(infos.size());
        for (size_t i = 0; i < infos.size(); ++i) {
          k[i] = !sameField(defaults, moved, infos[i]);
        }
        return k;
      }();
      return keys;
    }

    // NUMOS's preset as preset keys: every field it sets to a fixed value, and every field it leaves
    // different from the defaults (some scale the base, e.g. a storm's density). Built over the defaults.
    void numosPresetKeys(numos::WeatherPreset which, numos::Settings& values, std::vector<bool>& set) {
      const numos::Settings defaults {};
      const numos::Settings moved = perturbedWeather();
      numos::Settings overDefaults = defaults;
      numos::applyWeather(numos::weatherPreset(which, defaults), overDefaults);
      numos::Settings overMoved = defaults;
      numos::applyWeather(numos::weatherPreset(which, moved), overMoved);
      const auto& infos = numos::settingInfos();
      set.assign(infos.size(), false);
      for (size_t i = 0; i < infos.size(); ++i) {
        const bool changed = !sameField(overDefaults, defaults, infos[i]);
        const bool fixed = weatherKeys()[i] && sameField(overDefaults, overMoved, infos[i]);
        set[i] = changed || fixed;
      }
      values = overDefaults;
    }

    float smoothstep01(float t) {
      t = std::clamp(t, 0.0f, 1.0f);
      return t * t * (3.0f - 2.0f * t);
    }

    // Stores a "x<factors>" value. Returns false for a setting that cannot be multiplied or a bad value.
    bool setMultiplier(NumosWeather::Preset& p, const numos::SettingInfo& info, const std::string& text) {
      const size_t i = indexOf(info);
      if (info.type == numos::SettingType::Float) {
        float v;
        if (!parseFloats(trim(text), &v, 1)) {
          return false;
        }
        floatField(*p.values, info) = v;
      } else if (info.type == numos::SettingType::Vec3) {
        numos::Vector3 v;
        if (!parseVec3(trim(text), v)) {
          return false;
        }
        vec3Field(*p.values, info) = v;
      } else {
        return false;
      }
      p.set[i] = true;
      p.relative[i] = true;
      return true;
    }

    // Reads one preset file. The preset is named after the file.
    // Returns false when the file cannot be read.
    bool parseWetnessPreset(WetnessPreset& preset, const std::string& key, const std::string& text) {
      for (size_t i = 0; i < std::size(kWetnessParameters); ++i) {
        const std::string name = kWetnessParameters[i].name;
        std::string legacy = "wetness" + name;
        legacy[7] = char(std::toupper(static_cast<unsigned char>(legacy[7])));
        if (key != "rtx.wetness." + name && key != legacy
            && !(name == "rainIntensity" && key == "precipitationIntensity")) {
          continue;
        }
        const bool relative = !text.empty() && (text[0] == 'x' || text[0] == 'X' || text[0] == '*');
        const std::string body = relative ? text.substr(1) : text;
        try {
          size_t end = 0;
          const float value = std::stof(body, &end);
          if (end == body.size() && std::isfinite(value) && value >= 0.0f && value <= 100.0f) {
            preset.values[i] = value;
            preset.relative[i] = relative;
          } else {
            Logger::warn("[NUMOS weather] Invalid wetness value for " + key);
          }
        } catch (...) {
          Logger::warn("[NUMOS weather] Invalid wetness value for " + key);
        }
        return true;
      }
      return false;
    }

    bool parsePrecipitationPreset(PrecipitationPreset& preset, const std::string& key, const std::string& text) {
      const bool relative = !text.empty() && (text[0] == 'x' || text[0] == 'X' || text[0] == '*');
      const std::string body = relative ? text.substr(1) : text;
      if (key == "rtx.weather.precipitation.color" || key == "precipitationColor") {
        std::string numbers = body;
        std::replace(numbers.begin(), numbers.end(), ',', ' ');
        std::istringstream input(numbers);
        Vector3 color;
        std::string trailing;
        if (input >> color.x >> color.y >> color.z && !(input >> trailing)
            && std::isfinite(color.x) && std::isfinite(color.y) && std::isfinite(color.z)) {
          preset.color = color;
          preset.colorRelative = relative;
        } else {
          Logger::warn("[NUMOS weather] Invalid precipitation color");
        }
        return true;
      }
      for (size_t i = 0; i < std::size(kPrecipitationParameters); ++i) {
        const std::string name = kPrecipitationParameters[i].name;
        std::string legacy = "precipitation" + name;
        legacy[13] = char(std::toupper(static_cast<unsigned char>(legacy[13])));
        if (key != "rtx.weather.precipitation." + name && key != legacy) { continue; }
        try {
          size_t end = 0;
          const float value = std::stof(body, &end);
          if (end == body.size() && std::isfinite(value) && value >= 0.0f) {
            preset.values[i] = value;
            preset.relative[i] = relative;
          } else {
            Logger::warn("[NUMOS weather] Invalid precipitation value for " + key);
          }
        } catch (...) {
          Logger::warn("[NUMOS weather] Invalid precipitation value for " + key);
        }
        return true;
      }
      return false;
    }

    bool parseVolumetricPreset(VolumetricWeatherPreset& preset, const std::string& key, const std::string& text) {
      std::string name = key;
      const std::string prefix = "rtx.volumetrics.";
      if (name.compare(0, prefix.size(), prefix) == 0) { name.erase(0, prefix.size()); }
      if (name == "volumetricAnisotropy") { name = "anisotropy"; }
      const bool relative = !text.empty() && (text[0] == 'x' || text[0] == 'X' || text[0] == '*');
      const std::string body = relative ? text.substr(1) : text;
      for (size_t i = 0; i < std::size(kVolumetricWeatherParameters); ++i) {
        if (name != kVolumetricWeatherParameters[i].name) { continue; }
        try {
          size_t end = 0;
          const float value = std::stof(body, &end);
          if (end != body.size() || !std::isfinite(value)) { throw std::invalid_argument("value"); }
          preset.values[i] = value;
          preset.relative[i] = relative;
        } catch (...) { Logger::warn("[NUMOS weather] Invalid volumetric value for " + key); }
        return true;
      }
      for (size_t i = 0; i < std::size(kVolumetricWeatherColors); ++i) {
        if (name != kVolumetricWeatherColors[i].name) { continue; }
        std::string numbers = body;
        std::replace(numbers.begin(), numbers.end(), ',', ' ');
        std::istringstream input(numbers);
        Vector3 color;
        std::string trailing;
        if (input >> color.x >> color.y >> color.z && !(input >> trailing)
            && std::isfinite(color.x) && std::isfinite(color.y) && std::isfinite(color.z)) {
          preset.colors[i] = color;
          preset.colorRelative[i] = relative;
        } else { Logger::warn("[NUMOS weather] Invalid volumetric color for " + key); }
        return true;
      }
      for (size_t i = 0; i < std::size(kVolumetricWeatherSwitches); ++i) {
        if (name != kVolumetricWeatherSwitches[i].name) { continue; }
        std::string value = text;
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (value == "true" || value == "1") { preset.switches[i] = true; }
        else if (value == "false" || value == "0") { preset.switches[i] = false; }
        else { Logger::warn("[NUMOS weather] Invalid volumetric switch for " + key); }
        return true;
      }
      return false;
    }

    bool readPresetFile(const fs::path& path, NumosWeather::Preset& p) {
      std::ifstream f(path, std::ios::binary);
      if (!f) {
        return false;
      }
      std::stringstream text;
      text << f.rdbuf();
      p.name = path.stem().string();
      p.water = WaterWeatherState::forName(p.name);
      numos::WeatherPreset which;
      p.numosBuiltIn = numos::weatherPresetFromName(p.name.c_str(), which);

      // The preset's own keys and the multipliers are read here. Every other line goes to NUMOS's parser.
      std::string numosText;
      std::istringstream lines(text.str());
      std::string line;
      while (std::getline(lines, line)) {
        std::string code = line;
        const size_t hash = code.find('#');
        if (hash != std::string::npos) {
          code.resize(hash);
        }
        const size_t eq = code.find('=');
        if (eq != std::string::npos) {
          const std::string key = trim(code.substr(0, eq));
          const std::string value = trim(code.substr(eq + 1));
          if (key == "weather.oceanRainWeight" || key == "weather.oceanStormWeight") {
            try {
              const float weight = std::stof(value);
              if (key == "weather.oceanRainWeight") { p.water.rain = weight; }
              else { p.water.storm = weight; }
            } catch (...) { Logger::warn("[NUMOS weather] Invalid ocean wave weight in " + path.string()); }
            continue;
          }
          const bool precipitationKey = parsePrecipitationPreset(p.precipitation, key, value);
          if (parseWetnessPreset(p.wetness, key, value) || precipitationKey || parseVolumetricPreset(p.volumetric, key, value)) {
            continue;
          }
          if (key == "weather.bindings") {
            std::istringstream ids(value);
            std::string id;
            while (std::getline(ids, id, ',')) {
              id = trim(id);
              if (!id.empty()) {
                p.bindings.push_back(id);
              }
            }
            continue;
          }
          if (key == "weather.transitionSeconds") {
            try {
              p.transitionSeconds = std::stof(value);
            } catch (...) {
              Logger::warn("[NUMOS weather] " + path.string() + ": weather.transitionSeconds is not a number");
            }
            continue;
          }
          if (key == "weather.enabled") {
            p.enabled = parseBool(value);
            continue;
          }
          if (!value.empty() && (value[0] == 'x' || value[0] == 'X')) {
            const numos::SettingInfo* info = numos::findSetting(key);
            if (info == nullptr || !setMultiplier(p, *info, value.substr(1))) {
              Logger::warn("[NUMOS weather] " + path.string() + ": " + key + " = " + value +
                           " is not a multiplier on a number or a colour; ignored");
            }
            continue;
          }
        }
        numosText += line;
        numosText += '\n';
      }
      numos::ExplicitSettings explicitlySet;
      numos::applyConfig(*p.values, numosText, &explicitlySet);
      for (size_t i = 0; i < p.set.size() && i < explicitlySet.size(); ++i) {
        if (explicitlySet[i]) {
          p.set[i] = true;
          p.relative[i] = false;
        }
      }
      return true;
    }

    // ------------------------------------------------------------------------------------------
    // Older single-file presets: [name] sections of the in-tree atmosphere's keys.
    // ------------------------------------------------------------------------------------------

    struct LegacySection {
      std::string name;
      std::vector<std::pair<std::string, std::string>> keys;
    };

    std::vector<LegacySection> parseLegacyFile(std::istream& in) {
      std::vector<LegacySection> sections;
      std::string line;
      while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#') {
          continue;
        }
        if (line.front() == '[' && line.back() == ']') {
          sections.emplace_back();
          sections.back().name = trim(line.substr(1, line.size() - 2));
          continue;
        }
        const size_t eq = line.find('=');
        if (eq != std::string::npos && !sections.empty()) {
          sections.back().keys.emplace_back(trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
        }
      }
      return sections;
    }

    struct LegacyValue {
      bool relative = false;
      numos::Vector3 v;
    };

    // An old value: absolute, or "x<value>" (also "*<value>") as a multiplier. Colours take three values.
    bool parseLegacyValue(const std::string& text, bool colour, LegacyValue& out) {
      std::string body = text;
      out.relative = !body.empty() && (body[0] == 'x' || body[0] == 'X' || body[0] == '*');
      if (out.relative) {
        body = trim(body.substr(1));
      }
      if (colour) {
        return parseVec3(body, out.v);
      }
      float f;
      if (!parseFloats(body, &f, 1)) {
        return false;
      }
      out.v = numos::Vector3(f, f, f);
      return true;
    }

    void putFloat(NumosWeather::Preset& p, const char* key, float value, bool relative) {
      if (const numos::SettingInfo* info = numos::findSetting(key)) {
        floatField(*p.values, *info) = value;
        p.set[indexOf(*info)] = true;
        p.relative[indexOf(*info)] = relative;
      }
    }

    void putVec3(NumosWeather::Preset& p, const char* key, const numos::Vector3& value, bool relative) {
      if (const numos::SettingInfo* info = numos::findSetting(key)) {
        vec3Field(*p.values, *info) = value;
        p.set[indexOf(*info)] = true;
        p.relative[indexOf(*info)] = relative;
      }
    }

    void putBool(NumosWeather::Preset& p, const char* key, bool value) {
      if (const numos::SettingInfo* info = numos::findSetting(key)) {
        *reinterpret_cast<bool*>(fieldPtr(*p.values, *info)) = value;
        p.set[indexOf(*info)] = true;
        p.relative[indexOf(*info)] = false;
      }
    }

    numos::Vector3 divide(const numos::Vector3& a, const numos::Vector3& b) {
      return numos::Vector3(a.x / std::max(b.x, 1e-8f), a.y / std::max(b.y, 1e-8f), a.z / std::max(b.z, 1e-8f));
    }

    numos::Vector3 multiply(const numos::Vector3& a, const numos::Vector3& b) {
      return numos::Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    // Converts one section to a preset over the user's settings `base` (for colours given as absolute
    // scattering coefficients). `skipped` collects the keys that have no NUMOS equivalent.
    NumosWeather::Preset convertLegacy(const LegacySection& section, const numos::Settings& base, std::string& skipped) {
      static const struct { const char* oldKey; const char* key; } kOneToOne[] = {
        { "coverage",                 "rtx.atmosphere.cloudCoverageMean" },
        { "cloudDepthMeters",         "rtx.atmosphere.cloudDepthMeters" },
        { "layer2PlayerOffsetMeters", "rtx.atmosphere.cloudLayer2CameraOffsetMeters" },
        { "layer2BaseMeters",         "rtx.atmosphere.cloudLayer2BaseHeightMeters" },
        { "layer2DepthMeters",        "rtx.atmosphere.cloudLayer2DepthMeters" },
        { "layer2Density",            "rtx.atmosphere.cloudLayer2DensityScale" },
        { "layer2Coverage",           "rtx.atmosphere.cloudLayer2CoverageMean" },
        { "airDensity",               "rtx.atmosphere.airDensity" },
        { "hazeDensity",              "rtx.atmosphere.aerosolDensity" },
        { "lightningPerMinute",       "rtx.atmosphere.lightningStrikesPerMinute" },
      };
      static const struct { const char* oldKey; const char* key; } kColours[] = {
        { "cloudColor",     "rtx.atmosphere.cloudColor" },
        { "sunIlluminance", "rtx.atmosphere.sunIlluminance" },
        { "lightningColor", "rtx.atmosphere.lightningColor" },
      };
      const numos::Settings defaults {};
      NumosWeather::Preset p;
      p.name = fileNameFor(section.name);
      p.water = WaterWeatherState::forName(section.name);
      bool hasCloudColor = false;
      LegacyValue cloudColor;
      bool hasLayer2Color = false;
      LegacyValue layer2Color;

      for (const auto& [key, text] : section.keys) {
        const bool precipitationKey = parsePrecipitationPreset(p.precipitation, key, text);
        if (parseWetnessPreset(p.wetness, key, text) || precipitationKey || parseVolumetricPreset(p.volumetric, key, text)) {
          continue;
        }
        LegacyValue v;
        bool handled = true;
        if (key == "enabled") {
          p.enabled = parseBool(text);
        } else if (key == "weather") {
          // "<hex id> <name>": both are ids the game can report.
          const size_t space = text.find_first_of(" \t");
          const uint32_t id = uint32_t(std::strtoul(text.c_str(), nullptr, 16));
          if (id != 0) {
            char hex[16];
            std::snprintf(hex, sizeof(hex), "%08X", id);
            p.bindings.push_back(hex);
          }
          if (space != std::string::npos && !trim(text.substr(space)).empty()) {
            p.bindings.push_back(trim(text.substr(space)));
          }
        } else if (key == "layer2Enable" || key == "layer2FollowPlayer") {
          putBool(p, key == "layer2Enable" ? "rtx.atmosphere.cloudLayer2Enable" : "rtx.atmosphere.cloudLayer2FollowCamera",
                  parseBool(text));
        } else if (key == "density" && parseLegacyValue(text, false, v)) {
          putFloat(p, "rtx.atmosphere.cloudDensity", v.relative ? v.v.x : v.v.x * defaults.cloudDensity / kLegacyCloudDensity,
                   v.relative);
        } else if (key == "profileDepthKm" && parseLegacyValue(text, false, v)) {
          putFloat(p, "rtx.atmosphere.nvdfProfileDepthKm",
                   v.relative ? v.v.x : v.v.x * defaults.nvdfProfileDepthKm / kLegacyProfileDepthKm, v.relative);
        } else if (key == "airScattering" && parseLegacyValue(text, true, v)) {
          // An absolute Rayleigh coefficient becomes the tint that gives it over the user's coefficient.
          putVec3(p, "rtx.atmosphere.airTint", v.relative ? v.v : divide(v.v, base.rayleighScattering), v.relative);
        } else if (key == "hazeScattering" && parseLegacyValue(text, true, v)) {
          putVec3(p, "rtx.atmosphere.aerosolTint", v.relative ? v.v : divide(v.v, base.mieScattering), v.relative);
        } else if (key == "cloudColor" && parseLegacyValue(text, true, v)) {
          putVec3(p, "rtx.atmosphere.cloudColor", v.v, v.relative);
          hasCloudColor = true;
          cloudColor = v;
        } else if (key == "aerialCompression" && parseLegacyValue(text, false, v) && v.v.x > 0.0f) {
          // Haze that reads c times as far is the user's haze world scale divided by c.
          putFloat(p, "rtx.atmosphere.hazeWorldScale", 1.0f / v.v.x, true);
        } else if (key == "layer2Color" && parseLegacyValue(text, true, v)) {
          hasLayer2Color = true;
          layer2Color = v;
        } else {
          handled = false;
          for (const auto& m : kOneToOne) {
            if (key == m.oldKey && parseLegacyValue(text, false, v)) {
              putFloat(p, m.key, v.v.x, v.relative);
              handled = true;
            }
          }
          for (const auto& m : kColours) {
            if (key == m.oldKey && parseLegacyValue(text, true, v)) {
              putVec3(p, m.key, v.v, v.relative);
              handled = true;
            }
          }
        }
        if (!handled) {
          skipped += (skipped.empty() ? "" : ", ") + key;
        }
      }

      // Layer 2's colour becomes its albedo as NUMOS migrates the deprecated cloudLayer2Color: its ratio
      // to the cloud colour, at most 1.
      if (hasLayer2Color) {
        const numos::Vector3 cloud = !hasCloudColor ? base.cloudColor
                                   : cloudColor.relative ? multiply(base.cloudColor, cloudColor.v) : cloudColor.v;
        const numos::Vector3 tint = layer2Color.relative ? multiply(base.cloudLayer2Color, layer2Color.v) : layer2Color.v;
        const numos::Vector3 ratio = divide(tint, numos::Vector3(std::max(cloud.x, 1e-3f), std::max(cloud.y, 1e-3f),
                                                                 std::max(cloud.z, 1e-3f)));
        putVec3(p, "rtx.atmosphere.cloudLayer2Albedo", numos::Vector3(std::clamp(ratio.x, 0.0f, 1.0f),
                std::clamp(ratio.y, 0.0f, 1.0f), std::clamp(ratio.z, 0.0f, 1.0f)), false);
      }
      return p;
    }
  }

  NumosWeather::Preset::Preset()
    : values(std::make_unique<numos::Settings>()),
      set(numos::settingInfos().size(), false),
      relative(numos::settingInfos().size(), false) { }
  NumosWeather::Preset::Preset(const Preset& o)
    : name(o.name), bindings(o.bindings), transitionSeconds(o.transitionSeconds), enabled(o.enabled),
      values(std::make_unique<numos::Settings>(*o.values)), set(o.set), relative(o.relative), wetness(o.wetness), precipitation(o.precipitation), volumetric(o.volumetric), water(o.water), dirty(o.dirty),
      numosBuiltIn(o.numosBuiltIn) { }
  NumosWeather::Preset& NumosWeather::Preset::operator=(const Preset& o) {
    if (this != &o) {
      name = o.name;
      bindings = o.bindings;
      transitionSeconds = o.transitionSeconds;
      enabled = o.enabled;
      values = std::make_unique<numos::Settings>(*o.values);
      set = o.set;
      relative = o.relative;
      wetness = o.wetness;
      precipitation = o.precipitation;
      volumetric = o.volumetric;
      water = o.water;
      dirty = o.dirty;
      numosBuiltIn = o.numosBuiltIn;
    }
    return *this;
  }
  NumosWeather::Preset::~Preset() = default;

  struct NumosWeather::State {
    numos::Settings base, from, current;
    numos::WeatherSnapshot snapshot;
    WetnessArgs wetnessFrom, wetnessCurrent;
    PrecipitationState precipitationFrom, precipitationCurrent;
    VolumetricWeatherState volumetricFrom, volumetricCurrent;
    WaterWeatherState waterFrom, waterCurrent;
    float elapsed = 0.0f;
    float duration = 0.0f;
    bool hasCurrent = false;
    bool held = false;
    bool snap = false;  // the next change of target shows at once
    Preset heldPreset;
    std::string lastGameWeather, lastManualPreset, lastLegacyFile;
    numos::Settings tuneBaseline, tuneStart;  // the user's settings, and the look written over them
  };

  NumosWeather::NumosWeather(DxvkDevice*) : m_state(std::make_unique<State>()) { }
  NumosWeather::~NumosWeather() = default;

  std::string NumosWeather::directory() const {
    fs::path p(presetDirectory());
    if (p.is_relative()) {
      p = fs::path(env::getExePath()).parent_path() / p;
    }
    return p.lexically_normal().string();
  }

  void NumosWeather::ensureLoaded() {
    if (m_loaded) {
      return;
    }
    m_loaded = true;
    writeMissingNumosPresets();
    reload();
  }

  void NumosWeather::writeMissingNumosPresets() {
    std::error_code ec;
    fs::create_directories(directory(), ec);
    // Each NUMOS preset is written once per folder: the marker lists the ones already written, so a
    // preset the user deleted or renamed stays gone.
    const fs::path marker = fs::path(directory()) / ".numos-presets-written";
    std::vector<std::string> written;
    {
      std::ifstream in(marker);
      std::string line;
      while (std::getline(in, line)) {
        line = trim(line);
        if (!line.empty()) {
          written.push_back(line);
        }
      }
    }
    bool markerChanged = false;
    for (uint32_t i = 0; i < uint32_t(numos::WeatherPreset::Count); ++i) {
      const numos::WeatherPreset which = numos::WeatherPreset(i);
      Preset p;
      p.name = numos::weatherPresetName(which);
      p.water = WaterWeatherState::forName(p.name);
      if (std::find(written.begin(), written.end(), p.name) != written.end()) {
        continue;
      }
      written.push_back(p.name);
      markerChanged = true;
      if (fs::exists(fs::path(directory()) / (p.name + ".conf"), ec)) {
        continue;  // shipped with the game's presets, or the user's
      }
      numosPresetKeys(which, *p.values, p.set);
      p.numosBuiltIn = true;
      std::string error;
      if (!save(p, &error)) {
        Logger::warn("[NUMOS weather] " + error);
      }
    }
    if (markerChanged) {
      std::ofstream out(marker, std::ios::trunc);
      for (const std::string& name : written) {
        out << name << '\n';
      }
    }
  }

  void NumosWeather::reload() {
    m_presets.clear();
    std::error_code ec;
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory(), ec)) {
      if (entry.is_regular_file(ec) && entry.path().extension() == ".conf") {
        files.push_back(entry.path());
      }
    }
    std::sort(files.begin(), files.end());
    for (const fs::path& path : files) {
      Preset p;
      if (readPresetFile(path, p)) {
        m_presets.push_back(std::move(p));
      }
    }
    m_saved = m_presets;
    Logger::info(str::format("[NUMOS weather] ", m_presets.size(), " presets in ", directory()));
  }

  bool NumosWeather::save(Preset& p, std::string* error) {
    std::ostringstream text;
    text << "# NUMOS weather preset \"" << p.name << "\". Edit it here or in the Remix dev menu (Sky > Weather > Weather Presets).\n"
         << "# While it shows, exactly the settings listed below override the user's; any rtx.atmosphere.* key may appear.\n"
         << "# A value written x<value> multiplies the user's own setting (per channel for colours).\n"
         << "# weather.bindings: the game weather ids it stands for (rtx.weather.gameWeather), comma separated.\n"
         << "weather.bindings = ";
    for (size_t i = 0; i < p.bindings.size(); ++i) {
      text << (i ? ", " : "") << p.bindings[i];
    }
    text << "\n";
    if (p.transitionSeconds >= 0.0f) {
      text << "weather.transitionSeconds = " << p.transitionSeconds << "\n";
    }
    if (!p.enabled) {
      text << "weather.enabled = false\n";
    }
    text << "\n";
    const auto& infos = numos::settingInfos();
    for (size_t i = 0; i < infos.size(); ++i) {
      if (p.set[i]) {
        text << infos[i].key << " = " << (p.relative[i] ? "x" : "") << numos::formatSetting(*p.values, infos[i]) << "\n";
      }
    }
    text << "weather.oceanRainWeight = " << p.water.rain << "\n"
         << "weather.oceanStormWeight = " << p.water.storm << "\n";
    for (size_t i = 0; i < p.precipitation.values.size(); ++i) {
      if (p.precipitation.values[i]) {
        text << "rtx.weather.precipitation." << kPrecipitationParameters[i].name << " = "
             << (p.precipitation.relative[i] ? "x" : "") << *p.precipitation.values[i] << "\n";
      }
    }
    if (p.precipitation.color) {
      const auto color = *p.precipitation.color;
      text << "rtx.weather.precipitation.color = " << (p.precipitation.colorRelative ? "x" : "")
           << color.x << ", " << color.y << ", " << color.z << "\n";
    }
    for (size_t i = 0; i < p.volumetric.values.size(); ++i) {
      if (p.volumetric.values[i]) {
        text << "rtx.volumetrics." << kVolumetricWeatherParameters[i].name << " = "
             << (p.volumetric.relative[i] ? "x" : "") << *p.volumetric.values[i] << "\n";
      }
    }
    for (size_t i = 0; i < p.volumetric.colors.size(); ++i) {
      if (p.volumetric.colors[i]) {
        const auto color = *p.volumetric.colors[i];
        text << "rtx.volumetrics." << kVolumetricWeatherColors[i].name << " = "
             << (p.volumetric.colorRelative[i] ? "x" : "") << color.x << ", " << color.y << ", " << color.z << "\n";
      }
    }
    for (size_t i = 0; i < p.volumetric.switches.size(); ++i) {
      if (p.volumetric.switches[i]) {
        text << "rtx.volumetrics." << kVolumetricWeatherSwitches[i].name << " = "
             << (*p.volumetric.switches[i] ? "true" : "false") << "\n";
      }
    }
    for (size_t i = 0; i < p.wetness.values.size(); ++i) {
      if (p.wetness.values[i].has_value()) {
        text << "rtx.wetness." << kWetnessParameters[i].name << " = "
             << (p.wetness.relative[i] ? "x" : "") << *p.wetness.values[i] << "\n";
      }
    }
    std::error_code ec;
    fs::create_directories(directory(), ec);
    const fs::path path = fs::path(directory()) / (p.name + ".conf");
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
      if (error) *error = "cannot write " + path.string();
      return false;
    }
    f << text.str();
    p.dirty = false;
    rememberSaved(p);
    return true;
  }

  void NumosWeather::rememberSaved(const Preset& p) {
    for (Preset& s : m_saved) {
      if (s.name == p.name) {
        s = p;
        return;
      }
    }
    m_saved.push_back(p);
  }

  NumosWeather::Preset& NumosWeather::create(const std::string& name, const Preset* from) {
    Preset p;
    if (from != nullptr) {
      p = *from;
    } else {
      *p.values = m_state->current;
      p.water = m_state->waterCurrent;
      p.precipitation.store(m_state->precipitationCurrent);
      p.volumetric.store(m_state->volumetricCurrent);
      p.set = weatherKeys();
    }
    p.name = name;
    p.bindings.clear();
    numos::WeatherPreset which;
    p.numosBuiltIn = numos::weatherPresetFromName(name.c_str(), which);
    p.dirty = true;
    m_presets.push_back(std::move(p));
    return m_presets.back();
  }

  NumosWeather::Preset& NumosWeather::createFromLook(const std::string& name, const numos::Settings& look,
                                                     const std::vector<bool>& keys) {
    const Preset empty;
    Preset& p = create(name, &empty);
    const auto& infos = numos::settingInfos();
    *p.values = look;
    p.precipitation.store(m_state->precipitationCurrent);
    p.volumetric.store(m_state->volumetricCurrent);
    for (size_t i = 0; i < infos.size(); ++i) {
      p.set[i] = weatherKeys()[i] || (i < keys.size() && keys[i]);
    }
    return p;
  }

  NumosWeather::Preset* NumosWeather::loadFile(const std::string& path, std::string* error) {
    ensureLoaded();
    fs::path file(path);
    if (file.is_relative()) {
      file = fs::path(env::getExePath()).parent_path() / file;
    }
    Preset p;
    std::error_code ec;
    if (!fs::is_regular_file(file, ec) || !readPresetFile(file, p)) {
      if (error) *error = "cannot read " + file.string();
      return nullptr;
    }

    // The file is copied into the preset folder under a name no preset has, so loading never replaces one.
    p.name = uniqueName(p.name);
    numos::WeatherPreset which;
    p.numosBuiltIn = numos::weatherPresetFromName(p.name.c_str(), which);
    m_presets.push_back(std::move(p));
    Preset& loaded = m_presets.back();
    if (!save(loaded, error)) {
      m_presets.pop_back();
      return nullptr;
    }
    return &loaded;
  }

  bool NumosWeather::rename(Preset& p, const std::string& newName, std::string* error) {
    if (newName.empty() || newName.find_first_of("\\/:*?\"<>|") != std::string::npos) {
      if (error) *error = "not a valid file name";
      return false;
    }
    if (find(newName) != nullptr) {
      if (error) *error = "a preset with that name exists";
      return false;
    }
    std::error_code ec;
    const fs::path oldPath = fs::path(directory()) / (p.name + ".conf");
    if (fs::exists(oldPath, ec)) {
      fs::rename(oldPath, fs::path(directory()) / (newName + ".conf"), ec);
      if (ec) {
        if (error) *error = ec.message();
        return false;
      }
    }
    if (preset() == p.name) {
      presetObject().setDeferred(newName);
    }
    if (m_preview == p.name) {
      m_preview = newName;
    }
    for (Preset& s : m_saved) {
      if (s.name == p.name) {
        s.name = newName;
      }
    }
    p.name = newName;
    numos::WeatherPreset which;
    p.numosBuiltIn = numos::weatherPresetFromName(newName.c_str(), which);
    return true;
  }

  bool NumosWeather::remove(const std::string& name, std::string* error) {
    std::error_code ec;
    fs::remove(fs::path(directory()) / (name + ".conf"), ec);
    if (ec) {
      if (error) *error = ec.message();
      return false;
    }
    const auto named = [&](const Preset& p) { return p.name == name; };
    m_presets.erase(std::remove_if(m_presets.begin(), m_presets.end(), named), m_presets.end());
    m_saved.erase(std::remove_if(m_saved.begin(), m_saved.end(), named), m_saved.end());
    if (m_preview == name) {
      m_preview.clear();
    }
    return true;
  }

  void NumosWeather::resetToNumos(Preset& p) {
    numos::WeatherPreset which;
    if (!numos::weatherPresetFromName(p.name.c_str(), which)) {
      return;
    }
    numosPresetKeys(which, *p.values, p.set);
    p.relative.assign(p.set.size(), false);
    p.dirty = true;
  }

  bool NumosWeather::revert(Preset& p) {
    for (const Preset& s : m_saved) {
      if (s.name == p.name) {
        p = s;
        p.dirty = false;
        return true;
      }
    }
    return false;
  }

  bool NumosWeather::hasSavedCopy(const Preset& p) const {
    return std::any_of(m_saved.begin(), m_saved.end(), [&](const Preset& s) { return s.name == p.name; });
  }

  void NumosWeather::addKey(Preset& p, size_t index) {
    const auto& infos = numos::settingInfos();
    if (index < infos.size() && !p.set[index]) {
      copyField(*p.values, m_state->current, infos[index]);
      p.set[index] = true;
      p.relative[index] = false;
      p.dirty = true;
    }
  }

  void NumosWeather::removeKey(Preset& p, size_t index) {
    if (index < p.set.size() && p.set[index]) {
      p.set[index] = false;
      p.relative[index] = false;
      p.dirty = true;
    }
  }

  NumosWeather::Preset* NumosWeather::find(const std::string& name) {
    for (Preset& p : m_presets) {
      if (p.name == name) {
        return &p;
      }
    }
    return nullptr;
  }

  NumosWeather::Preset* NumosWeather::boundPreset(const std::string& gameIds) {
    std::istringstream ids(gameIds);
    std::string id;
    while (std::getline(ids, id, ',')) {
      id = trim(id);
      if (id.empty()) {
        continue;
      }
      for (Preset& p : m_presets) {
        if (!p.enabled) {
          continue;
        }
        for (const std::string& b : p.bindings) {
          if (equalsNoCase(b, id)) {
            return &p;
          }
        }
      }
    }
    return nullptr;
  }

  std::string NumosWeather::uniqueName(const std::string& base) {
    const std::string name = fileNameFor(base);
    if (find(name) == nullptr) {
      return name;
    }
    for (int n = 2;; ++n) {
      const std::string candidate = name + " " + std::to_string(n);
      if (find(candidate) == nullptr) {
        return candidate;
      }
    }
  }

  NumosWeather::Preset& NumosWeather::storeLook(Preset* preset, const std::string& weatherId) {
    if (preset == nullptr) {
      const Preset empty;
      preset = &create(uniqueName(weatherId.empty() ? std::string("Custom") : "Custom " + weatherId), &empty);
      if (!weatherId.empty()) {
        preset->bindings.push_back(weatherId);
      }
    }
    const auto& infos = numos::settingInfos();
    for (size_t i = 0; i < infos.size(); ++i) {
      if (preset->set[i] || weatherKeys()[i]) {
        copyField(*preset->values, m_state->current, infos[i]);
        preset->set[i] = true;
        preset->relative[i] = false;
      }
    }
    preset->water = m_state->waterCurrent;
      preset->precipitation.store(m_state->precipitationCurrent);
      preset->volumetric.store(m_state->volumetricCurrent);
    preset->dirty = true;
    return *preset;
  }

  void NumosWeather::beginTuning(const std::string& presetName, const std::string& weatherId) {
    if (m_tuning) {
      return;
    }
    State& s = *m_state;
    s.tuneBaseline = NumosSettings::resolved();
    s.tuneStart = s.current;
    {
      RtxOptionLayerTarget target(RtxOptionEditTarget::User);
      NumosSettings::write(s.tuneStart);
    }
    m_tuning = true;
    m_tuningPreset = presetName;
    m_tuningWeather = weatherId;
    s.snap = true;
  }

  NumosWeather::Preset* NumosWeather::finishTuning(bool keep) {
    if (!m_tuning) {
      return nullptr;
    }
    State& s = *m_state;
    Preset* stored = nullptr;
    if (keep) {
      const numos::Settings live = NumosSettings::resolved();
      stored = m_tuningPreset.empty() ? nullptr : find(m_tuningPreset);
      if (stored == nullptr) {
        const Preset empty;
        stored = &create(uniqueName(m_tuningWeather.empty() ? std::string("Custom") : "Custom " + m_tuningWeather), &empty);
        if (!m_tuningWeather.empty()) {
          stored->bindings.push_back(m_tuningWeather);
        }
      }
      // The preset's keys, every weather field, and anything else the sliders moved while tuning.
      const auto& infos = numos::settingInfos();
      for (size_t i = 0; i < infos.size(); ++i) {
        const bool moved = !sameField(live, s.tuneStart, infos[i]) && !numos::isDeprecated(infos[i]);
        if (stored->set[i] || weatherKeys()[i] || moved) {
          copyField(*stored->values, live, infos[i]);
          stored->set[i] = true;
          stored->relative[i] = false;
        }
      }
      stored->water = s.waterCurrent;
      stored->precipitation.store(s.precipitationCurrent);
      stored->volumetric.store(s.volumetricCurrent);
      stored->dirty = true;
    }
    {
      RtxOptionLayerTarget target(RtxOptionEditTarget::User);
      NumosSettings::write(s.tuneBaseline);
    }
    m_tuning = false;
    m_tuningPreset.clear();
    m_tuningWeather.clear();
    s.snap = true;
    if (stored != nullptr) {
      std::string error;
      if (!save(*stored, &error)) {
        Logger::warn("[NUMOS weather] " + error);
      }
    }
    return stored;
  }

  size_t NumosWeather::importLegacy(const std::string& path, bool again, std::string* report) {
    ensureLoaded();
    fs::path file(path);
    if (file.is_relative()) {
      file = fs::path(env::getExePath()).parent_path() / file;
    }
    std::ifstream in(file);
    if (!in) {
      const std::string message = "cannot open the older preset file " + file.string();
      Logger::warn("[NUMOS weather] " + message);
      if (report) *report = message;
      return 0;
    }
    const std::vector<LegacySection> sections = parseLegacyFile(in);

    const fs::path marker = fs::path(directory()) / kLegacyMarker;
    std::vector<std::string> imported;
    {
      std::ifstream m(marker);
      std::string line;
      while (std::getline(m, line)) {
        line = trim(line);
        if (!line.empty()) {
          imported.push_back(line);
        }
      }
    }

    size_t count = 0;
    std::string names;
    for (const LegacySection& section : sections) {
      const std::string key = lower(section.name);
      const bool done = std::find(imported.begin(), imported.end(), key) != imported.end();
      if (done && !again) {
        continue;
      }
      std::string skipped;
      Preset converted = convertLegacy(section, m_state->base, skipped);

      // A preset of the same name (in any case: the file system ignores it) takes the converted look,
      // keeps its own bindings after the section's, and is backed up first.
      Preset* target = nullptr;
      for (Preset& p : m_presets) {
        if (equalsNoCase(p.name, converted.name)) {
          target = &p;
        }
      }
      std::error_code ec;
      if (target != nullptr) {
        const fs::path conf = fs::path(directory()) / (target->name + ".conf");
        fs::copy_file(conf, fs::path(directory()) / (target->name + ".conf.bak"), fs::copy_options::overwrite_existing, ec);
        for (const std::string& b : target->bindings) {
          const bool known = std::any_of(converted.bindings.begin(), converted.bindings.end(),
                                         [&](const std::string& c) { return equalsNoCase(b, c); });
          if (!known) {
            converted.bindings.push_back(b);
          }
        }
        converted.name = target->name;
        converted.numosBuiltIn = target->numosBuiltIn;
        converted.transitionSeconds = target->transitionSeconds;
        *target = converted;
      } else {
        m_presets.push_back(converted);
        target = &m_presets.back();
      }
      std::string error;
      if (!save(*target, &error)) {
        Logger::warn("[NUMOS weather] " + error);
        continue;
      }
      ++count;
      names += (names.empty() ? "" : ", ") + target->name;
      if (!done) {
        imported.push_back(key);
      }
      Logger::info(str::format("[NUMOS weather] converted \"", section.name, "\" from ", file.string(), " into ",
                               target->name, ".conf", skipped.empty() ? std::string() : "; not converted: " + skipped));
    }

    std::ofstream out(marker, std::ios::trunc);
    for (const std::string& name : imported) {
      out << name << '\n';
    }
    if (report) {
      *report = count ? str::format("Imported ", count, " from the older preset file: ", names, ".")
                      : std::string("Nothing new to import from the older preset file.");
    }
    return count;
  }

  void NumosWeather::applyPreset(const Preset* p, numos::Settings& s) const {
    if (p == nullptr) {
      return;
    }
    const auto& infos = numos::settingInfos();
    for (size_t i = 0; i < infos.size() && i < p->set.size(); ++i) {
      if (!p->set[i]) {
        continue;
      }
      if (p->relative[i]) {
        multiplyField(s, *p->values, infos[i]);
      } else {
        copyField(s, *p->values, infos[i]);
      }
    }
  }

  void NumosWeather::blend(const numos::Settings& a, const numos::Settings& b, float t, numos::Settings& out) {
    out = a;
    for (const numos::SettingInfo& info : numos::settingInfos()) {
      switch (info.type) {
      case numos::SettingType::Float: {
        float& o = *reinterpret_cast<float*>(fieldPtr(out, info));
        const float x = *reinterpret_cast<const float*>(fieldPtr(a, info));
        const float y = *reinterpret_cast<const float*>(fieldPtr(b, info));
        o = x + (y - x) * t;
        break;
      }
      case numos::SettingType::Vec3: {
        numos::Vector3& o = *reinterpret_cast<numos::Vector3*>(fieldPtr(out, info));
        const numos::Vector3& x = *reinterpret_cast<const numos::Vector3*>(fieldPtr(a, info));
        const numos::Vector3& y = *reinterpret_cast<const numos::Vector3*>(fieldPtr(b, info));
        o = numos::Vector3(x.x + (y.x - x.x) * t, x.y + (y.y - x.y) * t, x.z + (y.z - x.z) * t);
        break;
      }
      default:
        if (t >= 0.5f) {
          copyField(out, b, info);
        }
        break;
      }
    }
    // The weather fields by NUMOS's own rules.
    numos::applyWeather(numos::blendWeather(numos::weatherFromSettings(a), numos::weatherFromSettings(b), t), out);
  }

  void NumosWeather::holdSnapshot(const numos::WeatherSnapshot& weather) {
    State& s = *m_state;
    s.heldPreset = Preset();
    s.heldPreset.name = kSeedTarget;
    s.heldPreset.precipitation.store(s.precipitationCurrent);
    s.heldPreset.volumetric.store(s.volumetricCurrent);
    numos::applyWeather(weather, *s.heldPreset.values);
    s.heldPreset.set = weatherKeys();
    s.held = true;
  }

  const numos::Settings& NumosWeather::update(const numos::Settings& base, float deltaSeconds) {
    ensureLoaded();
    State& s = *m_state;
    s.base = base;
    const auto volumetricBase = RtxGlobalVolumetrics::baseWeatherState();
    const auto volumetricFor = [&](const Preset* selected) {
      auto state = volumetricBase;
      if (controlVolumetricFog() && selected != nullptr) { selected->volumetric.apply(state); }
      return state;
    };
    const auto precipitationBase = PrecipitationSystem::baseState();
    const auto precipitationFor = [&](const Preset* selected) {
      auto state = precipitationBase;
      if (selected != nullptr) { selected->precipitation.apply(state); }
      return state;
    };
    WetnessArgs wetnessBase;
    WetnessOptions::fillShaderParams(wetnessBase);
    const auto wetnessFor = [&](const Preset* selected) {
      WetnessArgs args = wetnessBase;
      if (selected != nullptr) {
        selected->wetness.apply(args);
      }
      return args;
    };

    const std::string legacyFile = legacyPresetFile();
    if (legacyFile != s.lastLegacyFile) {
      s.lastLegacyFile = legacyFile;
      if (!legacyFile.empty()) {
        importLegacy(legacyFile, false);
      }
    }

    // While tuning, the sliders are the look: no preset, no blend.
    if (m_tuning) {
      s.wetnessCurrent = wetnessBase;
      if (!s.hasCurrent || !controlVolumetricFog()) { s.volumetricCurrent = volumetricBase; }
      // Keep ocean weather fixed while tuning the sky look.
      s.current = base;
      s.hasCurrent = true;
      s.elapsed = s.duration = 0.0f;
      m_targetName.clear();
      s.snapshot = numos::weatherFromSettings(s.current);
      return s.current;
    }

    const std::string gameId = gameWeather();
    const std::string manual = preset();
    if (gameId != s.lastGameWeather || manual != s.lastManualPreset) {
      s.held = false;  // a new weather from the game or the menu replaces a held seed
      s.lastGameWeather = gameId;
      s.lastManualPreset = manual;
    }

    Preset* previewPreset = m_preview.empty() ? nullptr : find(m_preview);
    Preset* gamePreset = followGame() ? boundPreset(gameId) : nullptr;
    const Preset* target = nullptr;
    if (previewPreset != nullptr) {
      target = previewPreset;
    } else if (s.held) {
      target = &s.heldPreset;
    } else if (gamePreset != nullptr) {
      target = gamePreset;
    } else {
      target = find(manual);
    }
    const std::string targetName = target ? target->name : std::string();
    const auto waterFor = [](const Preset* selected) {
      auto state = selected ? selected->water : WaterWeatherState{};
      state.normalize();
      return state;
    };

    // The game blends weathers itself: follow its blend factor exactly.
    const float gameBlend = gameWeatherBlend();
    if (previewPreset == nullptr && !s.held && gamePreset != nullptr && gameBlend >= 0.0f) {
      numos::Settings outgoing = base;
      applyPreset(boundPreset(gameWeatherOutgoing()), outgoing);
      numos::Settings incoming = base;
      applyPreset(gamePreset, incoming);
      blend(outgoing, incoming, std::clamp(gameBlend, 0.0f, 1.0f), s.current);
      s.wetnessCurrent = WetnessPreset::blend(wetnessFor(boundPreset(gameWeatherOutgoing())), wetnessFor(gamePreset), gameBlend, gamePreset->wetness.values[1].has_value());
      s.waterCurrent = WaterWeatherState::blend(waterFor(boundPreset(gameWeatherOutgoing())), waterFor(gamePreset), gameBlend);
      s.precipitationCurrent = PrecipitationPreset::blend(precipitationFor(boundPreset(gameWeatherOutgoing())), precipitationFor(gamePreset), gameBlend);
      s.volumetricCurrent = VolumetricWeatherPreset::blend(volumetricFor(boundPreset(gameWeatherOutgoing())), volumetricFor(gamePreset), gameBlend);
      s.hasCurrent = true;
      s.elapsed = s.duration = 0.0f;
      s.snap = false;
      m_targetName = targetName;
      s.snapshot = numos::weatherFromSettings(s.current);
      return s.current;
    }

    numos::Settings goal = base;
    applyPreset(target, goal);
    const WetnessArgs wetnessGoal = wetnessFor(target);
    const WaterWeatherState waterGoal = waterFor(target);
    const auto precipitationGoal = precipitationFor(target);
    const auto volumetricGoal = volumetricFor(target);
    if (!s.hasCurrent || targetName != m_targetName) {
      s.wetnessFrom = s.hasCurrent ? s.wetnessCurrent : wetnessGoal;
      s.waterFrom = s.hasCurrent ? s.waterCurrent : waterGoal;
      s.precipitationFrom = s.hasCurrent ? s.precipitationCurrent : precipitationGoal;
      s.volumetricFrom = s.hasCurrent ? s.volumetricCurrent : volumetricGoal;
      s.from = s.hasCurrent ? s.current : goal;
      s.duration = !s.hasCurrent || s.snap ? 0.0f
                 : (target && target->transitionSeconds >= 0.0f ? target->transitionSeconds : transitionSeconds());
      s.elapsed = 0.0f;
      s.hasCurrent = true;
      s.snap = false;
      m_targetName = targetName;
    }
    s.elapsed += std::max(deltaSeconds, 0.0f);
    const float t = s.duration > 0.0f ? s.elapsed / s.duration : 1.0f;
    s.wetnessCurrent = WetnessPreset::blend(s.wetnessFrom, wetnessGoal, smoothstep01(t), target && target->wetness.values[1].has_value());
    s.waterCurrent = WaterWeatherState::blend(s.waterFrom, waterGoal, smoothstep01(t));
    s.precipitationCurrent = PrecipitationPreset::blend(s.precipitationFrom, precipitationGoal, smoothstep01(t));
    s.volumetricCurrent = VolumetricWeatherPreset::blend(s.volumetricFrom, volumetricGoal, smoothstep01(t));
    if (t >= 1.0f) {
      s.current = goal;  // settled: live edits to the preset or the settings show at once
    } else {
      blend(s.from, goal, smoothstep01(t), s.current);
    }
    s.snapshot = numos::weatherFromSettings(s.current);
    return s.current;
  }

  VolumetricWeatherState NumosWeather::volumetricState(const VolumetricWeatherState& base) const {
    return controlVolumetricFog() && m_state->hasCurrent ? m_state->volumetricCurrent : base;
  }

  const PrecipitationState& NumosWeather::precipitationState() const {
    return m_state->precipitationCurrent;
  }

  WaterWeatherState NumosWeather::waterState() const {
    return m_state->hasCurrent ? m_state->waterCurrent : WaterWeatherState{};
  }

  void NumosWeather::fillWetnessArgs(WetnessArgs& args) const {
    if (!m_state->hasCurrent) {
      return;
    }
    for (const auto& parameter : kWetnessParameters) {
      args.*parameter.member = m_state->wetnessCurrent.*parameter.member;
    }
    args.weatherWetnessActive = m_state->wetnessCurrent.weatherWetnessActive;
  }

  const numos::WeatherSnapshot* NumosWeather::currentSnapshot() const {
    return m_state->hasCurrent ? &m_state->snapshot : nullptr;
  }

  float NumosWeather::transitionProgress() const {
    return m_state->duration > 0.0f ? std::min(m_state->elapsed / m_state->duration, 1.0f) : 1.0f;
  }

  const numos::Settings& NumosWeather::lastBase() const {
    return m_state->base;
  }

  const numos::Settings& NumosWeather::current() const {
    return m_state->current;
  }

}
