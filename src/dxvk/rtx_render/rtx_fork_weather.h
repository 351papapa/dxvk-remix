#pragma once

// rtx_fork_weather.h — fork-owned weather preset declarations.
// Defines 348 RTX_OPTIONs (12 presets x 29 fields) under the
// rtx.weather.preset.<presetName> namespace.
//
// Field bucket breakdown: 19 cloud + 3 atmosphere + 3 sky/moon mood + 4 volumetric.
//
// Usage: invoke DECLARE_ALL_WEATHER_PRESETS() inside the RtxOptions struct body
// (see rtx_options.h). The macro expands all 12 preset declarations inline.
//
// The WEATHER_PRESET_FIELD_LIST(X) X-macro is preserved here for use by Task 2
// (WeatherSnapshot struct member generation). It expands to one X(type, name,
// default) call per field, and is the single source of truth for the field
// list. DECLARE_WEATHER_PRESET delegates to it via a per-preset binder, so
// adding a field requires only one edit (in WEATHER_PRESET_FIELD_LIST) and the
// 12 preset declarations regenerate automatically.

#include "rtx_option.h"
#include "../../util/util_vector.h"

// ---------------------------------------------------------------------------
// Field-list X-macro — single source of truth for the 29 weather fields.
// Consumed by DECLARE_WEATHER_PRESET (via the per-preset binders) and also
// available for Task 2 (WeatherSnapshot struct member declaration).
// Expands to: X(type, name, defaultValue), one entry per field.
// ---------------------------------------------------------------------------
#define WEATHER_PRESET_FIELD_LIST(X)                                                               \
  /* Cloud (19) */                                                                                 \
  X(float,   cloudDensity,                              1.0f)                                      \
  X(float,   cloudCoverageMean,                         0.5f)                                      \
  X(float,   cloudCoverageSpread,                       0.2f)                                      \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                   \
  X(float,   cloudTypeMean,                             0.5f)                                      \
  X(float,   cloudTypeSpread,                           0.2f)                                      \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                   \
  X(float,   cloudAnvilBias,                            0.3f)                                      \
  X(float,   cloudWindShearStrength,                    0.5f)                                      \
  X(Vector3, cloudColor,                                Vector3(0.89f, 0.92f, 1.0f))               \
  X(float,   cloudWindSpeed,                            0.02f)                                     \
  X(float,   cloudWindDirection,                        45.0f)                                     \
  X(float,   cloudShadowStrength,                       0.0f)                                      \
  X(float,   cloudAnisotropy,                           0.6f)                                      \
  X(float,   cloudThickness,                            3.05f)                                     \
  X(float,   cloudDetailWeight,                         1.0f)                                      \
  X(Vector3, cloudShadowTint,                           Vector3(0.55f, 0.65f, 0.85f))              \
  X(float,   cloudShadowTintStrength,                   1.0f)                                      \
  X(float,   cloudSunsetWarmth,                         0.95f)                                     \
  /* Atmosphere (3) */                                                                             \
  X(float,   airDensity,                                1.0f)                                      \
  X(float,   aerosolDensity,                            1.0f)                                      \
  X(Vector3, sunIlluminance,                            Vector3(20.0f, 20.0f, 20.0f))              \
  /* Sky/moon mood (3) */                                                                          \
  X(float,   nightSkyBrightness,                        0.008f)                                    \
  X(float,   moonNeeStrength,                           1.0f)                                      \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                      \
  /* Volumetric (4); volumetricAnisotropy avoids clash with cloudAnisotropy */                     \
  X(Vector3, transmittanceColor,                        Vector3(0.999f, 0.999f, 0.999f))           \
  X(float,   transmittanceMeasurementDistanceMeters,    200.0f)                                    \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.999f, 0.999f, 0.999f))           \
  X(float,   volumetricAnisotropy,                      0.0f)

// ---------------------------------------------------------------------------
// Per-field RTX_OPTION generator. Takes the preset name plus the X-macro's
// (type, fieldName, defaultVal) tuple and emits one RTX_OPTION declaration in
// the rtx.weather.preset.<presetName> namespace with getter
// presetName_fieldName.
// ---------------------------------------------------------------------------
#define WEATHER_PRESET_RTX_OPTION_FOR(presetName, type, fieldName, defaultVal)                     \
  RTX_OPTION("rtx.weather.preset." #presetName, type, presetName##_##fieldName, defaultVal,        \
             "Weather preset '" #presetName "' value for " #fieldName ". Override per-game in user.conf.")

// ---------------------------------------------------------------------------
// Per-preset binder macros. WEATHER_PRESET_FIELD_LIST(X) expects X to be a
// 3-arg macro (type, name, default), but RTX_OPTION generation also needs the
// preset name. Each binder closes over a specific preset name and forwards to
// WEATHER_PRESET_RTX_OPTION_FOR.
//
// Adding a new preset requires (a) defining a new binder here and (b) adding
// a DECLARE_WEATHER_PRESET line to DECLARE_ALL_WEATHER_PRESETS below.
// ---------------------------------------------------------------------------
#define WEATHER_PRESET_BIND_clear(type, name, def)         WEATHER_PRESET_RTX_OPTION_FOR(clear,         type, name, def);
#define WEATHER_PRESET_BIND_partlyCloudy(type, name, def)  WEATHER_PRESET_RTX_OPTION_FOR(partlyCloudy,  type, name, def);
#define WEATHER_PRESET_BIND_overcast(type, name, def)      WEATHER_PRESET_RTX_OPTION_FOR(overcast,      type, name, def);
#define WEATHER_PRESET_BIND_hazy(type, name, def)          WEATHER_PRESET_RTX_OPTION_FOR(hazy,          type, name, def);
#define WEATHER_PRESET_BIND_foggy(type, name, def)         WEATHER_PRESET_RTX_OPTION_FOR(foggy,         type, name, def);
#define WEATHER_PRESET_BIND_drizzle(type, name, def)       WEATHER_PRESET_RTX_OPTION_FOR(drizzle,       type, name, def);
#define WEATHER_PRESET_BIND_rainstorm(type, name, def)     WEATHER_PRESET_RTX_OPTION_FOR(rainstorm,     type, name, def);
#define WEATHER_PRESET_BIND_thunderstorm(type, name, def)  WEATHER_PRESET_RTX_OPTION_FOR(thunderstorm,  type, name, def);
#define WEATHER_PRESET_BIND_snow(type, name, def)          WEATHER_PRESET_RTX_OPTION_FOR(snow,          type, name, def);
#define WEATHER_PRESET_BIND_blizzard(type, name, def)      WEATHER_PRESET_RTX_OPTION_FOR(blizzard,      type, name, def);
#define WEATHER_PRESET_BIND_sandstorm(type, name, def)     WEATHER_PRESET_RTX_OPTION_FOR(sandstorm,     type, name, def);
#define WEATHER_PRESET_BIND_smoggy(type, name, def)        WEATHER_PRESET_RTX_OPTION_FOR(smoggy,        type, name, def);

// ---------------------------------------------------------------------------
// Single-preset macro. Walks WEATHER_PRESET_FIELD_LIST via the binder for
// preset N, emitting all 29 RTX_OPTION declarations. Must be invoked inside a
// class body (RTX_OPTION declares inline static members).
// ---------------------------------------------------------------------------
#define DECLARE_WEATHER_PRESET(N) WEATHER_PRESET_FIELD_LIST(WEATHER_PRESET_BIND_##N)

// ---------------------------------------------------------------------------
// Umbrella macro. Invoke inside RtxOptions struct body to declare all 348
// RTX_OPTIONs (12 presets x 29 fields).
// ---------------------------------------------------------------------------
#define DECLARE_ALL_WEATHER_PRESETS()   \
  DECLARE_WEATHER_PRESET(clear)         \
  DECLARE_WEATHER_PRESET(partlyCloudy)  \
  DECLARE_WEATHER_PRESET(overcast)      \
  DECLARE_WEATHER_PRESET(hazy)          \
  DECLARE_WEATHER_PRESET(foggy)         \
  DECLARE_WEATHER_PRESET(drizzle)       \
  DECLARE_WEATHER_PRESET(rainstorm)     \
  DECLARE_WEATHER_PRESET(thunderstorm)  \
  DECLARE_WEATHER_PRESET(snow)          \
  DECLARE_WEATHER_PRESET(blizzard)      \
  DECLARE_WEATHER_PRESET(sandstorm)     \
  DECLARE_WEATHER_PRESET(smoggy)

// ---------------------------------------------------------------------------
// WeatherSnapshot + WeatherBlender — Task 2 additions.
// Lives in dxvk::fork_weather namespace. Included by rtx_fork_weather.cpp;
// forward-use in rtx_fork_hooks.h needs only the hook forward declarations
// (no WeatherBlender include required there).
// ---------------------------------------------------------------------------
#include <string>

namespace dxvk { namespace fork_weather {

  // -------------------------------------------------------------------------
  // WeatherSnapshot — a plain-value copy of all 29 renderer weather params.
  // Members are auto-generated from the single-source-of-truth X-macro so
  // that any field addition automatically propagates here.
  // -------------------------------------------------------------------------
  struct WeatherSnapshot {
#define WEATHER_PRESET_FIELD_AS_MEMBER_(type, name, defaultValue) type name = defaultValue;
    WEATHER_PRESET_FIELD_LIST(WEATHER_PRESET_FIELD_AS_MEMBER_)
#undef WEATHER_PRESET_FIELD_AS_MEMBER_
  };

  // -------------------------------------------------------------------------
  // WeatherBlender — per-frame lerp pipeline.
  //
  // Reads __weather.target + __weather.blend_seconds from the GameStateStore,
  // lerps from m_previousSnapshot toward the named preset's RTX_OPTION values
  // over m_blendDurationSec seconds, and writes interpolated values into the
  // Derived layer of each underlying RTX_OPTION via setImmediately().
  //
  // Dormant when __weather.target is absent or unknown — zero upstream
  // behavioural change.
  //
  // Caller (Task 3) provides deltaTimeSeconds from the per-frame render loop.
  // ImGui surface (Task 4) implemented in showImguiSettings().
  // -------------------------------------------------------------------------
  class WeatherBlender {
  public:
    WeatherBlender();
    ~WeatherBlender();

    // Called once per frame from fork_hooks::updateWeatherBlender (Task 3).
    void update(float deltaTimeSeconds);

    // Renders the ImGui weather-preset panel. Placeholder until Task 4.
    void showImguiSettings();

    bool isPaused() const { return m_paused; }
    void setPaused(bool paused) { m_paused = paused; }

  private:
    // Preset cache — empty string means "not yet active".
    std::string m_previousPresetName;
    std::string m_targetPresetName;

    // Blend timeline.
    float m_blendStartTimeSec  = 0.0f;
    float m_blendDurationSec   = 1.0f;
    float m_currentTimeSec     = 0.0f;

    bool m_paused = false;

    // Snapshot of renderer state at the moment the last blend began (or the
    // retarget mid-blend captured the partially-blended state).
    WeatherSnapshot m_previousSnapshot;

    // Writes interpolated snapshot values to the Derived RTX_OPTION layer.
    void applyBlendedValues(float t);

    // Returns a WeatherSnapshot populated from the current renderer RTX_OPTION
    // getters (not from any preset table). Used at first-activation to seed
    // m_previousSnapshot so the initial blend transitions smoothly from
    // whatever the renderer was already doing.
    WeatherSnapshot snapshotCurrentValues() const;

    // Writes blend progress state back to the GameStateStore.
    void publishStateToGameStateStore(float t) const;
  };

} }  // namespace dxvk::fork_weather
