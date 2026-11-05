# Weather-controlled volumetric fog

Enable **Weather controls volumetric fog** in Sky > Weather to let the active weather preset change global volumetrics. The option is `rtx.weather.controlVolumetricFog`, defaults to false and does not enable the volumetric renderer itself. It is independent of graphics quality presets.

In Weather Presets > Volumetric fog, tick a field to override it, or leave it unticked to inherit the global value. Float and colour overrides can multiply the base value. Save writes the preset file; preview shows it without changing game weather.

Preset files accept the same `rtx.volumetrics.*` names as global settings for transmittance colour and distance, single-scattering albedo, anisotropy, sun/consumer gains, heterogeneous noise, froxel distance, fixed-function fog remapping, translucent shadows and depth offset. For example:

```ini
weather.bindings = rain
weather.transitionSeconds = 15
rtx.volumetrics.transmittanceMeasurementDistanceMeters = 100
rtx.volumetrics.fogSunVisibilityGain = 1
rtx.volumetrics.enableTranslucentShadows = true
```

These fields control the global froxel fog system. NUMOS height fog is a different effect and is not a substitute for them. The legacy field name `volumetricAnisotropy` is accepted when importing old presets and saves as `rtx.volumetrics.anisotropy`.

The game integration only reports current/outgoing weather IDs and, optionally, its blend fraction through the NoSave game-weather fields. It does not write the changing fog values into the user configuration.

See [implementation details](WeatherVolumetricsDesign.md) for interpolation and ownership.

## Migrated game weather presets

The migrated game configuration enables `rtx.weather.followGame` and loads six editable files from `rtx-remix/weather-rpv2rc2`. `weather.bindings` maps game weather IDs 0 through 5 to clear, partly cloudy, rainstorm, foggy, overcast and thunderstorm presets respectively. The plugin publishes the incoming/outgoing game weather IDs; the preset files own appearance and transition duration.

Unsupported legacy sky parameters are omitted. They use the new runtime base/default values, and their historical values in `unmapped.json` are reference only. Edit the corresponding preset in the Weather Presets menu or its `.conf` file to tune the look. The default migration keeps the 15-second transition, independent wetness state and existing rain/storm precipitation intensity.
