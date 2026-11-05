# Wetness integration

This local fork adds opt-in wet surface shading to the v2 scene path. Enable
`rtx.wetness.enabled` in Material Options > Wetness / Rain. It defaults off and
is independent of graphics quality presets because it changes the authored look.

## Instance classification

The D3D9 frontend snapshots private render state 210 for each submitted draw.
Set it through the D3D8-to-D3D9 converter's direct render-state function when
the host's D3D8 state API rejects private values. Clear or restore it after a
tagged draw; it is state, not a one-shot command.

| Bits | Meaning |
| --- | --- |
| 0..7 | Rain amount, 0 dry to 255 fully wet |
| 8 | Road classification |
| 9 | Reserved, preserved |
| 10 | Force wetness eligibility despite road and no-rain filtering |
| 11 | Disable wetness, overriding bit 10 |
| 24..31 | 1 vehicle, 2 world, 3 paint; other classes are dry |

The classification is instance metadata. It does not create a new shared
material or geometry identity. Changing it invalidates the retained draw
description, and the CPU resets DFF overrides from the source tag each frame.
Original geometry hashes survive replacement resolution for DFF matching.
The 256-byte GPU surface record stores a compact tag in the upper half of
data15.w. Its lower half remains the SCCR shell slot, including shell patching.

The world-up setting `rtx.zUp` must match the scene. Wetting and puddle slope
tests use that axis. Local vehicle droplet coordinates retain the host mesh's
local orientation. Optional wet shaders compile as separate variants; the
disabled variants retain their original names and definitions.

## Game values

Use `remixapi_Interface.SetGameValue` with UTF-8 text. The interface version and
layout are unchanged. The x86 bridge forwards these existing API calls.

| Key | Value |
| --- | --- |
| `__wetness.norain_zones_v1` | `1 count minX maxX minY maxY minZ maxZ ...` |
| `__wetness.rain_ending` | `1` while rain is ending, `0` otherwise |

Zone count is at most 4096. Coordinates must be finite and each minimum less
than its maximum. Invalid updates retain the last valid snapshot. `1 0` clears
all zones. Bounds are open, matching the plugin's cull-zone semantics. Zones
only filter shading when `respectNoRainZones` is enabled. The acceleration
buffer is replaced on changes so an in-flight frame retains its own snapshot.

The optional global rain-ending roughness override defaults off. A stale `1`
is limited to the greater of ten seconds and the configured transition time.
Publish `0` when the transition ends or on load; normal material roughness is
otherwise restored by the instance's rain amount returning to zero.

## Captured DFF rules

`forceEnableDff` and `forceDisableDff` are case-insensitive name fragments,
separated by semicolons or line breaks. Disable wins. The mapping is read from
`logs/DffDrawCalls` next to the host executable, or its parent when the host is
inside `.trex`. It uses DffCapture schema 1 JSON, not guessed hashes or model
names. Ambiguous shared geometry is left unclassified. Reload and diagnostic
counts are available in the wetness menu. See [DrawCapture](DrawCapture.md).

## Weather presets

NUMOS weather files can contain the float parameters in `rtx.wetness.*` and
`rtx.wetness.rainIntensity`. A missing key inherits the user's wetness setting;
missing rain intensity uses each instance's rain byte. `x2` multiplies the
corresponding user setting. Absolute rain intensity is clamped to 0..1.

The weather preset editor exposes these values under Wet surfaces. They blend
with the same selected preset, game weather blend and transition duration as
the sky. Computed values live on NumosWeather and are copied to frame constants;
they are never written back into RTX configuration layers. Preset resolution
continues when another sky provider is selected.

The existing legacy-file importer also accepts `wetnessPuddleNoiseScale` and
the other `wetness`-prefixed surface parameters, plus `precipitationIntensity`
as surface rain intensity. This mapping does not restore particle precipitation
or the removed `__weather.*` interface. Importing a field is not proof of
identical old/new sky appearance. Existing NUMOS migration rules still apply.

## Scope

Opaque wet sheen, puddles, ripples, local vehicle drops/streaks, translucent
drops and animated-water micro-wave detail are included. Shore wetting depends
on the separate water module and remains inactive until that module supplies
shore data. Physical water simulation and rain particles are separate features.

GPU validation must cover wet/dry transitions, multiple instances sharing one
material, exclusions and both up-axis conventions. Successful compilation or
CPU packing tests alone do not establish correct shading in a game.
