# Window and street lamp lighting

Enable `rtx.windowLight.enabled` in Material Options > Window Lighting. It
defaults off. This module gates an existing opaque material's emission; the
material must already have an emissive color/texture and nonzero intensity.
It does not create point lights or replace the emission texture.

## Matching

`textureNames` contains case-insensitive name fragments separated by semicolons
or line breaks. It matches replacement albedo filenames and original texture
names from schema 1 DffCapture JSON under `logs/DffDrawCalls` next to the game
executable (its parent for a `.trex` host). Ambiguous original names are skipped.
The captured original texture hash survives replacement and retained reuse.
Use the wetness menu's capture-map reload button after changing capture files.

`streetLampsEnabled` defaults off. When enabled, `streetLampTextureNames` uses
the same matching rules and takes precedence over window rules. Lamps switch
as a group and preserve their original emission footprint; windows are randomly
gated by connected geometry and UV tile. Eye materials are excluded.

## Host clock

Send UTF-8 values using `remixapi_Interface.SetGameValue`. The existing x86
bridge forwards these keys; no C ABI change is required.

| Key | Value |
| --- | --- |
| `__window.hour` | Finite decimal game hour, wrapped into 0..24 |
| `__window.day` | Unsigned 32-bit calendar day, incremented at midnight |
| `__window.badweather` | `0`/`1`, `false`/`true` or `False`/`True` |

Malformed updates preserve the last valid value. Keys are independent; publish
all three on startup and after loading a save. RemixCoreVC sends changed values
and retries failed API writes. Missing time uses a constant lit ratio and keeps
lamps on; missing day uses the manual `nightSeed` alone.

`startHour` and `endHour` define a night that may cross midnight. The lit share
starts at `litRatio` and falls linearly to zero at the end. Outside the interval,
windows and lamps are off, except that bad weather enables lamps. Equal start
and end hours select a constant night. `nightSeed` changes the random pattern.
For a night crossing midnight the preceding day's ordinal remains in use after
midnight, so the pattern does not reshuffle until the next evening.

## Implementation and limits

The 256-byte surface record remains unchanged in size. Bits 15..16 of data13.z
store classification; non-eye data15.xy stores island offset and stable hash.
Eye origins, SCCR shell slots and wetness bits keep their original layout.
Offsets belong to each uploaded surface slot, including split meshes, rather
than being overwritten on a shared CPU instance. Serial, parallel, incremental
and GPU point-instancer uploads all use the same record format.

Island roots are cached by geometry content. Exact equal positions join
duplicated quad vertices, and the shader adds the integer UV tile to the random
identity. Float32 positions and uint16/uint32 indices are read with byte bounds
checks. Unmappable or unsupported source formats fall back to one island per
triangle; two-triangle window quads then require CPU-readable source geometry
to stay together. This limitation is inherited from the original CPU topology
approach and is explicit rather than reading GPU-local memory as CPU data.

The emission gate is a small uniform branch and integer hash in the existing
material evaluator; it does not introduce a new tracing pass or texture read.
The larger wetness shader variants remain independently selectable. No
performance improvement is claimed without a scene benchmark.

## Validation

`test_window_light` covers midnight/day changes, missing clock values, lamp
weather override, packed ABI isolation, name rules, duplicated vertices,
16/32-bit indices and truncated buffers. `tests/rtx/apps/D3D9WindowLighting`
contains the GPU fixture: eight independent quads in one retained D3D9 draw.
The migration validation manifest records which GPU/bridge cases were run.
