# Game post-processing tint

The `__postfx.tint_v1` GameStateStore convention supplies a live RGB feedback
coefficient to external post-processing effects. It uses the existing
`SetGameValue` API and adds no exported functions or bridge commands.

## Producer

Send one string containing `valid red green blue`, with whitespace-separated
values. Valid data uses `valid = 1` and finite RGB values in the inclusive range
0 to 1. These are authored feedback coefficients; the runtime performs no
color-space conversion. For an 8-bit source, divide each channel by 255.

Publish all four fields in one transaction when the effective value changes.
Send `0 0 0 0` on reset, loss of the source or a camera mode that does not supply
the intended tint. The runtime rejects missing, malformed, overlength (over 128
bytes), nonfinite and out-of-range data. A shader chooses its own fallback when
the validity flag is zero. Producers must publish invalidity explicitly when a
source stops being available; the value is not a clocked sample.

## External shader interface

The frame buffer at descriptor binding 13 appends these members:

| Member | Type | Byte offset |
| --- | --- | --- |
| `gamePostFxTint` | `float3` | 1360 |
| `gamePostFxTintValid` | `uint` | 1372 |

The frame size is 1376 bytes. All existing member offsets remain unchanged.
Older effects remain compatible. Effects reading the appended members require
the matching runtime and bindings header; install the runtime before those
effects. This data does not use RTX option layers, saved effect parameters or
per-frame configuration writes.

An effect can blend changing tint values normally while resetting history when
it switches between live tint and its manual fallback. The application owns the
source semantics; the runtime does not parse a game's weather or timecycle file.


## Camera medium flags

The frame's former reserved word at byte 1356 is cameraMediumFlags (uint).
Bit 0 identifies a resolved camera water medium; remaining bits are reserved.
Classification compares the configured nonzero water-medium hash to the
current camera's resolved source material hash. Disabling water volume lighting
does not change this camera classification. The frame size remains 1376 bytes.
The value is runtime-generated and does not add a GameStateStore producer key.

Effects built with this field require the matching runtime for water detection.
Older runtimes leave the reserved word zero, while older effects ignore it.
All existing non-padding fields retain their offsets.
