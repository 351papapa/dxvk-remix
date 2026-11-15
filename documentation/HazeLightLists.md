# HAZE local-light lists

The NUMOS aerial-perspective path selects its own lights. RTXDI's surface-light
reservoir budgets do not determine these column lists. The original cluster and
whole-depth column records contain a count followed by at most 16 indices.
Discarding later candidates can remove an entire local-light shaft.

## Representation and compatibility

The host-generated overlay keeps the upstream record stride of 17 uints (68 bytes).
Header bit 31 selects grouped format; the remaining bits hold the actual number
of intersecting candidates K. The 16 payload words provide 512 group bits. With
N = the same frame's apLocalLightCount, group size is max(ceil(N/512), 1). Bit g
represents every index in [g*groupSize, min((g+1)*groupSize,N)). For N <= 512 this
is exact; up to the supported global maximum 4096, extra members are conservative.
All payload words are rewritten every cull dispatch, including empty records.
The current constants, packed light table and records must belong to the same
frame: a count crossing 512 changes decoding. K alone cannot detect stale data.

Legacy headers still read at most 16 indices. The host validates original-pass
FNV, payload FNV and layout version 1 for all three sidecars before selecting
any of them. Missing/stale files cause whole-set fallback, with an explicit log.
The embedded composite accepts both formats. Do not deploy these sidecars with
an older runtime whose embedded composite only accepts indices.

## Geometry and cost

Conservative cone rejection narrows column/cluster candidates, with fallbacks for
wide cones and numerical uncertainty. The airlight consumer rejects ray segments
with zero possible contribution before its eight taps. It does not redistribute
the sample locations, change shadow sampling, lower quality, or alter exposure.
Retaining previously dropped lights can cost additional GPU work, so performance
must be measured in the same settled scene. CPU mathematical tests do not predict
the total GPU frame-time change.

## Build and diagnostics

`scripts-common/numos_haze_shaders.py` generates modified copies outside the NUMOS
submodule, preserving its Apache 2.0 notices and validating pinned source hashes,
descriptor ABI and specialization constants. Meson generates the composite
include overlay before building the main shaders. The normal NUMOS build emits
the cull, untraced lighting and traced lighting .haze.spv/.haze.meta pairs.

Internal diagnostics are off by default and have no persisted quality settings:

- `RTX_NUMOS_HAZE_REFERENCE=1`: use all three library passes for an A/B reference.
  The common composite still uses the dual decoder and conservative ray rejection;
  this is a list-pass reference, not an unmodified upstream DLL.
- `RTX_NUMOS_HAZE_PROBE=1`: read up to 300 eligible frames through four asynchronous
  staging slots. No waitIdle or per-frame file open is used.
- `RTX_NUMOS_HAZE_PROBE_START_FRAME`: first frame ID, default 1800.
- `RTX_NUMOS_HAZE_PROBE_ARM_FILE`: optional absolute marker path, checked about
  every 250 ms only when the probe is enabled. The path must first be observed
  absent, then appear as a file. This one-shot mode starts the 300-frame budget
  at the trigger frame and supersedes START_FRAME. It never creates/deletes the
  marker or restarts a completed burst. Confirm the scene before creating it.
- `RTX_NUMOS_HAZE_PROBE_TILES`: three normalized screen positions `u,v;u,v;u,v`.
  Mapping uses the froxel size and tile size, including partial edge tiles.

The log reports CPU/GPU byte equality, light count, fog/history parameters, raw
column records, decoded membership, and actual GPU light positions/radiances.
Missing culls, unavailable readbacks, full rings and pending shutdown records are
reported explicitly. The probe is a diagnostic observer, not a synchronization fix.

Acceptance requires stable camera/player streaming origin, FOV, resolution,
settled weather and time. Check target membership, recorded shaft continuity,
HAZE pass timings and total GPU time. Exclude capture/readback intervals from performance measurements and report the
actual frame-generation state, including when user settings override an environment
value. A probe-enabled run after its completed burst is not a probe-disabled run.
