# Procedural grass

## Configuration

Enable Procedural Grass in Material Options. `rtx.grass.enabled` defaults off.
Choose a static USD prototype with a default prim, supported bound materials,
and nonzero height. Relative paths resolve against the host executable folder;
the bridge server uses the parent of `.trex`. Absolute paths are supported.
The prototype is converted to local Z-up, centered horizontally and normalized
to height one with its base at zero. Height and width then control its size.

`diffuseTextures` selects replacement albedo filenames. `originalTextures`
selects captured original texture names on meshes without a mesh replacement.
Both lists are semicolon-separated, exact and ASCII case-insensitive. Original
names come from `logs/DffDrawCalls` mappings; ambiguous names are skipped.
These are content-selection controls, independent of the graphics presets.

Radius, density, height, width, maximum slope, fade distance, ground offset
and instance budget control placement. The budget is bounded to 1–512 clumps,
with nearest visible roots taking priority. Optional front/rear range reduces
the rear radius to half. Optional frustum culling removes distant offscreen
roots; disabling it preserves grass for offscreen shadows and reflections.
Optional boundary insetting keeps prototype bounds inside open ground edges.
All three switches default off. Increment `reloadVersion` to reload the USD.
Every option has a menu control. The menu also shows loading status and counts.

## Scene and worker lifetime

The scene owns GrassSystem. It scans the unified instance table, including direct
retained original geometry without a ReplacementInstance owner. Original texture
identity remains separate from replacement material identity. Grass-generated
instances are excluded as placement sources and from anti-cull preservation.

CPU-readable triangle-list sources support indexed 16/32-bit and nonindexed
geometry, float3/float4 positions and bounded strides. Skinned geometry and
pending GPU writes are skipped. Unmappable geometry is not read back implicitly.
Local oriented triangles are cached by geometry hashes; world transforms identify
placement patches. Scene clearing, source/configuration changes and reloads
cancel obsolete work. Two owned workers generate deterministic area-weighted
roots and pack them in identity order. Geometry and root budgets bound the work.

World slope, horizontal distance and front/rear tests use `rtx.zUp`. The USD
prototype remains local Z-up and its root transform aligns to the ground normal.
Frame data is passed in typed constants, without writing derived values to options.

## GPU integration

Dedicated compute entries initialize selection, cull, choose nearest roots by
radix selection, prefix-sum and scatter accepted roots, then emit surface and
TLAS records. No tracing shader variant is needed for these separate passes.
Each grass batch reserves a CPU-owned template surface followed by GPU-owned
clump slots. Both serial and parallel emission account for the template, retain
the 21-bit surface limit, and participate in the existing ownership self-check.
The GPU surface record remains 256 bytes. Ordinary point instancers use their
existing culling path; grass batches use their own selection pass.

Allocation identities associate batches across frames without raw-pointer reuse.
Previous root identities and slot mappings preserve motion history as selection
changes. Dispatch occurs after surface/material upload and before TLAS build.
Turning the feature off retires its instances and clears placement history.

## Validation status

Release build and the placement, window-lighting and wetness unit tests pass.
Seventeen synthetic GPU cases cover native/x86 bridge, Y-up/Z-up, indexed and
nonindexed ground, full/incremental and serial/parallel uploads, reload, disable,
budget reduction, optional culling and missing assets. A 257-group scene exercises
parallel per-geometry BLAS emission and its sequential-reference self-check.
Two additional window/wetness GPU regressions preserve the established masks.
See `tests/rtx/apps/D3D9Grass` for reproduction. Full-game frame-rate acceptance
is still pending. The existing common-device-object shutdown warning remains.
