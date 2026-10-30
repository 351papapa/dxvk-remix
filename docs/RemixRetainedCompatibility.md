# Local retained-catalog and captured-surface compatibility

This local fork restores the fixed-width host protocol used by WorldMeshVC. It is separate from the public 1.x retained API; the public API table and version have not changed. New integrations should use the public retained scene API. Existing hosts must resolve and negotiate this private interface explicitly.

## Entry points and contracts

`remixapi_GetRetainedInterface(2, sizeof(remixapi_RetainedInterface), &table)` is a named export of the native runtime and the matching bridge client. `remix_retained.h` defines the unchanged 168-byte record, 80-byte blend block and 40-byte status. A batch contains at most 4096 records. Bridge mesh handles are translated on the server; pointers are never passed between processes.

- `Clear` starts each session with a strictly increasing nonzero epoch. A successful newer Clear resets sticky errors, counters and the catalog. A stale or repeated epoch is rejected asynchronously.
- `BatchUpsert` and `BatchRemove` change the catalog in command order. Mesh destruction removes its dependent private records. Missing materials fail replay closed.
- `CommitFrame` enables the unordered set exactly once for the current runtime frame. The next frame must commit again after camera acquisition and before RTX injection; a missing commit hides the unordered set. Resuming does not require resending unchanged records.
- `DrawOrdered` validates the full ID group before submission and applies its order through the scene's existing decal-order counter. Ordered records must be explicitly drawn each frame. An invalid group cannot leave a visible successful prefix.
- `GetStatus` drains CPU command processing before returning the catalog status. It does not wait for GPU completion or read GPU buffers. Hosts should query during initialization, reset or diagnostics rather than each frame.

The adapter owns descriptions and dependency indices, while the new `RetainedScene` owns persistent render instances, shared geometry and GPU instancing. Ordinary unchanged commits perform no per-record rebuild or draw resubmission. A mesh/material event touches its indexed users; scene resets, texture-cache resets, mod changes and relevant translation-option changes rebuild affected descriptions. There is no old retained backend, quality reduction or CPU expansion of GPU instances.

## Captured legacy surfaces

`remix_legacy_surface.h` and `remix_legacy_surface_v2.h` attach per-surface original identities. These must come from the current session's observed game draws. Do not fabricate identities for real game content. The geometry's API storage/cache identity remains distinct from its original replacement/capture identity.

The runtime verifies the configured hash-rule fingerprint, nonzero draw/geometry keys, the draw = geometry XOR material relation, surface counts and extension exclusivity. Material zero is a valid untextured identity. V2 also carries the original absence of normals; a missing normal stays missing instead of being replaced with an invented normal buffer. Unknown flags, reserved bits, invalid strides and malformed payload lengths are rejected.

V2 support is an end-to-end handshake. A bridge host requires `GetGameValue("vc.client.legacyVertexAttributesVersion") == "2"`; the client rewrites the query to a bridge key and the server rewrites it to the runtime key. The native runtime advertises only `vc.api.legacyVertexAttributesVersion`. Intermediate keys are not independently advertised. A size probe returns 2 including the terminator; a one-byte buffer remains untouched. Do not infer support just from a header or synthesize a capability response.

Original texture categories include zero-texture rules, normal smoothing, decal variants, baked-lighting/opacity exclusions and player/particle categories. Material replacements resolve by the original game texture key; capture keeps the observed material key. Mesh replacements resolve by the original draw key through the current frontend resolver.

## Game texture material binding

The matching bridge implements `dxvk_GetTextureHash` for its D3D9 texture objects. The native frontend drains pending CPU commands, obtains the existing DXVK image hash and registers the actual texture view with the scene texture table. It neither hashes pixels on the CPU nor reads pixels back from the GPU. `CreateMaterial` accepts `game:0x<16-digit-hash>` and confirms that an opaque material resolved a valid albedo texture before returning success. Failure returns a null handle.

`vc.api.gameTextureMaterialVersion`, `vc.api.gameSurfaceVersion`, `vc.api.legacySurfaceVersion`, `vc.api.zeroLegacyMaterialVersion` and `vc.api.retainedOrderedVersion` return `"1"`. `vc.api.legacySurfaceRule` returns the active rule fingerprint in hexadecimal.

The provided old source does not register `rtx.vcApi.textureHashMode`; its current texture lookup always uses `DxvkImage::getHash()`, CPU command synchronization and texture-table registration. That behavior is preserved directly here. The obsolete configuration key is not a capability signal. Old compact-vertex/index experimental storage toggles are not reintroduced.

## GPU-instancing bridge payload

The existing `InstanceInfoGpuInstancingEXT` serializer is now connected to `DrawInstance` in both bridge endpoints. One payload contains the 12-byte serialization header followed by contiguous 48-byte transforms. The bridge accepts up to 65536 transforms (3 MiB); nonzero count with null data, a duplicate extension, an over-limit count, mismatched stype and any nonexact payload length are rejected. The client sends one draw command, not one command per transform. Count zero retains the native public API's ordinary single-instance fallback semantics.

## Validation and support limits

Plugin policy tests are distinct from live runtime verification. The migration fixture exercises native and bridge texture lookup, real V2 negotiation, retained/ordered lifetimes, epoch recovery, transforms, mesh replacement, resource destruction and GPU-instance counts 0/1/128. Static 2048-record runs check stable catalog/submission counters without status calls in every frame. Compare actual AOV and composite images with controls and inspect instance/GPU counters before accepting visual or frame-time equivalence.

The private named exports are implemented for the native x64 runtime and x86 bridge/x64 server. Upstream ARM64/ARM64EC shims are unmaintained and have not been extended or validated for this local protocol.

Provenance: fixed-width protocol definitions and compatibility semantics were adapted from the user's local rebased9-19 fork. NVIDIA-derived API/bridge source retains its original MIT notices; see `CREDITS.md`, `LICENSE` and `LICENSE-MIT`.
