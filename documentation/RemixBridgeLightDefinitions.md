# Bridge light definition updates

The 32-bit bridge can update a light through its existing bridge handle. Clients
must first query `GetGameValue("__bridge.lightDefinitions.version", ...)` and
require the null-terminated string `"1"` with a returned size of two bytes.
Support requires the new client/server pair and a capable runtime. A matching
older bridge pair with the new runtime, or the new pair with an older runtime,
keeps using `CreateLight`. The public ABI and all existing command numbers are unchanged.

The bridge accepts `LIGHT_INFO` followed by exactly one sphere, rectangle, disk,
cylinder, distant, dome or USD extension. Multiple extensions and cycles are
rejected before writing any command bytes. Fields use the existing cross-arch
CreateLight serializers, including owned USD optional values and dome paths.

`UpdateLightDefinition` success acknowledges queue admission. Runtime rejection
is reported by the ordered query `__bridge.lightDefinitions.status`, which returns
`processed=N;errors=N;lastError=N`. Errors and lastError remain sticky for the
bridge session. This query waits for earlier device commands to be handled, but
is not a GPU completion fence. Do not query it for every light.

Create a light before updating it. Retain the same handle after an update and
continue existing DrawLightInstance and visibility decisions. Definition updates enter the render command stream in call order, before the
native D3D9 endFrame injection and later queued destruction. A separate
post-Present update queue would delay moving lights and could restore an earlier
update after destruction. General batched create/destroy ordering is unchanged;
callers must not use destroyed handles.

The capability query uses separate public, server and runtime names. The server
capability name is private to the bridge. An old client cannot advertise support
by forwarding the public query to the runtime.

The existing bridge handshake requires matching client/server build versions.
Do not mix their binaries: this is rejected before capability negotiation.
The new-client/old-server negative test recorded version rejection but required
the test harness to terminate its own client after 100 seconds. It is an
unsupported combination and is not a successful fallback test.

This is a transport and lifetime optimization. No quality options or presets
change. CPU validation, cross-architecture serialization and game-route evidence
are recorded with the integration artifacts.

## Validation on 2026-10-31

The compiled source is `c9e3cac7969e206c7e5e20b7eda69538619b7ad3`.
Both bridge architectures passed seven unit suites and cross-architecture
serialization in both directions. A real x86-client/x64-server/runtime fixture
passed 2,637 checks covering seven light forms, 700 definition changes, stable
handles, black lights, destruction, rejection of a stale handle, and device
reset. Both supported fallback combinations passed 94 checks each.
All 1,595 Slang SPIR-V binaries match the previously accepted runtime exactly.

One discarded 60-second warmup and one 300-second route per configuration gave
33.698 -> 34.028 base FPS (29.676 -> 29.388 ms), 1% low 18.370 -> 18.309 FPS,
P99 47.494 -> 47.791 ms, and 74 -> 68 frames over 50 ms.
This is a single-run observation, not an established percentage gain.
The comparison includes the matching Core host adaptation and keeps the
accepted WorldMesh change and rendering settings fixed.
The game-session log reached at least 524,288 updates with zero
recorded rejection errors. Power-of-two snapshots do not give an exact formal
interval total. Visual spot checks do not establish pixel equivalence.
