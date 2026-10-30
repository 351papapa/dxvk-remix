# Bridge capability negotiation

The local x86 bridge exposes `vc.bridge.autoInstancePersistentLightsVersion` through the existing `GetGameValue` function. A supported bridge returns the null-terminated UTF-8 string `1`; the required buffer size is2bytes including the terminator. A size query uses a null output buffer with size0. A smaller output buffer is not written. Usual invalid argument checks still apply.

This key describes the client bridge's ability to serialize the ordered `AutoInstancePersistentLights` command. It is answered locally by the client rather than forwarded to a server that might be newer. Use a matched bridge client/server pair. The native runtime already implements the operation and does not need this bridge probe.

An older bridge may expose a non-null no-op function slot. Hosts must treat an absent key, a failed query or a value other than `1` as unsupported and explicitly submit their live lights. `UpdateLightDefinition` remains unsupported by this bridge; this capability does not advertise that function. Public C API structs, function signatures and version are unchanged.

## D3D9 camera and skinned meshes

`SetupCamera` is null in the x86 bridge interface. The game's D3D9 view and projection transforms provide its camera. Plugins that also support a native API host must check the function pointer before requiring an explicit camera submission. WorldMeshVC uses this D3D9 camera when the slot is absent.

Skinning array counts are total element counts, including all vertices and bone influences. The bridge transfers exactly `blendWeights_count * sizeof(float)` and `blendIndices_count * sizeof(uint32_t)` bytes. Build and deploy both bridge architectures together. Skinning and instance transforms continue to execute on the GPU.

The protected-buffer regression covers 3, 16 and 256 vertices with 1, 2 and 4 influences, checks exact payload lengths and roundtrip contents, and rejects source overreads. It reproduces the former bridge overread before the fix.

## Empty retained replacements

An empty USD mesh replacement is a valid suppression of its original geometry. The retained catalog stays active, and later record updates or mod reloads can restore meshes. Replacement lights require a drawn mesh anchor, as on the native D3D9 path.
