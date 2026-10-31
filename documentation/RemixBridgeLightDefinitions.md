# Bridge light definition updates

The 32-bit bridge can update a light through its existing bridge handle. Clients
must first query `GetGameValue("__bridge.lightDefinitions.version", ...)` and
require the null-terminated string `"1"` with a returned size of two bytes.
This requires a matching client, server and runtime. Older pairs keep using
`CreateLight`. The public ABI and all existing command numbers are unchanged.

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

The capability query uses separate public, server and runtime names. An old
client does not translate the public name and cannot accidentally advertise
support merely because it is paired with a new server.

This is a transport and lifetime optimization. No quality options or presets
change. CPU validation, cross-architecture serialization and game-route evidence
are recorded with the integration artifacts.
