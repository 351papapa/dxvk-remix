# Draw correlation capture

The D3D9 frontend accepts an opt-in correlation pair in render states 213
(session) and 214 (draw token). Both must be nonzero. Reset the token to zero
after the captured draw, and reset both states when the session ends.
The states belong to the device. Their values are copied into the queued draw
record before the command-stream thread runs it.

A tagged draw uses the frontend's per-draw path, bypassing the unchanged-object
early check. Once its geometry hash futures complete, the frontend appends two
CSV records under the host executable's `logs` directory. For the bridge server
in `.trex`, the directory belongs to the parent game directory.

- `dff_remix_hashes.csv`: decimal session and token, followed by hexadecimal
  replacement mesh hash, first color texture hash, and geometry hash.
- `dff_remix_states.csv`: decimal session and token, alpha-test enable/reference/
  compare operation, blend enable, color source/destination/operation, texture
  color arguments/operation, texture alpha arguments/operation, texture factor,
  texture-factor blend flag, alpha source/destination/operation, color write mask,
  baked vertex lighting flag, and cull mode.

The CSV names and column order are compatible with DffCapture. The frontend
does not parse game models. It records only draws accepted for ray tracing.
The mesh hash uses the runtime's configured geometry hash rule and original
material, rather than a hash calculated by the plugin.

Records are deduplicated within a session and flushed before returning. Failed
writes are not marked as seen, so later draws can retry. Session changes clear
the in-memory deduplication sets and append to the existing files.

This interface adds no Remix API table slots, named exports, or default-enabled
settings. The x86 bridge's existing ordered SetRenderState transport carries the
two values without a new wire command.
