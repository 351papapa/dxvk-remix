# D3D9 GPU water fixture

Build `water_smoke.cpp` with MSVC C++17, the public include directory and
`user32.lib`. Define `REMIX_ALLOW_X86` for the bridge build. Run in an isolated
working directory with arguments `<runtime directory> <phase>`; the directory
contains the matching runtime dependencies or bridge client/server pair.

The fixture renders a sloping shore, supplies the real GPU surface and solver
protocol, and checks camera-medium calls plus consumer acknowledgements.
Phases: 0 normal waves, 1 flat input, 2 disable, 3 disable/re-enable, 4 domain
translation, 5 underwater camera, 6 wave/flow/hull events, 7 invalid grid input,
8 paused producer clock, 10 long-run paired foam vertex diagnostics,
14 two impulses in one packet, 15 the same impulses in two queued packets.
Phases 14/15 must use identical configuration and capture frame for comparison.

Set `DXVK_RTX_CONFIG_FILE` to the included configuration and replace the four
asset placeholders with DDS files: white RGBA foam, flat tangent normal and a
nonconstant caustic pattern. Capture frame 140 through
`DXVK_DUMP_SCREENSHOT_FRAME`; set output/log directories independently.
Use the same inputs for native and bridge comparisons. Tests must be serialized
because the legacy shared-memory names are process-global.

After compiling the shaders, run `check_foam_spirv.py <rtx_shaders directory>`.
It verifies unsigned world-cell addressing and thread-group size in the actual
compiled kernels. Runtime paired diagnostics must report zero repeated and
cross-variant mismatches; process exit alone is not sufficient.
