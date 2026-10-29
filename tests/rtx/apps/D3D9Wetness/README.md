# D3D9 wetness GPU regression scene

Build from an x64 Visual Studio command prompt with the repository public/include
folder on the include path. Link user32.lib. For the x86 bridge build, define
REMIX_ALLOW_X86 and use the same public header.

Run `wetness-smoke.exe <runtime-directory> <case>` in an isolated test folder.
For the x86 build, runtime-directory is the folder with the x86 bridge d3d9.dll;
its .trex folder must contain the matching server/runtime and a bridge.conf with
`exposeRemixApi = True`. No game installation is required or changed.

Cases: 0 dry; 1 dry then wet; 2 dry/wet/dry; 3 blocked by an API no-rain box;
4 force eligibility inside that box; 5 clear the box mid-run; 6 Y-up wet plane.
The three draws share a vertex buffer, texture and material. The left draw has
no rain amount, the middle draw changes, and the right draw explicitly disables
wetness. No draw capture session is enabled, so the retained path is exercised.

Use a config with `rtx.wetness.enabled = True`, `rtx.zUp = True` (False for case 6),
`rtx.debugView.debugViewIdx = 25`, `rtx.debugView.replaceCompositeOutput = True`,
`rtx.initializer.asyncShaderPrewarming = False`, and
`rtx.skipObjectsWithUnknownCamera = False`. For cases 3..5 also set
`rtx.wetness.respectNoRainZones = True`.

Set DXVK_CONFIG_FILE to that config, DXVK_DUMP_SCREENSHOT_FRAME=80 and
DXVK_SCREENSHOT_PATH to a unique output folder. The runtime exports a 320x240
RGBA16F DDS debug view. The final test writes 110 presents and releases its device.
The process must exit zero and print `Wetness draw submission complete`.

Read the 3x5 pixel patches around (105,115), (160,115), (215,115). Wet cases
1,4,5,6 should make the middle patch less than half the right patch while the
left/right controls stay within 20%. Cases 0,2,3 keep all three within 20%.
The absolute values vary with temporal reconstruction, so compare controls
within the same image rather than assuming an exact hardcoded pixel value.

For weather integration, point rtx.weather.presetDirectory to isolated presets
and select a preset via rtx.weather.preset. A preset with
`rtx.wetness.rainIntensity = 0` keeps every draw dry. A preset with intensity 1
wets both eligible draws, including the left draw whose own rain byte is zero;
the explicitly disabled right draw stays dry.

On the validation machine, running inside a restricted process sandbox caused
NVIDIA NGX shutdown to wait for its telemetry named pipe. The same binary with
normal process permissions exited correctly. The baseline runtime also reports
53 common device objects not disposed of at shutdown; this test does not treat
that pre-existing log message as proof of a wetness failure.
