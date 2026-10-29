# Window lighting GPU fixture

Build `window_smoke.cpp` as a console app with the public Remix include path
and user32.lib. Run `window-smoke.exe <runtime-directory> <phase>` from an
isolated working directory. Use the matching bridge client for an x86 build.

The fixture has eight separated quads with duplicated corner vertices in one
draw. It captures only frames 0 and 1; the remaining 108 frames use normal
retained draw processing. Place a schema 1 mapping for its constant texture
hash `b5eb6b71a7136b32` with texture name `test_window` in `logs/DffDrawCalls`
next to the executable. Use `rtx.windowLight.textureNames = test_window`.
Enable legacy emission with white color and intensity 1, use debug view 30,
and capture frame 80 with `DXVK_DUMP_SCREENSHOT_FRAME` / `DXVK_SCREENSHOT_PATH`.

Phase 0: night. Phase 1: noon. Phase 2: noon and bad weather. Phase 3: change
night to noon at frame 60. Phase 4: disable the module at frame 60. Lamp tests
also enable streetLampsEnabled and match `test_window` as street lamps.
Compare serial and parallel uploads with accelIncrementalSelfCheckFrames = 1,
and test wetness variants independently. All API writes and device calls must
succeed, and the process must exit normally. These are synthetic GPU checks,
not a complete game traversal.
