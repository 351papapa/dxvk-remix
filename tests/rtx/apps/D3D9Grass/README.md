# Procedural grass GPU fixture

Build `grass_smoke.cpp` as a console app with `public/include` and user32.lib.
Use x64 for the runtime directly, or x86 with the matching bridge client.
Place `grass-prototype.usda` beside the executable and `grass_smoke.json` in
its `logs/DffDrawCalls` folder. The bridge server resolves these paths relative
to the parent of `.trex`. Run `grass-smoke.exe <runtime-directory> <phase>`
from an isolated writable working directory.

Set `DXVK_RTX_CONFIG_FILE` to an absolute path to one of the supplied configs.
Set `DXVK_DUMP_SCREENSHOT_FRAME=110`, `DXVK_SCREENSHOT_PATH` and `DXVK_LOG_PATH`
to the output directory. Allow time for the first driver pipeline compilation.
The fixture renders 140 frames. Only the first two capture DFF tags; later
frames exercise retained scene reuse. Debug view 30 shows the emissive grass
on a dark nonemissive ground. The prototype is deliberately simple geometry.

Phases: 0 is Z-up; 2 is Y-up (use grass-yup.conf); 3 disables grass at frame 60;
4 reloads the USD; 5 reduces the budget to 32; 7 uses 16-bit indices; 8 uses
32-bit indices. Phase 9 adds 256 distant, distinct meshes; use grass-many.conf
to exercise parallel dynamic-BLAS emission. The default draw is nonindexed. Test both full and incremental
surface/material uploads, serial and parallel surface preparation, optional
range/frustum/edge culling, missing assets, and native/bridge paths.

Expected: visible grass when enabled; no grass when disabled or missing its
asset; reduced nearby coverage after the budget shrinks; matching masks for
indexed/nonindexed and serial/parallel runs with otherwise identical settings.
Missing assets must report the path error and leave rendering usable. Other
cases must have no new runtime errors or surface ownership/self-check failures.
These checks do not measure game frame rate or replace full game validation.
