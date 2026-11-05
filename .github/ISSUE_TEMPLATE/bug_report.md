---
name: Bug report
about: Report crashes, rendering issues etc.
title: ''
labels: ''
assignees: ''

---

Please describe your issue as accurately as possible. Remix Plus is not an NVIDIA project: report Remix Plus bugs here, not on NVIDIA's RTX Remix tracker.

**Game and setup:** the game, 32-bit (bridge) or 64-bit, any Remix mods, your GPU and driver version, and the Remix Plus version (the `DXVK_Remix:` line near the top of `remix-dxvk.log`).

**What happened:** what you did, what you expected, what you saw. Add a screenshot for rendering issues.

**Logs:** attach `remix-dxvk.log` from the game's `rtx-remix/logs/` folder, and for a 32-bit game also `bridge32.log` and `bridge64.log`. After a crash, also attach the newest `.dmp` file from `rtx-remix/logs/.sentry-native/reports/`. If you changed settings, attach your `rtx.conf` and `user.conf`.
