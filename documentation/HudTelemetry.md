# HUD telemetry export

The HUD can export the same visible text and its base-frame timing history to CSV.
Export is disabled when `dxvk.hudExport` is empty. It does not read GPU timing
queries or move rendering calculations to the CPU.

## Enable a capture

Create the output directory first, then add these settings to the game's configuration:

```ini
dxvk.hud = fps,frametimes
dxvk.hudExport = "telemetry/dxvk_hud.csv"
dxvk.hudExportIntervalMs = 250
rtx.showDxvkHud = True
```

Select HUD items with `dxvk.hud` or the existing `DXVK_HUD` environment variable.
The `frametimes` item is required for raw base-frame history. The `fps` item adds
the displayed FPS estimate. Export includes only selected, visible HUD items.

The Remix graphics menu exposes **HUD CSV output path** and **HUD CSV sample
interval (ms)** beside **Show DXVK HUD**. Confirm a path with Enter or leave the
field to apply it. Clear the path to stop exporting. The interval is clamped to
16-10000 ms and does not enable export on its own.

Starting an export or changing its path replaces the target file. Relative paths
use the runtime process working directory. An inaccessible path is logged once
and is not retried every frame. Correct the path, or clear it and enter it again
after making the parent directory writable.

## Interpret the results

`frametime_us` measures the interval between base-frame HUD updates. It excludes
additional frame-generation presents. It measures the runtime's base submission
cadence, not game simulation ticks or GPU execution time.

The visible **FPS** text can include generated frames. It is a rounded, rolling
estimate and must remain separate from base-frame average FPS, lows and frame-time
percentiles. Keep graphics, frame-generation settings, route and warm-up conditions
the same when comparing captures.

Each sampled CSV batch repeats the latest 300 base-frame intervals. Do not count
all 300 rows as new frames. Use `frametime_source_id` to deduplicate overlapping
history as described in [the CSV format](HudTelemetryFormat.md).
