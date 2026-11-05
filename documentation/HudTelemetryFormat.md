# HUD CSV format and synchronization

The header is:

```csv
frame,time_ms,kind,x,y,value
```

- `frame` is the HUD render counter. It identifies one sampled output batch.
- `time_ms` is elapsed steady-clock time since opening this output file.
- `kind` is `text`, `frametime_source_id` or `frametime_us`.
- For `text`, `x` and `y` are the same pixel coordinates used by HUD drawing.
- For `frametime_source_id`, `value` is the newest base-frame update ID.
- For `frametime_us`, `x` is the history index, 0-299 from oldest to newest, and
  `value` is the interval in microseconds. `y` is zero.

Values are CSV-quoted and embedded quotes are doubled. Numeric columns use the
classic locale. The renderer writes the source ID before its corresponding 300
history rows, all with the same `frame` and `time_ms`.

For newest source ID `S`, history slot `i` belongs to source frame `S - 299 + i`.
Skip uninitialized slots before the first base-frame update and skip IDs already
consumed. Repeated renders without a new update retain the same source ID and
history. If more than 300 base frames pass between samples, record the missing
interval as a gap rather than inventing measurements.

The frame-time update increments the source ID and changes its ring buffer under
one mutex. Rendering takes the ID, ring index and all samples under the same lock,
then releases it before formatting CSV or issuing draws. FPS and its displayed
frame-time string likewise use one snapshot, so both text fields describe the
same rolling estimate.

Export begins only on sampled visible HUD renders. The renderer flushes once at
the end of each sampled batch. Unsampled renders perform no writes or flushes;
a disabled export performs no file I/O. An open or write failure stops that output
until its path changes. These controls do not change HUD selection, generated-frame
counts or base-frame update timing.
