# BLAS and opacity-micromap cache migration

The comparison baseline is Remix Plus commit `268fb5b240e35f283710ac6b3d6ccf2e8f2f14c7`. The old fork's implementation was read from the existing `dxvk-remix` checkout. Its NVIDIA MIT notices remain in the modified source files.

## BLAS repacking is supplied by stable clusters

The old `rtx.blasRepack.enabled=True` pass reconsidered compatible static cached buckets every 600 frames. It required more than 16 excess buckets and rejected groups above 4,096 instances or 500,000 primitives. It deliberately deferred while OMM rebinding was pending. It also changed the unchanged-scene path to touch only BLASes still referenced by the scene, allowing idle pooled structures to expire.

The new baseline already supplies the underlying behavior continuously in `AccelManager::mergeInstancesIntoBlas`:

- `clusterIndex` indexes clean static buckets by their complete `BlasBucketKey`.
- Routing a stable instance pulls every clean bucket with that key into the rebuild. This combines fragmented membership when a cluster changes rather than waiting 600 frames.
- `m_pendingClusterMerges` sorts members by `(key, instance ID)` before building buckets. The result depends on cluster membership, not which bucket happened to become dirty.
- `rtx.accel.clusterYoungFrames` defaults to 30 and `clusterFoldsPerFrame` to 2. Young and temporarily volatile buckets join their stable cluster after aging, with bounded work per frame. `clusterFoldDue` defeats the unchanged-scene early return when a fold is due.
- The key includes spatial cell, preserve state and volatility; `maxPrimsPerMergedBucket` retains the new primitive cap. Merging by the old, smaller key would discard these boundaries and can make rebuilds or ray traversal more expensive.
- The unchanged-scene branch already touches only cached merged and active dynamic BLASes. The old pool-touch workaround is therefore redundant.

These paths exist in the unmodified comparison baseline. The periodic repack implementation is not copied into the new runtime. The old key is retired and no longer controls runtime behavior. A migrated configuration may retain its original text for traceability, accompanied by an explicit comment identifying the replacement; it must not be presented as an implemented option. Keep the new cluster mechanism and its existing defaults. This is a structural equivalence of purpose, not a claim of identical bucket membership or measured frame time.

Relevant source locations are the `clusterFoldDue` fast-path condition, the young/volatile `foldKeys` loop, the `clusterIndex.equal_range` pull, and the `m_pendingClusterMerges` sort/fill in `src/dxvk/rtx_render/rtx_accel_manager.cpp`. Options are in `rtx_options.h` under `rtx.accel.clusterYoungFrames`, `clusterFoldsPerFrame`, and `rtx.maxPrimsPerMergedBucket`.

## Bounded idle OMM reclaim remains necessary

The new baseline's LRU pressure eviction runs only when memory is missing or the OMM budget shrinks. Its `requestTrim` path requires an explicit request and uses traced-frame age. Neither implements the old proactive 30-second sweep while there is spare budget.

The migration restores these config keys:

| Key | Runtime default | Migrated game value |
| --- | ---: | ---: |
| `rtx.opacityMicromap.cache.idleTimeoutSeconds` | 0 (off) | 30 |
| `rtx.opacityMicromap.cache.idleMaxEntriesPerSweep` | 32 | 32 |
| `rtx.opacityMicromap.cache.idleMaxMiBPerSweep` | 32 | 32 |
| `rtx.opacityMicromap.cache.idleDiagnostics` | False | False |

Controls are in the existing Opacity Micromap Cache panel. The timeout is the enable control. Any zero timeout, entry cap or memory cap disables proactive reclaim. These are preset-independent cache-lifetime controls and do not change image quality, mesh detail or shader sampling.

The sweep runs at most once per second, scans at most 256 LRU entries and advances a hash cursor between sweeps. It removes at most the configured count, with a soft byte cap allowing one oversized entry to finish. Only ready micromaps are eligible. Hardware OMM buffers, bake arrays, software stores and compaction resources still in use by the GPU are retained.

One frame timestamp is reused by actual hardware binds, software surface use and cached BLAS touches. Recently used entries receive an eight-traced-frame grace period, so resuming after a wall-clock pause cannot immediately discard live micromaps before the periodic refresh. In the unchanged-scene path, dynamic BLASes skip binding as well; when idle reclaim is enabled, the existing periodic touch helper also refreshes the micromaps held by active dynamic BLASes. It walks the BLAS reference lists, not every instance.

Destruction goes through the existing `destroyOmmData` method. That preserves the new hash-specific rebind notification, BLAS-held strong references and deferred software bindless-slot release. Pressure eviction and explicit trims remain in place. `rtx.ommHoldBoundMicromaps=True` and `rtx.opacityMicromap.cache.keepMergedBlasMicromapsInUse=True` retain their new defaults in the migrated preset.

## Validation boundary

On 2026-10-29, 38 standalone C++ cases passed using the production `reclaimIdle`, `touchBoundOmm`, and `touchMergedBlasMicromaps` methods extracted verbatim from the modified source. These cover disabled controls, time/count/byte boundaries, unfinished entries, all six GPU-resource guards, cursor continuation, 256-entry scan bounds, stale cursor recovery, frame-index wrap, diagnostics, stale BLAS references, dynamic fast-path protection and all eight resume frame phases.

These tests execute the production cache-selection and touch logic with fake cache/resource objects. They do not replace whole-runtime compilation, Vulkan lifetime validation or the requested game frame-time comparison. No GPU visibility, instance transforms or BLAS construction work was moved to the CPU by this change.
