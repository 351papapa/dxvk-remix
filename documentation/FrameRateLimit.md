# Frame Rate Limit

`rtx.frameRateLimit` caps the frame rate (0 is off). It is a user setting: the user menu shows it under Display Settings and saves it to `user.conf`. A change takes effect on the next frame.

## Where the wait happens

A cap keeps input latency low only when it holds the game back *before* the game samples input and simulates its next frame. A wait anywhere later lets finished frames queue up behind it, and every queued frame adds a frame interval of latency. `RtxReflex::limitFrameRate` picks one of three limiters each frame, and only one of them holds the game's next frame.

| Case | Limiter | Where the game is held |
|---|---|---|
| Runtime in the game process, Reflex initialized and `rtx.reflexMode` not None, and Reflex sleeps on this present path | Reflex | `NvLL_VK_Sleep` in `RtxReflex::sleep()`, with `minimumIntervalUs` = 1e6 / limit set by `RtxReflex::updateMode` |
| Runtime in the game process, any other case | Present thread | `FramePacer::waitForFrameStart` right after `RtxReflex::sleep()` in the D3D9 or D3D11 present |
| Runtime inside `NvRemixBridge.exe` (32-bit game) | Bridge client | `FramePacer::waitForFrameStart` in the bridge client's `Present`, in the game process |

The D3D9 swapchain always calls `sleep()`. The D3D11 swapchain calls it only when `rtx.reflexD3D11Markers` is on, so without it the present-thread limiter is used.

With the present-thread or the bridge-client limiter, the runtime also spaces its presents (see [Spacing the presents](#spacing-the-presents)). Reflex paces its own presents, so it gets no second wait.

## Why the bridge never uses Reflex for the cap

Under the bridge, the game runs in the 32-bit client process and the runtime runs in the 64-bit server process. Reflex sleeps on the server's main thread at the end of the server's `Present`, and its simulation markers measure the server's processing, not the game's.

The client's `Present` sends the frame to the server, then waits on the shared "Present" semaphore. The server releases that semaphore after its own `Present` returns. With `presentSemaphoreMaxFrames` at 1, the client may be one frame ahead, so it returns from `Present` for frame N once the server has finished frame N-1. A cap enforced on the server therefore lets the game sample input for frame N+1 a whole frame interval before that frame is rendered. A limiter that hooks the server's Vulkan present (an overlay frame limiter applied to `NvRemixBridge.exe`, or DXVK's `d3d9.maxFrameRate`) is worse, because the server's main thread also runs ahead of its own presents.

The bridge client waits after the semaphore and before it returns to the game. The server is then idle and up to date, and the game samples input right after the wait, so the frame it simulates is rendered at once.

The runtime posts the limit to the client over the Remix message channel (`kFrameRateLimitMsgName`, the limit in `wParam`) whenever it changes. The server does not set a Reflex interval in this case, so the game is never limited twice. Reflex still runs its low-latency sleep in the server when `rtx.reflexMode` asks for it.

## Holding the frame start

`FramePacer::waitForFrameStart` (`src/util/util_frame_pacer.h`) keeps a schedule of deadlines one interval apart and returns at each one.

- A frame that arrives early waits for its deadline.
- A frame that arrives late by no more than 1/32 of the interval keeps the schedule. The next frame is then shorter by that small overshoot, so timer jitter does not lower the average rate.
- A frame that arrives later than that starts a new schedule from the moment it arrives. No frame is shortened to catch up.

Catching up after a late frame keeps the average rate exactly at the cap, but it does so with a short frame right after the long one. On a frame time graph that pair is a spike up followed by a spike down, and on screen it is a stutter. Starting a new schedule gives a single long frame instead, at the cost of a rate slightly below the cap while the game cannot quite hold it.

## Spacing the presents

The frame start only fixes when the game begins a frame. The frame reaches the screen after the game has simulated it and the runtime has rendered it, and both take a different time each frame. Under the bridge the server also processes the frame asynchronously. So even a perfectly held frame start leaves the present times uneven.

`RtxReflex::pacePresent` evens them out. The D3D9 and D3D11 swapchains call it on the CS thread right before the present's command list is submitted, after the frame's rendering work has been submitted. It runs `FramePacer::holdFrame`, which holds a present until the interval, less 1/32 of it, has passed since the previous present, and lets a late present through at once.

- A frame that took less work than the one before it is held until its spacing is reached, instead of reaching the screen early.
- A frame that took more work goes through at once and shows as one long frame. The next frames may each be up to 1/32 of an interval short, so the presents drift back to the frame starts within a few frames.

The hold adds latency only to frames that come right after a slower one, and only by the difference in work. It never holds the game itself: the CS thread waits, while the game and, under the bridge, the server's main thread carry on with the next frame.

## Waiting precisely

Both waits sleep on a high-resolution waitable timer until half a millisecond before the deadline and spin with `YieldProcessor` for the rest. The spin does not yield the thread, because a yield can hand the core to another ready thread for a whole scheduler quantum. While it waits, the thread runs at `THREAD_PRIORITY_HIGHEST` so the busy game and renderer threads do not delay its wake-up, and it returns to its own priority afterwards. The timer has no tolerance set, so Windows does not coalesce it, and `timeBeginPeriod` does not affect a high-resolution timer. Without a high-resolution timer (Windows before 10 version 1803) the whole wait spins at the thread's own priority, which costs a CPU core but keeps the pacing.

Frame generation runs after the cap, so the displayed rate is a multiple of the cap.

## Measuring the pacing

Each wait logs a line once every 5 seconds while the cap is on:

```
[FrameLimit] bridge client wait: target 16.67 ms, 300 frames, mean 16.70, p50 16.67, p99 17.10, max 24.30 ms, 2 over +2 ms, 9 late, max wake overshoot 0.040 ms
[FrameLimit] present: target 16.67 ms, 300 frames, mean 16.70, p50 16.62, p99 17.40, max 24.90 ms, 2 over +2 ms, 41 late, max wake overshoot 0.035 ms
```

| Label | Process and log file | Measures |
|---|---|---|
| `bridge client wait` | Bridge client in the game process, `bridge32.log` | The time between frame starts, as the game sees them |
| `present thread wait` | Runtime in the game process, `remix-dxvk.log` | The same, for a game without the bridge |
| `present` | Runtime (in `NvRemixBridge.exe` under the bridge), `remix-dxvk.log` | The time between presents queued on the CS thread, which is what reaches the screen |

The intervals are measured from one return of the wait to the next. "over +2 ms" counts intervals more than 2 ms longer than the target. "late" counts frames that arrived after their deadline (or after their spacing, for `present`) and so did not wait. "max wake overshoot" is the furthest past its deadline that any wait returned, which shows whether the timer and the scheduler are on time.

- Spikes in `present` but a steady `bridge client wait`: the variation is in the game's simulation or in the server's processing and rendering. A frame whose work grew by more than the rest of its interval reaches the screen late.
- Spikes in `bridge client wait` with many late frames: the game or the server cannot hold the cap. A lower cap is smoother.
- A large wake overshoot: the waiting thread is not scheduled on time.
