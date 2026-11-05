/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

// The frame rate limiter shared by the runtime and the 32-bit bridge client.
// documentation/FrameRateLimit.md explains where each process waits and how the waits are scheduled.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

#include <windows.h>

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace dxvk {

  // The runtime posts this message to the bridge client with the limit in frames per second as wParam (0 is off).
  static constexpr const char* kFrameRateLimitMsgName = "UWM_REMIX_FRAME_RATE_LIMIT_MSG";

  // Holds the calling thread to a target frame rate and keeps statistics of the frame intervals it produces.
  // It sleeps on a high-resolution waitable timer and spins only for the last half millisecond.
  class FramePacer {
  public:
    FramePacer() {
      LARGE_INTEGER frequency;
      ::QueryPerformanceFrequency(&frequency);
      m_ticksPerSecond = frequency.QuadPart;

      m_timer = ::CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    }

    ~FramePacer() {
      if (m_timer != nullptr) {
        ::CloseHandle(m_timer);
      }
    }

    FramePacer(const FramePacer&) = delete;
    FramePacer& operator=(const FramePacer&) = delete;

    // Forgets the schedule and the statistics, so the next frame starts a new interval.
    void reset() {
      m_framesPerSecond = 0;
      m_deadline = 0;
      m_previousFrame = 0;
      resetStatistics();
    }

    // Returns once one frame interval has passed since the previous frame's deadline.
    // A frame that arrives late by at most the tolerance keeps the schedule, so the next frame makes up the small overshoot.
    // A frame that arrives later than that starts a new schedule from now, so no frame is shortened to catch up.
    void waitForFrameStart(uint32_t framesPerSecond) {
      if (!start(framesPerSecond)) {
        return;
      }

      const int64_t now = ticks();

      if (m_deadline == 0) {
        m_deadline = now + m_interval;
        m_previousFrame = now;
        return;
      }

      if (now < m_deadline) {
        sleepUntil(m_deadline);
        m_deadline += m_interval;
      } else if (now - m_deadline <= tolerance()) {
        m_deadline += m_interval;
        ++m_lateFrames;
      } else {
        m_deadline = now + m_interval;
        ++m_lateFrames;
      }

      endFrame();
    }

    // Returns once at least one frame interval, less the tolerance, has passed since the previous call returned.
    // It holds a frame that comes early and lets a late frame through at once.
    // It keeps no schedule, so a late frame is never followed by a shortened one.
    void holdFrame(uint32_t framesPerSecond) {
      if (!start(framesPerSecond)) {
        return;
      }

      if (m_previousFrame == 0) {
        m_previousFrame = ticks();
        return;
      }

      const int64_t earliest = m_previousFrame + m_interval - tolerance();

      if (ticks() < earliest) {
        sleepUntil(earliest);
      } else {
        ++m_lateFrames;
      }

      endFrame();
    }

    // Formats the statistics of the last reporting period into line, then starts a new period.
    // Returns false while the current period is still running.
    bool takeReport(const char* label, std::string& line) {
      if (!m_reportReady) {
        return false;
      }

      const uint32_t samples = (std::min)(m_frames, kMaxSamples);
      float p50 = 0.0f;
      float p99 = 0.0f;

      if (samples) {
        float* const first = m_intervalsMs;
        float* const last = m_intervalsMs + samples;
        std::nth_element(first, first + samples / 2, last);
        p50 = first[samples / 2];
        const uint32_t p99Index = (std::min)(samples - 1, samples * 99 / 100);
        std::nth_element(first, first + p99Index, last);
        p99 = first[p99Index];
      }

      char buffer[320];
      std::snprintf(buffer, sizeof(buffer),
                    "[FrameLimit] %s: target %.2f ms, %u frames, mean %.2f, p50 %.2f, p99 %.2f, max %.2f ms, "
                    "%u over +2 ms, %u late, max wake overshoot %.3f ms",
                    label, toMs(m_interval), m_frames, m_frames ? m_sumMs / m_frames : 0.0, p50, p99, m_maxMs,
                    m_overFrames, m_lateFrames, m_maxOvershootMs);
      line = buffer;

      resetStatistics();
      return true;
    }

  private:
    // Sized for a 5 second period at 1000 frames per second, the highest limit.
    static constexpr uint32_t kMaxSamples = 8192;
    static constexpr double kReportPeriodSeconds = 5.0;
    static constexpr float kOverThresholdMs = 2.0f;

    int64_t ticks() const {
      LARGE_INTEGER counter;
      ::QueryPerformanceCounter(&counter);
      return counter.QuadPart;
    }

    double toMs(int64_t ticks) const {
      return double(ticks) * 1000.0 / double(m_ticksPerSecond);
    }

    // A frame late by this much keeps the schedule in waitForFrameStart, and holdFrame lets frames come this much early.
    int64_t tolerance() const {
      return m_interval / 32;
    }

    // Applies the rate for this frame and returns whether the limiter is on.
    // A change of rate starts a new schedule and a new statistics period.
    bool start(uint32_t framesPerSecond) {
      if (framesPerSecond != m_framesPerSecond) {
        reset();
        m_framesPerSecond = framesPerSecond;
        m_interval = framesPerSecond ? m_ticksPerSecond / framesPerSecond : 0;
      }

      return framesPerSecond != 0;
    }

    void endFrame() {
      const int64_t now = ticks();
      const float intervalMs = float(toMs(now - m_previousFrame));
      m_previousFrame = now;

      if (m_frames < kMaxSamples) {
        m_intervalsMs[m_frames] = intervalMs;
      }

      ++m_frames;
      m_sumMs += intervalMs;
      m_maxMs = (std::max)(m_maxMs, intervalMs);

      if (intervalMs > float(toMs(m_interval)) + kOverThresholdMs) {
        ++m_overFrames;
      }

      if (m_periodStart == 0) {
        m_periodStart = now;
      } else if (toMs(now - m_periodStart) >= kReportPeriodSeconds * 1000.0) {
        m_reportReady = true;
      }
    }

    void resetStatistics() {
      m_frames = 0;
      m_sumMs = 0.0;
      m_maxMs = 0.0f;
      m_overFrames = 0;
      m_lateFrames = 0;
      m_maxOvershootMs = 0.0f;
      m_periodStart = 0;
      m_reportReady = false;
    }

    void sleepUntil(int64_t deadline) {
      // A thread woken at normal priority can wait behind the game's and the renderer's busy threads,
      // so the wait runs at raised priority and restores the previous priority afterwards.
      const HANDLE thread = ::GetCurrentThread();
      const int priority = ::GetThreadPriority(thread);
      const bool raisePriority = m_timer != nullptr && priority != THREAD_PRIORITY_ERROR_RETURN && priority < THREAD_PRIORITY_HIGHEST;

      if (raisePriority) {
        ::SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
      }

      // The timer can wake up to about half a millisecond late, so the last stretch is spent spinning.
      // The spin never yields, because a yield can hand the core to another thread for a whole scheduler quantum.
      const int64_t spinTicks = m_ticksPerSecond / 2000;
      const int64_t remaining = deadline - ticks();

      if (m_timer != nullptr && remaining > spinTicks) {
        // The due time is relative and counted in 100 ns units.
        LARGE_INTEGER dueTime;
        dueTime.QuadPart = -((remaining - spinTicks) * 10000000 / m_ticksPerSecond);

        if (dueTime.QuadPart < 0 && ::SetWaitableTimer(m_timer, &dueTime, 0, nullptr, nullptr, FALSE)) {
          ::WaitForSingleObject(m_timer, INFINITE);
        }
      }

      int64_t now = ticks();

      while (now < deadline) {
        ::YieldProcessor();
        now = ticks();
      }

      m_maxOvershootMs = (std::max)(m_maxOvershootMs, float(toMs(now - deadline)));

      if (raisePriority) {
        ::SetThreadPriority(thread, priority);
      }
    }

    // Without a high-resolution timer (Windows before 10 version 1803) this is null and the whole wait spins.
    HANDLE m_timer = nullptr;
    int64_t m_ticksPerSecond = 1;
    uint32_t m_framesPerSecond = 0;
    int64_t m_interval = 0;
    int64_t m_deadline = 0;
    int64_t m_previousFrame = 0;

    float m_intervalsMs[kMaxSamples] = {};
    uint32_t m_frames = 0;
    double m_sumMs = 0.0;
    float m_maxMs = 0.0f;
    uint32_t m_overFrames = 0;
    uint32_t m_lateFrames = 0;
    float m_maxOvershootMs = 0.0f;
    int64_t m_periodStart = 0;
    bool m_reportReady = false;
  };

}
