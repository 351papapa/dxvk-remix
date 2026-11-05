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

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../util/config/config.h"
#include "rtx_common_object.h"
#include "rtx_option.h"

namespace dxvk {

  class RtxContext;
  class RtxOptionLayer;
  class SweepValidationCapture;

  // Steps through a fixed table of rendering settings, one every few seconds, so that a Vulkan validation run
  // covers every major rendering path without anyone touching the menus, and each step's performance is measured.
  // The steps are option values in a temporary layer above the user and quality layers, so removing the layer
  // brings back the user's own values. The user and quality layers are also restored afterwards, because some
  // options write other options from their change callbacks.
  // A sweep starts only from the dev menu, so a saved setting can never start one.
  class RtxSettingsSweep : public CommonDeviceObject {
  public:
    RTX_OPTION_ARGS("rtx.sweep", float, stepSeconds, 10.0f,
                    "Seconds each settings sweep step is held and measured, counted over ray traced frames once the "
                    "step's shaders have finished compiling.",
                    args.minValue = 1.0f, args.maxValue = 3600.0f,
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.sweep", float, startDelaySeconds, 5.0f,
                    "Seconds the settings sweep waits after it is started from the menu, counted once the game shows a "
                    "ray traced scene, so the menu can be closed before the first step.",
                    args.minValue = 0.0f, args.maxValue = 3600.0f,
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.sweep", std::string, steps, std::string(""),
                    "Limits the settings sweep to the steps whose name contains one of these comma-separated words, "
                    "ignoring case, for example \"restir, preset\". Empty runs every step.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.sweep", bool, loop, false,
                    "Starts the settings sweep again after the last step instead of stopping. The user's settings are "
                    "restored between passes.",
                    args.flags = RtxOptionFlags::UserSetting);

    explicit RtxSettingsSweep(DxvkDevice* device);
    ~RtxSettingsSweep();

    // Starts a sweep at the next frame end. Does nothing while a sweep is active.
    void requestStart();

    // Stops the active sweep at the next frame end and restores the user's settings.
    void requestStop();

    // Returns true from the start request until the user's settings are restored.
    bool isActive() const;

    // Advances the sweep. Call once per frame on the CS thread, before RtxOptionManager::applyPendingValues.
    void onFrameEnd(RtxContext& ctx, bool cameraValid, bool rayTracedThisFrame);

    // Returns true while a step is measured, so GPU frame timestamps are taken.
    bool isMeasuring() const {
      return m_measuring;
    }

    // Takes the Remix GPU time of one timestamped frame. Call on the CS thread.
    void onGpuSample(uint32_t frameId, double measuredMs);

    // Returns false when the sweep has nothing to show on the HUD.
    bool getHudMessage(std::string& text, std::string& subText) const;

    // Returns one line describing the sweep's progress, for the dev menu.
    std::string getStatus() const;

  private:
    enum class State {
      Idle,
      WaitingForWorld,
      Running,
      Restoring
    };

    // Requirements a step needs from the device or the installed runtimes.
    enum Requirement : uint32_t {
      kNeedsNothing = 0,
      kNeedsSharc = 1 << 0,
      kNeedsNrc = 1 << 1,
      kNeedsDlss = 1 << 2,
      kNeedsRayReconstruction = 1 << 3,
      kNeedsDlfg = 1 << 4,
      kNeedsFsr = 1 << 5,
      kNeedsFsrFrameGen = 1 << 6,
      kNeedsOpacityMicromaps = 1 << 7,
    };

    struct Step {
      std::string group;
      std::string name;
      std::vector<std::pair<std::string, std::string>> overrides;
      uint32_t requirements = kNeedsNothing;
    };

    // A copy of one layer's values, kept as config text for restoring and as value strings for comparing.
    struct LayerSnapshot {
      const RtxOptionLayer* layer = nullptr;
      Config config;
      std::unordered_map<const RtxOptionImpl*, std::string> values;
    };

    using Clock = std::chrono::steady_clock;

    static std::vector<Step> buildStepTable();
    std::string checkRequirements(uint32_t requirements) const;
    void buildPlan();

    void openRun();
    void closeRun();
    void start(RtxContext& ctx);
    void applyStep(RtxContext& ctx);
    void advance(RtxContext& ctx, bool rayTracedThisFrame, double frameSeconds);
    void logResolvedOverrides();
    void writeStepPerf();
    void finishPass();
    void restore(RtxContext& ctx);
    void emitMarker(RtxContext& ctx, const std::string& text);

    static LayerSnapshot takeSnapshot(const RtxOptionLayer* layer);
    // Returns the number of options whose value in the layer differed from the snapshot.
    static uint32_t restoreSnapshot(const LayerSnapshot& snapshot);
    void releaseLayer();

    void setHud(const std::string& text, const std::string& subText, double seconds);

    std::atomic<bool> m_startRequested { false };
    std::atomic<bool> m_stopRequested { false };
    std::atomic<bool> m_active { false };

    State m_state = State::Idle;
    RtxOptionLayer* m_layer = nullptr;
    std::vector<Step> m_plan;
    std::vector<std::string> m_skipped;
    std::vector<LayerSnapshot> m_snapshots;
    size_t m_stepIndex = 0;
    uint32_t m_completedSteps = 0;
    uint32_t m_pass = 0;
    bool m_loopAfterRestore = false;

    Clock::time_point m_worldStart;
    bool m_worldSeen = false;
    Clock::time_point m_lastFrame;
    Clock::time_point m_stepApplied;
    bool m_stepReady = false;
    bool m_sawCompile = false;
    uint32_t m_peakCompiles = 0;
    uint32_t m_stableFrames = 0;
    double m_dwellSeconds = 0.0;

    // Performance of the step being measured.
    bool m_measuring = false;
    uint32_t m_measureStartFrameId = 0;
    double m_compileWaitSeconds = 0.0;
    std::vector<float> m_frameMs;
    std::vector<float> m_gpuMs;

    // Outputs of one sweep run, from the start request to the end of the restore.
    bool m_validationLayerActive = false;
    std::unique_ptr<SweepValidationCapture> m_capture;
    std::FILE* m_perfCsv = nullptr;

    uint32_t m_restoreFrame = 0;

    mutable std::mutex m_hudMutex;
    std::string m_hudText;
    std::string m_hudSubText;
    Clock::time_point m_hudUntil;
    std::string m_status = "Idle";
  };

}
