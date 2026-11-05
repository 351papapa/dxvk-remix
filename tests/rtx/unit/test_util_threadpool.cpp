/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include <cstring>
#include <random>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <thread>

#include "../../test_utils.h"
#include "../../../src/util/util_threadpool.h"
#include "../../../src/util/util_atomic_queue.h"
#include "../../../src/util/util_timer.h"
#include "../../../src/tracy/Tracy.hpp"

namespace dxvk {
  // Note: Logger needed by some shared code used in this Unit Test.
  Logger Logger::s_instance("test_util_threadpool.log");
}

using namespace dxvk;
using namespace std;
using namespace chrono;

inline uint64_t hwCounter() {
#if _M_ARM64
  return _ReadStatusReg(ARM64_SYSREG(3,3, 9,13,0));
#else
  return __rdtsc();
#endif
}

namespace {
  void check(bool condition, const char* what) {
    if (!condition) {
      throw DxvkError(str::format("Check failed: ", what));
    }
  }

  void waitFor(const std::atomic<uint32_t>& counter, uint32_t value) {
    while (counter.load() < value) {
      std::this_thread::yield();
    }
  }

  // A result large enough to show which task wrote it, like the D3D9 geometry results.
  struct TaggedResult {
    uint32_t tag = 0;
    uint32_t payload[15] = {};
  };

  TaggedResult makeResult(uint32_t tag) {
    TaggedResult result;
    result.tag = tag;
    for (uint32_t i = 0; i < 15; i++) {
      result.payload[i] = tag * 31 + i;
    }
    return result;
  }

  bool matches(const TaggedResult& result, uint32_t tag) {
    if (result.tag != tag) {
      return false;
    }
    for (uint32_t i = 0; i < 15; i++) {
      if (result.payload[i] != tag * 31 + i) {
        return false;
      }
    }
    return true;
  }
}

class ThreadPoolTestApp {
public:
  static void run() {
    cout << "Begin idle worker wake and shutdown tests" << endl;
    test_idleWake(true);
    test_idleWake(false);
    test_idleWake<false>(true);
    test_blockingPoolLifetime();
    cout << "Begin smoke test" << endl;
    test_smoke<0>();
    cout << "Begin task cancellation test" << endl;
    test_smoke<4>();
    cout << "Begin misc tests" << endl;
    test_misc();
    cout << "Begin held future slot test" << endl;
    test_heldFutureKeepsSlot();
    cout << "Begin dropped future test" << endl;
    test_droppedFuturesFreeSlots();
    cout << "Begin result lifetime test" << endl;
    test_resultLifetime();
    cout << "Begin move test" << endl;
    test_move();
    cout << "Begin lagging consumer test" << endl;
    test_laggingConsumer();
    cout << "Begin stress slot reuse test" << endl;
    test_stressSlotReuse();
    cout << "WorkerThreadPool successfully smoke tested" << endl;
  }

private:
  template<bool WorkStealing = true>
  static void test_idleWake(bool blockWhenIdle) {
    constexpr uint32_t rounds = 24, batch = 4;
    std::array<std::atomic<uint32_t>, rounds * batch> runs{};
    std::atomic<uint32_t> completed = 0;
    WorkerThreadPool<8, WorkStealing, false> pool(2, "geometry-processing", blockWhenIdle);
    check(pool.waitsForWork() == blockWhenIdle, "the pool uses the requested idle policy");

    // A single enqueue must suffice to wake a sleeping worker. Alternate singles with bursts
    // and reuse the same ring slots across idle periods.
    uint32_t scheduled = 0;
    for (uint32_t round = 0; round < rounds; ++round) {
      std::this_thread::sleep_for(milliseconds(2));
      const uint32_t count = round % 2 ? batch : 1;
      std::array<Future<TaggedResult>, batch> results;
      for (uint32_t i = 0; i < count; ++i) {
        results[i] = pool.Schedule([&, tag = scheduled + i] {
          ++runs[tag];
          ++completed;
          return makeResult(tag);
        });
        check(results[i].valid(), "a worker accepts work after an idle interval");
      }
      const auto deadline = steady_clock::now() + seconds(3);
      while (completed < scheduled + count && steady_clock::now() < deadline)
        std::this_thread::sleep_for(milliseconds(1));
      check(completed == scheduled + count, "workers wake without a second enqueue");
      for (uint32_t i = 0; i < count; ++i)
        check(matches(results[i].get(), scheduled + i), "each woken task returns its tagged result");
      scheduled += count;
    }
    for (uint32_t tag = 0; tag < scheduled; ++tag)
      check(runs[tag] == 1, "each scheduled task executes once");
    std::this_thread::sleep_for(milliseconds(2));
    // Destruction must also wake workers when there is no pending task.
  }

  static void test_blockingPoolLifetime() {
    auto token = std::make_shared<int>(42);
    std::atomic<uint32_t> cancelledRuns = 0, droppedRuns = 0;
    std::promise<void> started;
    auto ready = started.get_future();
    {
      WorkerThreadPool<16, true, false> pool(1, "geometry-processing", true);
      std::promise<void> release;
      auto gate = release.get_future().share();
      auto active = pool.Schedule([&started, gate] {
        started.set_value();
        gate.wait();
      });
      check(active.valid(), "the blocking pool accepts its first task");
      check(ready.wait_for(seconds(3)) == std::future_status::ready, "the worker starts its first task");

      auto cancelled = pool.Schedule([&, token] { ++cancelledRuns; });
      auto dropped = pool.Schedule([&, token] { ++droppedRuns; });
      auto tail = pool.Schedule([token] { return *token; });
      check(cancelled.valid() && dropped.valid() && tail.valid(), "the busy worker accepts queued tasks");
      cancelled.cancel();
      dropped.reset();
      release.set_value();
      check(tail.get() == 42 && droppedRuns == 1 && cancelledRuns == 0,
            "dropping a queued future preserves execution and cancelling it skips execution");
      active.get();

      // Outstanding captures must be released whether they finish before shutdown or are drained.
      for (uint32_t i = 0; i < 8; ++i) {
        auto pending = pool.Schedule([token] { return token; });
        check(pending.valid(), "the pool accepts unread results before shutdown");
      }
    }
    check(token.use_count() == 1, "pool shutdown releases all captures and unread results");
  }

  template<uint32_t cancelPeriod>
  static void test_smoke() {
    ZoneScoped;
    // We'll need to size the pool according to the number of tasks
    //  here since its a ring buffer and this is a single thread...
    const uint32_t numThreads = 8;
    const uint32_t numTasks = 2000;

    WorkerThreadPool<numTasks> threadPool(numThreads);
    check(!threadPool.waitsForWork(), "the default low-latency workers keep yielding");
    cout << "Created thread pool with " << numThreads << " threads" << endl;

    // Create params for lambda ahead of schedule
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<uint32_t> uni(0, 100);
    uint32_t a[numTasks], b[numTasks], c[numTasks], d[numTasks];

    for (uint32_t i = 0; i < numTasks; i++) {
      a[i] = uni(rng);
      b[i] = uni(rng);
      c[i] = uni(rng);
      d[i] = uni(rng);
    }

    vector<Future<uint32_t>> results(numTasks);
    {
      Timer t;
      const uint64_t s = hwCounter();

      for (uint32_t i = 0; i < numTasks; i++) {
        // Spawn N tasks that sleep a bit, and return a 1
        auto future = threadPool.Schedule([a = a[i], b = b[i], c = c[i], d = d[i]]()->uint32_t {
          ZoneScoped;
          // Create some varying workloads
          double sleepFor = (double) (a + b + c + d) / 4;
          auto start = high_resolution_clock::now();
          while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < sleepFor);

          return 1;
        });

        if (!future.valid()) {
          throw DxvkError("Failed to schedule task");
        }

        if (cancelPeriod > 0 && (i % cancelPeriod) == 0) {
          future.cancel();
        }

        results[i] = std::move(future);
      }

      const uint64_t e = hwCounter();

      cout << "Scheduled " << numTasks << " tasks in " << e - s << " clocks" << endl;
    }

    // Count all the return values (1's) and make sure everyone made it home
    uint32_t resultCount = 0;
    for (Future<uint32_t>& result : results) {
      if (result.valid()) {
        resultCount += result.get();
      }
    }

    const uint32_t expectedNumTasks = cancelPeriod > 0 ? (numTasks - numTasks / cancelPeriod) : numTasks;

    if (resultCount != expectedNumTasks) {
      throw DxvkError("Results didnt match");
    }

    // Check for ABA problem by making sure the correct count returned
    cout << "Counted the result, expected:" << expectedNumTasks << ", got:" << resultCount << endl;

    FrameMark;
  }

  static void test_misc() {
    const uint32_t numThreads = 4;
    const uint32_t numTasks = 32;

    auto* threadPool = new WorkerThreadPool<numTasks>(numThreads);
    cout << "Created thread pool with " << numThreads << " threads" << endl;

    uint32_t result = 0;
    auto future = threadPool->Schedule([&result]() {
      cout << "Hello from a void() future!" << endl;
      result += 1;
    });

    if (!future.valid()) {
      throw DxvkError("Failed to schedule task");
    }

    future.get();

    if (result != 1) {
      throw DxvkError("Result didnt match");
    }

    class DestuctorTester {
      // Using unique ptr to emulate a destructive move
      std::unique_ptr<uint32_t> param;
    public:
      DestuctorTester(DestuctorTester&& v) = default;
      DestuctorTester& operator =(DestuctorTester&&) = default;

      explicit DestuctorTester(uint32_t& param)
      : param { &param }
      { }

      ~DestuctorTester() {
        if (auto v = param.get()) {
          ++(*v);
          cout << "Hello from task destructor!" << endl;
          param.release();
        }
      }
    };

    DestuctorTester tester(result);
    future = threadPool->Schedule([tester = std::move(tester)]() {});

    if (!future.valid()) {
      throw DxvkError("Failed to schedule task");
    }

    // The lambda destructor is executed _after_ the future result is set.
    // We need to either wait for the result to update, or finalize the thread pool.
    // A Future must not outlive its pool, so it is released first.
    future.reset();
    delete threadPool;

    if (result != 2) {
      throw DxvkError("Result didnt match");
    }
  }

  // A slot whose Future is still held is not reused, even after its task ran.
  static void test_heldFutureKeepsSlot() {
    // One thread and four tasks per thread give a ring of four task slots.
    WorkerThreadPool<4> threadPool(1);
    std::atomic<uint32_t> executed = 0;

    std::vector<Future<TaggedResult>> held;
    for (uint32_t i = 0; i < 4; i++) {
      held.push_back(threadPool.Schedule([i, &executed]() {
        executed++;
        return makeResult(i);
      }));
      check(held.back().valid(), "a task is scheduled while slots are free");
      waitFor(executed, i + 1);
    }

    // Every task has run, so the worker queue is empty, but every slot is still owned by a Future.
    Future<TaggedResult> rejected = threadPool.Schedule([&executed]() {
      executed++;
      return makeResult(100);
    });
    check(!rejected.valid(), "Schedule rejects a task while the next slot's Future is held");

    Future<TaggedResult> inlined = threadPool.ScheduleOrRun([]() {
      return makeResult(101);
    });
    check(inlined.valid(), "ScheduleOrRun always returns a valid Future");
    check(threadPool.tasksRunByCaller() == 1, "ScheduleOrRun runs the task on the caller when no slot is free");
    check(matches(inlined.get(), 101), "a task run by the caller returns its own result");

    for (uint32_t i = 0; i < 4; i++) {
      check(matches(held[i].get(), i), "a held Future returns its own task's result");
    }

    // The slots are free again once their Futures were read.
    Future<TaggedResult> next = threadPool.Schedule([]() {
      return makeResult(102);
    });
    check(next.valid(), "a slot is reused after its Future was read");
    check(matches(next.get(), 102), "a reused slot returns the new task's result");
  }

  // A Future dropped without get() frees its slot once the task ran, and the task still runs.
  static void test_droppedFuturesFreeSlots() {
    WorkerThreadPool<4> threadPool(1);
    std::atomic<uint32_t> executed = 0;
    const uint32_t numTasks = 10000;

    for (uint32_t i = 0; i < numTasks; i++) {
      Future<TaggedResult> dropped = threadPool.ScheduleOrRun([i, &executed]() {
        executed++;
        return makeResult(i);
      });
      check(dropped.valid(), "ScheduleOrRun always returns a valid Future");
    }
    waitFor(executed, numTasks);
    check(executed.load() == numTasks, "every dropped task ran exactly once");

    // No slot leaked: three tasks fit in the worker queue and in the ring again. A worker frees a slot just after its
    // lambda returns, so each slot gets a second to become free.
    std::vector<Future<TaggedResult>> held;
    for (uint32_t i = 0; i < 3; i++) {
      const auto start = steady_clock::now();
      Future<TaggedResult> future;
      while (!future.valid() && steady_clock::now() - start < seconds(1)) {
        future = threadPool.Schedule([i]() { return makeResult(i); });
      }
      check(future.valid(), "dropped Futures leave no slot in use");
      held.push_back(std::move(future));
    }
    for (uint32_t i = 0; i < 3; i++) {
      check(matches(held[i].get(), i), "a slot freed by a dropped Future returns the new task's result");
    }
  }

  // A result nobody reads is destroyed, whichever of the worker and the Future finishes last.
  static void test_resultLifetime() {
    auto token = std::make_shared<uint32_t>(7);
    {
      WorkerThreadPool<64> threadPool(2);
      std::atomic<uint32_t> executed = 0;
      std::atomic<bool> go = false;

      // Dropped before the task runs: the worker destroys the result.
      for (uint32_t i = 0; i < 8; i++) {
        Future<std::shared_ptr<uint32_t>> dropped = threadPool.Schedule([token, &executed, &go]() {
          while (!go.load()) {
            std::this_thread::yield();
          }
          executed++;
          return token;
        });
        check(dropped.valid(), "a task is scheduled");
      }
      go = true;
      waitFor(executed, 8);

      // Dropped after the task ran: the Future destroys the result.
      std::vector<Future<std::shared_ptr<uint32_t>>> late;
      for (uint32_t i = 0; i < 8; i++) {
        late.push_back(threadPool.Schedule([token, &executed]() {
          executed++;
          return token;
        }));
      }
      waitFor(executed, 16);
      late.clear();

      // Read: the caller owns the result.
      Future<std::shared_ptr<uint32_t>> read = threadPool.Schedule([token]() { return token; });
      check(*read.get() == 7, "a read result is the task's result");

      // Destroying the pool joins the workers, so every lambda and result is destroyed after this scope.
    }
    check(token.use_count() == 1, "every unread result and every lambda was destroyed");
  }

  // A moved-from Future is invalid, so a result is read only once.
  static void test_move() {
    WorkerThreadPool<16> threadPool(1);
    Future<TaggedResult> first = threadPool.Schedule([]() { return makeResult(5); });
    Future<TaggedResult> second = std::move(first);
    check(!first.valid(), "a moved-from Future is invalid");
    check(second.valid(), "a moved-to Future is valid");

    Future<TaggedResult> third;
    third = std::move(second);
    check(!second.valid(), "a move-assigned-from Future is invalid");
    check(matches(third.get(), 5), "a moved Future returns its task's result");
    check(!third.valid(), "a read Future is invalid");

    Future<TaggedResult> ready = Future<TaggedResult>::ready(makeResult(6));
    Future<TaggedResult> movedReady = std::move(ready);
    check(!ready.valid(), "a moved-from ready Future is invalid");
    check(matches(movedReady.get(), 6), "a ready Future returns its value");
    check(!movedReady.valid(), "a read ready Future is invalid");

    // Assigning over a Future that owns a slot releases that slot.
    Future<TaggedResult> overwritten = threadPool.Schedule([]() { return makeResult(7); });
    overwritten = threadPool.Schedule([]() { return makeResult(8); });
    check(matches(overwritten.get(), 8), "an assigned Future returns the new task's result");
  }

  // The D3D9 pattern: a producer schedules three tasks per draw and queues the Futures for a consumer thread, which
  // falls behind by more draws than the pool has task slots. Every Future must return its own draw's result.
  static void test_laggingConsumer() {
    uint64_t runByCaller = 0;
    const uint32_t mismatches = laggingConsumer(20000, runByCaller);
    cout << "Lagging consumer: " << runByCaller << " of " << 20000 * 3 << " tasks ran on the producer" << endl;
    check(mismatches == 0, "every Future returned its own draw's result");
    check(runByCaller > 0, "the consumer fell behind by more draws than the pool has slots");
  }

  // The same pattern under stress mode: one slot per worker and random delays before each task.
  static void test_stressSlotReuse() {
    stress::Config config;
    config.delayUs = 200;
    config.delayChance = 0.5f;
    config.forceSlotReuse = true;
    config.seed = 7;
    stress::configure(config);
    uint64_t runByCaller = 0;
    const uint32_t mismatches = laggingConsumer(3000, runByCaller);
    const uint64_t delays = stress::delaysInjected();
    stress::configure(stress::Config());
    cout << "Stress: " << runByCaller << " of " << 3000 * 3 << " tasks ran on the producer, " << delays << " delays" << endl;
    check(mismatches == 0, "under stress, every Future returned its own draw's result");
    check(delays > 0, "stress mode delayed some tasks");
    check(runByCaller > 0, "stress mode left the pool fewer slots than draws in flight");
    check(!stress::active(), "a default configuration turns stress mode off");
  }

  // Returns the number of draws whose Futures returned another draw's results.
  static uint32_t laggingConsumer(const uint32_t numDraws, uint64_t& runByCaller) {
    struct Draw {
      uint32_t id = 0;
      Future<TaggedResult> hashes;
      Future<TaggedResult> bounds;
      Future<TaggedResult> skinning;
    };

    // Two threads and 64 tasks per thread give 128 task slots, and the queue holds up to 255 draws (765 Futures).
    WorkerThreadPool<64> threadPool(2);
    AtomicQueue<Draw, 256> draws;
    std::atomic<uint32_t> mismatches = 0;

    std::thread consumer([&]() {
      uint32_t expected = 0;
      Draw draw;
      while (expected < numDraws) {
        if (!draws.pop(draw)) {
          std::this_thread::yield();
          continue;
        }
        // The consumer stalls now and then, as a CS thread does when the GPU is shared or the game is in the background.
        if ((expected % 1000) == 0) {
          std::this_thread::sleep_for(milliseconds(5));
        }
        const uint32_t base = draw.id * 3;
        if (draw.id != expected ||
            !matches(draw.hashes.get(), base) ||
            !matches(draw.bounds.get(), base + 1) ||
            !matches(draw.skinning.get(), base + 2)) {
          mismatches++;
        }
        expected++;
      }
    });

    for (uint32_t i = 0; i < numDraws; i++) {
      const uint32_t base = i * 3;
      Draw draw;
      draw.id = i;
      draw.hashes = threadPool.ScheduleOrRun([base]() { return makeResult(base); });
      draw.bounds = threadPool.ScheduleOrRun([base]() { return makeResult(base + 1); });
      draw.skinning = threadPool.ScheduleOrRun([base]() { return makeResult(base + 2); });
      while (!draws.push(std::move(draw))) {
        std::this_thread::yield();
      }
    }
    consumer.join();

    runByCaller = threadPool.tasksRunByCaller();
    return mismatches.load();
  }
};

int main() {
  try {
    ThreadPoolTestApp::run();
  }
  catch (const dxvk::DxvkError& e) {
    cerr << e.message() << endl;
    return -1;
  }

  return 0;
}
