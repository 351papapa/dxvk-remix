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
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <new>
#include <optional>
#include <thread>
#include <vector>
#include <type_traits>
#include <utility>
#include <assert.h>
#include "util_atomic_queue.h"
#include "util_env.h"
#include "util_math.h"
#include "util_fastops.h"
#include "util_bit.h"
#include "util_stress.h"
#include "sync/sync_spinlock.h"

namespace dxvk {
  const size_t kLambdaStorageCapacity = 256;
  // Note: use up to 64 bytes for state
  const size_t kResultStorageCapacity = 256 - 64;

  using TaskId = uint32_t;

  // One slot of a WorkerThreadPool: a scheduled lambda and its result.
  // The worker that runs the lambda and the Future that owns the result each release the slot once. The slot is
  // free again only after both have released it, so a Future still held never reads a later task's result.
  struct Task {
    Task() = default;
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    // Stores the lambda. The slot must be free.
    template<typename LambdaType, typename ResultType>
    void capture(LambdaType&& lambda) {
      using Lambda = std::decay_t<LambdaType>;
      static_assert(sizeof(Lambda) <= kLambdaStorageCapacity, "Task object storage space overrun!");
      static_assert(alignof(Lambda) <= 64, "Task lambda alignment is too large!");
      if constexpr (!std::is_void_v<ResultType>) {
        static_assert(sizeof(ResultType) <= kResultStorageCapacity, "Result object storage space overrun!");
        static_assert(alignof(ResultType) <= 64, "Task result alignment is too large!");
      }
      assert(free());

      new (m_lambdaStorage.data()) Lambda(std::forward<LambdaType>(lambda));
      m_run = &run<Lambda, ResultType>;
      m_destroyResult = nullptr;
      m_cancelled.store(false, std::memory_order_relaxed);
      m_state.store(0, std::memory_order_relaxed);
    }

    // Runs the lambda on a worker, unless the task was cancelled, and releases the worker's side of the slot.
    void operator() () {
#ifdef _DEBUG
      if (!m_run) {
        throw DxvkError("Task thunk was not initialized!");
      }
#endif
      stress::delay(stress::Site::PoolTask);
      m_run(*this);
      m_run = nullptr;
      finish(kExecuted);
    }

    bool free() const {
      return m_state.load(std::memory_order_acquire) == kFree;
    }

    // Skips the lambda if it has not started yet.
    void cancel() {
      m_cancelled.store(true, std::memory_order_release);
    }

    // Waits for the lambda, then moves its result out and releases the owner's side of the slot.
    template<typename ResultType>
    ResultType take() {
      wait();
      if constexpr (std::is_void_v<ResultType>) {
        finish(kReleased);
      } else {
        assert(m_destroyResult != nullptr && "The task was cancelled and has no result.");
        ResultType result = std::move(*std::launder(reinterpret_cast<ResultType*>(m_resultStorage.data())));
        finish(kReleased);
        return result;
      }
    }

    // Releases the owner's side of the slot without reading the result. The lambda still runs unless it was cancelled.
    void release() {
      finish(kReleased);
    }

  private:
    // m_state holds kExecuted and kReleased while the task is in use, and only kFree once both sides released it.
    static constexpr uint32_t kExecuted = 1;
    static constexpr uint32_t kReleased = 2;
    static constexpr uint32_t kFree = 4;

    template<typename Lambda, typename ResultType>
    static void run(Task& task) {
      Lambda& lambda = *std::launder(reinterpret_cast<Lambda*>(task.m_lambdaStorage.data()));
      if (!task.m_cancelled.load(std::memory_order_acquire)) {
        if constexpr (std::is_void_v<ResultType>) {
          lambda();
        } else {
          new (task.m_resultStorage.data()) ResultType(lambda());
          task.m_destroyResult = &destroy<ResultType>;
        }
      }
      lambda.~Lambda();
    }

    template<typename ResultType>
    static void destroy(void* pResult) {
      std::launder(reinterpret_cast<ResultType*>(pResult))->~ResultType();
    }

    void wait() const {
      while ((m_state.load(std::memory_order_acquire) & kExecuted) == 0) {
        std::this_thread::yield();
      }
    }

    // The second side to release destroys a result nobody took and frees the slot.
    void finish(uint32_t side) {
      const uint32_t previous = m_state.fetch_or(side, std::memory_order_acq_rel);
      assert((previous & (side | kFree)) == 0 && "A task slot was released twice.");
      if ((previous | side) == (kExecuted | kReleased)) {
        if (m_destroyResult != nullptr) {
          m_destroyResult(m_resultStorage.data());
          m_destroyResult = nullptr;
        }
        m_state.store(kFree, std::memory_order_release);
      }
    }

    alignas(64) std::array<uint8_t, kLambdaStorageCapacity> m_lambdaStorage;
    alignas(64) std::array<uint8_t, kResultStorageCapacity> m_resultStorage;
    void (*m_run)(Task&) = nullptr;
    void (*m_destroyResult)(void*) = nullptr;
    std::atomic<uint32_t> m_state = kFree;
    std::atomic<bool> m_cancelled = false;
  };

  // The result of a scheduled task, or a value computed without a task. A Future owns its task slot until it is read,
  // reset or destroyed, so it is move-only.
  template<typename ResultType>
  class Future {
  public:
    Future() = default;

    explicit Future(Task& task)
    : m_pTask { &task } { }

    Future(Future&& other) noexcept
    : m_pTask { std::exchange(other.m_pTask, nullptr) }
    , m_value { std::move(other.m_value) } {
      other.m_value.reset();
    }

    Future& operator=(Future&& other) noexcept {
      if (this != &other) {
        reset();
        m_pTask = std::exchange(other.m_pTask, nullptr);
        m_value = std::move(other.m_value);
        other.m_value.reset();
      }
      return *this;
    }

    Future(const Future&) = delete;
    Future& operator=(const Future&) = delete;

    ~Future() {
      reset();
    }

    static Future ready(ResultType&& value) {
      Future future;
      future.m_value.emplace(std::move(value));
      return future;
    }

    // Waits for the result and moves it out. The Future is invalid afterwards.
    ResultType get() {
      assert(valid());
      if (m_value.has_value()) {
        ResultType value = std::move(*m_value);
        m_value.reset();
        return value;
      }
      return std::exchange(m_pTask, nullptr)->template take<ResultType>();
    }

    bool valid() const {
      return m_pTask != nullptr || m_value.has_value();
    }

    // Skips the task if it has not started, and drops the result.
    void cancel() {
      if (m_pTask != nullptr) {
        m_pTask->cancel();
      }
      reset();
    }

    // Drops the result. A scheduled task still runs, and its slot is reused after it has.
    void reset() {
      if (Task* pTask = std::exchange(m_pTask, nullptr)) {
        pTask->release();
      }
      m_value.reset();
    }

  private:
    Task* m_pTask = nullptr;
    std::optional<ResultType> m_value;
  };

  template<>
  class Future<void> {
  public:
    Future() = default;

    explicit Future(Task& task)
    : m_pTask { &task } { }

    Future(Future&& other) noexcept
    : m_pTask { std::exchange(other.m_pTask, nullptr) }
    , m_ready { std::exchange(other.m_ready, false) } { }

    Future& operator=(Future&& other) noexcept {
      if (this != &other) {
        reset();
        m_pTask = std::exchange(other.m_pTask, nullptr);
        m_ready = std::exchange(other.m_ready, false);
      }
      return *this;
    }

    Future(const Future&) = delete;
    Future& operator=(const Future&) = delete;

    ~Future() {
      reset();
    }

    static Future ready() {
      Future future;
      future.m_ready = true;
      return future;
    }

    void get() {
      assert(valid());
      if (m_ready) {
        m_ready = false;
        return;
      }
      std::exchange(m_pTask, nullptr)->take<void>();
    }

    bool valid() const {
      return m_pTask != nullptr || m_ready;
    }

    void cancel() {
      if (m_pTask != nullptr) {
        m_pTask->cancel();
      }
      reset();
    }

    void reset() {
      if (Task* pTask = std::exchange(m_pTask, nullptr)) {
        pTask->release();
      }
      m_ready = false;
    }

  private:
    Task* m_pTask = nullptr;
    bool m_ready = false;
  };

  /**
    * \brief Implements a async task scheduler, optimized
    *        for tasks of varying execution time using a
    *        work stealing algorithm.
    *
    *  NumThreads: How many threads to spawn (up to 255)
    *  NumTasksPerThread: Size of the task queue ring buffer
    *  WorkStealing: Enables the work stealing features of the scheduler
    *  LowLatency: Enables the low-latency mode where workers will spin instead of
    *              waiting for tasks on a conditional variable
    *  (ctor)workerName: Name given to threads with the pattern: workerName(N)
    *  (ctor)waitWhenIdle: A pool built with LowLatency=false can also select
    *                     yielding workers at construction. LowLatency=true always spins.
    *
    *  Example usage:
    *   // Creates 1 thread, and uses it to return PI via a future
    *   WorkerThreadPool threadPool(1, "thread-pool-name");
    *   Future<float> result = threadPool.Schedule([]{ return 3.14159265359f; });
    *   float pi = result.get();
    *
    *  Every Future must be destroyed before the pool.
    */
  template<size_t NumTasksPerThread, bool WorkStealing = true, bool LowLatency = true>
  class WorkerThreadPool {
    using Queue = AtomicQueue<TaskId, NumTasksPerThread>;
    using QueuePtr = std::unique_ptr<Queue>;

    struct Nop { };
    using OnAddCondition = std::conditional_t<LowLatency, Nop, dxvk::condition_variable>;
    using TaskMutex = std::conditional_t<LowLatency, Nop, dxvk::mutex>;

  public:
    WorkerThreadPool(uint8_t numThreads, const char* workerName = "Nameless Worker Thread", bool waitWhenIdle = !LowLatency)
    : m_numThread(std::clamp(numThreads, (uint8_t)1u, (uint8_t)dxvk::thread::hardware_concurrency())),
      m_waitWhenIdle(!LowLatency && waitWhenIdle) {
      assert(!LowLatency || !waitWhenIdle);
      // Note: round up to a closest power-of-two so we can use mask as modulo
      m_taskCount = 1 << (32 - bit::lzcnt(static_cast<uint32_t>(NumTasksPerThread * m_numThread) - 1));
      m_minTaskCount = std::min(m_taskCount, static_cast<uint32_t>(1u << (32 - bit::lzcnt(static_cast<uint32_t>(m_numThread) - 1))));
      m_tasks.reset(new Task[m_taskCount]);
      m_workerTasks.resize(m_numThread);
      m_workerThreads.resize(m_numThread);
      // Create the work queues first!  We need to create
      // then all since work stealing may access the other
      // queues.
      for (int i = 0; i < m_numThread; i++) {
        m_workerTasks[i] = std::make_unique<Queue>();
      }

      // Start the worker threads
      for (int i = 0; i < m_numThread; i++) {
        m_workerThreads[i] = std::thread([this, i, workerName] {
          env::setThreadName(str::format(workerName, "(", i, ")"));
          processWork(i);
        });
      }
    }

    ~WorkerThreadPool() {
      // Stop all the worker threads
      m_stopWork = true;

      if constexpr (!LowLatency) {
        std::unique_lock<TaskMutex> lock(m_taskMutex);
        m_condOnAdd.notify_all();
      }

      for (auto& worker : m_workerThreads) {
        worker.join();
      }

      if (m_numTasks > 0) {
        for (auto& workerTasks : m_workerTasks) {
          TaskId taskId;
          while (workerTasks->pop(taskId)) {
            // Cancel the actual task job
            m_tasks[taskId].cancel();
            // Execute the task to dispatch the destructor
            m_tasks[taskId]();
            --m_numTasks;
          }
        }
      }

      assert(m_numTasks == 0 && "Tasks left in thread pool queue after destruction!");
    }

    // Schedules a task to be executed by the thread pool.
    // Returns an invalid Future when the worker's queue is full, or when the next task slot is still owned by a Future
    // or still running.
    template <uint8_t Affinity = 0xFF, typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
    Future<R> Schedule(F&& f) {
      uint32_t thread;
      TaskId taskId;
      Task* pTask = reserve<Affinity>(thread, taskId);
      if (pTask == nullptr) {
        return Future<R>();
      }
      return submit<F, R>(*pTask, taskId, thread, std::forward<F>(f));
    }

    // Schedule, except that a task the pool cannot take runs on the calling thread, so it always returns a valid
    // Future. The caller does the work itself when the consumers of earlier results fall behind.
    template <uint8_t Affinity = 0xFF, typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
    Future<R> ScheduleOrRun(F&& f) {
      uint32_t thread;
      TaskId taskId;
      Task* pTask = reserve<Affinity>(thread, taskId);
      if (pTask != nullptr) {
        return submit<F, R>(*pTask, taskId, thread, std::forward<F>(f));
      }
      ++m_tasksRunByCaller;
      if constexpr (std::is_void_v<R>) {
        f();
        return Future<R>::ready();
      } else {
        return Future<R>::ready(f());
      }
    }

    // The number of tasks ScheduleOrRun ran on the calling thread.
    uint64_t tasksRunByCaller() const {
      return m_tasksRunByCaller;
    }

    bool waitsForWork() const { return !LowLatency && m_waitWhenIdle; }

  private:
    // Picks the worker and returns the next task slot, or nullptr when either is not available.
    template <uint8_t Affinity>
    Task* reserve(uint32_t& thread, TaskId& taskId) {
      // Is the affinity mask valid?
      const uint8_t affinityMask = std::min(popcnt_uint8(Affinity), m_numThread);

      // Schedule work on the appropriate thread
      thread = fast::findNthBit(Affinity, (uint8_t) (m_schedulerIndex++ % affinityMask));
      assert(thread < m_numThread);

      // A slot is reused only after its last Future released it and its worker finished with it. Slots are taken in
      // ring order, so a slot still in use blocks scheduling until it is free.
      taskId = m_taskId & slotMask();
      Task& task = m_tasks[taskId];
      if (m_workerTasks[thread]->isFull() || !task.free()) {
        return nullptr;
      }
      return &task;
    }

    template <typename F, typename R>
    Future<R> submit(Task& task, TaskId taskId, uint32_t thread, F&& f) {
      // Atomic queue is SPSC, so we don't need to take a lock here
      // since we know this will always be called from a single thread.
      ++m_taskId;
      assert(&m_tasks[taskId] == &task);

      // Capture task lambda
      task.capture<F, R>(std::forward<F>(f));

      // A worker may pop and decrement the count as soon as push publishes the task.
      ++m_numTasks;
      m_workerTasks[thread]->push(std::move(taskId));

      if constexpr (!LowLatency) {
        if (m_waitWhenIdle) {
          // Synchronize with the worker's predicate check before notifying. A lone
          // submission must wake a sleeper even when no further task is submitted.
          std::unique_lock<TaskMutex> lock(m_taskMutex);
          if constexpr (WorkStealing)
            m_condOnAdd.notify_one();
          else
            m_condOnAdd.notify_all();
        }
      }

      return Future<R>(task);
    }

    // The ring of slots in use: all of them, or under stress (rtx.debug.stress.forceSlotReuse) one per worker, so
    // a slot is reused as early as possible.
    uint32_t slotMask() const {
      if (stress::active() && stress::forceSlotReuse()) {
        return m_minTaskCount - 1;
      }
      return m_taskCount - 1;
    }

    void processWork(const uint32_t workerId) {
      while (true) {
        // Using a conditional wait in high-latency mode
        if constexpr (!LowLatency) {
          if (m_waitWhenIdle) {
            std::unique_lock<TaskMutex> lock(m_taskMutex);
            m_condOnAdd.wait(lock, [this] {
              return m_numTasks > 0 || m_stopWork.load();
            });
          }
        }

        // Master halt
        if (m_stopWork) {
          return;
        }

        // Try executing a task from our queue
        if (executeTask(workerId))
          continue;

        if (WorkStealing) {
          // There's no work to do!
          // Steal work from other queues
          bool workStolen = false;
          for (uint32_t i = 1; i < m_numThread; i++) {
            const uint32_t victim = (workerId + i) % m_numThread;
            if (executeTask(victim)) {
              workStolen = true;
              break;
            }
          }

          // If nothing to steal, yield this thread
          if (!workStolen && !waitsForWork()) {
            std::this_thread::yield();
          }
        }
      }
    }

    // True if front pop, False if back pop
    bool executeTask(const uint32_t workerId) {
      TaskId taskId;
      {
        // Since we're using an SPSC queue, we must take a lock when
        // popping, since we may be stealing (or be stolen from) by
        // another thread.
        std::unique_lock<sync::Spinlock> lock(m_threadMutex);

        if (!m_workerTasks[workerId]->pop(taskId)) {
          return false;
        }

        --m_numTasks;
      }

      // Execute the task
      m_tasks[taskId]();

      return true;
    }

    std::unique_ptr<Task[]> m_tasks;
    std::atomic<TaskId> m_taskId = 0;
    uint32_t m_taskCount;
    uint32_t m_minTaskCount;

    // Add the task to the queue and notify a worker thread
    //  just distribute evenly to all threads for some mask denoted by Affinity.
    size_t m_schedulerIndex = 0;

    uint8_t m_numThread;
    const bool m_waitWhenIdle;

    uint64_t m_tasksRunByCaller = 0;

    std::atomic<bool> m_stopWork = false;

    // Used conditionally to wait for tasks in high-latency mode
    TaskMutex m_taskMutex;
    OnAddCondition m_condOnAdd;

    // Used to synchronize intra-thread stealing
    sync::Spinlock m_threadMutex;

    std::vector<std::thread> m_workerThreads;

    // We expect high volume of potentially small tasks via "Schedule" per-
    //  frame, and require extremely low overhead to hit the 100's of FPS.
    // Use a lock-free circular queue here for two reasons (profiled):
    //  1. Non-circular queue incurs allocation overhead thats unacceptable
    //  2. Use of mutex, and CVs, incur overhead thats unacceptable
    std::vector<QueuePtr> m_workerTasks;
    std::atomic_uint32_t m_numTasks { 0 };
  };
} //dxvk
