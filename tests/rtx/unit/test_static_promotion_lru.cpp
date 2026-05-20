/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include "../../test_utils.h"
#include "../../../src/dxvk/rtx_render/rtx_fork_static_promotion.h"

namespace dxvk {
  // Note: Logger needed by some shared code used in this Unit Test.
  Logger Logger::s_instance("test_static_promotion_lru.log");
}

namespace dxvk {

  // The pool stores RtInstance* but never dereferences them in the routines
  // under test. We use opaque sentinel pointers as stand-ins.
  static RtInstance* makeFakeInstance(int id) {
    return reinterpret_cast<RtInstance*>(static_cast<intptr_t>(id) | 0x1000);
  }

  class TestApp {
  public:
    void testAddAndFind() {
      using namespace static_promotion;
      StaticPromotionPool pool;
      RtInstance* inst = makeFakeInstance(1);
      pool.addInstance(0xAAAA, inst, /*frame*/ 100);
      if (pool.findBucketFor(inst) == nullptr) {
        throw DxvkError("findBucketFor failed after addInstance");
      }
      if (pool.getBucketCount() != 1) {
        throw DxvkError("bucket count != 1 after single add");
      }
      if (pool.getTotalMembers() != 1) {
        throw DxvkError("total members != 1 after single add");
      }
    }

    void testRemoveCleansEmptyBucket() {
      using namespace static_promotion;
      StaticPromotionPool pool;
      RtInstance* inst = makeFakeInstance(2);
      pool.addInstance(0xBBBB, inst, /*frame*/ 1);
      pool.removeInstance(inst);
      if (pool.getBucketCount() != 0) {
        throw DxvkError("empty bucket was not garbage-collected");
      }
    }

    void testRekeyMovesInstance() {
      using namespace static_promotion;
      StaticPromotionPool pool;
      RtInstance* inst = makeFakeInstance(3);
      pool.addInstance(0xCCCC, inst, /*frame*/ 1);
      pool.addInstance(0xDDDD, inst, /*frame*/ 2);
      auto* bucket = pool.findBucketFor(inst);
      if (!bucket || bucket->bucketKeyHash != 0xDDDD) {
        throw DxvkError("rekey did not move instance to new bucket");
      }
      if (pool.getBucketCount() != 1) {
        throw DxvkError("rekey left the old bucket alive");
      }
    }

    void testLruEvictsOldest() {
      using namespace static_promotion;
      StaticPromotionPool pool;
      RtInstance* a = makeFakeInstance(10);
      RtInstance* b = makeFakeInstance(11);
      RtInstance* c = makeFakeInstance(12);
      pool.addInstance(0xA, a, /*frame*/ 100);
      pool.setBucketBytes(0xA, 50 * 1024 * 1024);
      pool.addInstance(0xB, b, /*frame*/ 200);
      pool.setBucketBytes(0xB, 50 * 1024 * 1024);
      pool.addInstance(0xC, c, /*frame*/ 300);
      pool.setBucketBytes(0xC, 50 * 1024 * 1024);

      const uint32_t evicted = pool.enforceMemoryBudget(120 * 1024 * 1024);

      if (evicted != 1) {
        throw DxvkError(str::format("expected 1 eviction, got ", evicted));
      }
      if (pool.findBucketFor(a) != nullptr) {
        throw DxvkError("LRU did not evict the oldest bucket (0xA)");
      }
      if (pool.findBucketFor(b) == nullptr || pool.findBucketFor(c) == nullptr) {
        throw DxvkError("LRU evicted the wrong bucket");
      }
    }

    void testTouchAllUpdatesRecency() {
      using namespace static_promotion;
      StaticPromotionPool pool;
      RtInstance* a = makeFakeInstance(20);
      pool.addInstance(0xE, a, /*frame*/ 100);
      pool.touchAll(/*frame*/ 500);
      auto* bucket = pool.findBucketFor(a);
      if (bucket->lastTouchedFrame != 500) {
        throw DxvkError("touchAll did not update lastTouchedFrame");
      }
    }

    void run() {
      testAddAndFind();
      testRemoveCleansEmptyBucket();
      testRekeyMovesInstance();
      testLruEvictsOldest();
      testTouchAllUpdatesRecency();
      std::cout << "All passed\n";
    }
  };
} // namespace dxvk


int main() {
  try {
    dxvk::TestApp testApp;
    testApp.run();
  }
  catch (const dxvk::DxvkError& error) {
    std::cerr << "FAIL: " << error.message() << std::endl;
    return 1;
  }
  std::cout << "PASS: static promotion LRU tests" << std::endl;
  return 0;
}
