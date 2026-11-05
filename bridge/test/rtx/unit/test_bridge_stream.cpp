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

// Streams payloads of many sizes, most of them larger than the data rings, through the device bridge's real
// channel code: the x86 build is the client and starts the x64 build as the server. No GPU and no game.
//   test_bridge_stream_x86.exe <path to test_bridge_stream_x64.exe> <client ring bytes> <server ring bytes> [commands] [seed]
// The client checks every server response and the server checks every command. Exit code 0 means every byte
// arrived in order and the out-of-buffer message box was never reached (it would block the run until the
// caller's timeout).

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "config/config.h"
#include "config/global_options.h"
#include "util_bridge_state.h"
#include "util_devicecommand.h"
#include "util_filesys.h"
#include "util_guid.h"

bool gbBridgeRunning = true;
bridge_util::Guid gUniqueIdentifier;

namespace {
  enum Kind : uint32_t {
    kRows = 1,          // whole rows in blobs of at most one item (texture uploads)
    kBigItem = 2,       // one send_data item larger than the ring
    kStagedBlob = 3,    // one begin_data_blob larger than the ring
    kManyWords = 4,     // a run of single words larger than the ring
    kKeptPointer = 5,   // a small item whose pointer must outlive a large item read after it
    kReadbackRows = 6,  // a server response of whole rows larger than the server ring
    kReadbackItem = 7,  // a server response holding one item larger than the server ring
    kRandomItem = 8,    // a command with one item of a random size
    kTerminate = 9,
  };

  constexpr uint32_t kTrailer = 0xC0FFEE11u;

  uint8_t patternByte(const uint32_t seed, const uint64_t i) {
    const uint64_t x = (uint64_t(seed) * 0x9E3779B97F4A7C15ull) ^ (i * 0xBF58476D1CE4E5B9ull);
    return (uint8_t) ((x >> 29) ^ (x >> 47) ^ i);
  }

  void fillPattern(uint8_t* p, const size_t size, const uint32_t seed, const uint64_t first = 0) {
    for (size_t i = 0; i < size; ++i) {
      p[i] = patternByte(seed, first + i);
    }
  }

  bool checkPattern(const uint8_t* p, const size_t size, const uint32_t seed, const uint64_t first = 0) {
    for (size_t i = 0; i < size; ++i) {
      if (p[i] != patternByte(seed, first + i)) {
        printf("  byte %llu of seed %u differs\n", (unsigned long long) (first + i), seed);
        return false;
      }
    }
    return true;
  }

  int g_failures = 0;
  void fail(const char* what, const uint32_t seed) {
    ++g_failures;
    printf("FAIL: %s (seed %u)\n", what, seed);
    fflush(stdout);
  }
}

#if defined(REMIX_BRIDGE_CLIENT)

namespace {
  size_t g_bigBytes = 0;

  void sendRows(const uint32_t seed, const uint32_t rowSize, const uint32_t numRows, std::vector<uint8_t>& image) {
    image.resize(size_t(rowSize) * numRows);
    fillPattern(image.data(), image.size(), seed);
    ClientMessage c(Commands::Bridge_DebugMessage);
    c.send_many((uint32_t) kRows, seed, rowSize, numRows);
    const uint32_t rowsPerBlob = (uint32_t) std::max<size_t>(1, DeviceBridge::maxItemBytes() / rowSize);
    for (uint32_t row = 0; row < numRows;) {
      const uint32_t blobRows = std::min(rowsPerBlob, numRows - row);
      uint8_t* const pBlob = c.begin_data_blob(size_t(blobRows) * rowSize);
      memcpy(pBlob, image.data() + size_t(row) * rowSize, size_t(blobRows) * rowSize);
      c.end_data_blob();
      row += blobRows;
    }
    c.send_data(kTrailer);
  }

  void sendBigItem(const uint32_t seed, const size_t size, std::vector<uint8_t>& buf) {
    buf.resize(size);
    fillPattern(buf.data(), size, seed);
    ClientMessage c(Commands::Bridge_DebugMessage);
    c.send_many((uint32_t) kBigItem, seed, (uint32_t) size);
    c.send_data((uint32_t) size, buf.data());
    c.send_data(kTrailer);
  }

  void sendStagedBlob(const uint32_t seed, const size_t size) {
    ClientMessage c(Commands::Bridge_DebugMessage);
    c.send_many((uint32_t) kStagedBlob, seed, (uint32_t) size);
    uint8_t* const pBlob = c.begin_data_blob(size);
    fillPattern(pBlob, size, seed);
    c.end_data_blob();
    c.send_data(kTrailer);
  }

  void sendManyWords(const uint32_t seed, const uint32_t count) {
    ClientMessage c(Commands::Bridge_DebugMessage);
    c.send_many((uint32_t) kManyWords, seed, count);
    for (uint32_t i = 0; i < count; ++i) {
      c.send_data(seed * 2654435761u + i);
    }
    c.send_data(kTrailer);
  }

  void sendKeptPointer(const uint32_t seed, const size_t bigSize, std::vector<uint8_t>& buf) {
    uint8_t smallItem[37];
    fillPattern(smallItem, sizeof(smallItem), seed ^ 0x5555);
    buf.resize(bigSize);
    fillPattern(buf.data(), bigSize, seed);
    ClientMessage c(Commands::Bridge_DebugMessage);
    c.send_many((uint32_t) kKeptPointer, seed);
    c.send_data(sizeof(smallItem), smallItem);
    c.send_data((uint32_t) bigSize, buf.data());
    c.send_data(kTrailer);
  }

  void requestReadbackRows(const uint32_t seed, const uint32_t rowSize, const uint32_t numRows) {
    UID uid = 0;
    {
      ClientMessage c(Commands::Bridge_DebugMessage);
      uid = c.get_uid();
      c.send_many((uint32_t) kReadbackRows, seed, rowSize, numRows);
    }
    if (DeviceBridge::waitForCommand(Commands::Bridge_Response, 0, nullptr, true, uid) != Result::Success) {
      fail("no readback rows response", seed);
      return;
    }
    const size_t total = size_t(rowSize) * numRows;
    size_t received = 0;
    bool ok = true;
    while (received < total) {
      void* p = nullptr;
      const size_t n = DeviceBridge::get_data_view(&p);
      if (p == nullptr || n == 0 || n % rowSize != 0 || n > total - received) {
        fail("readback rows blob has a bad size", seed);
        ok = false;
        break;
      }
      ok = ok && checkPattern((const uint8_t*) p, n, seed, received);
      received += n;
    }
    if (DeviceBridge::get_data() != kTrailer) {
      ok = false;
    }
    DeviceBridge::pop_front();
    if (!ok) {
      fail("readback rows content", seed);
    }
  }

  void requestReadbackItem(const uint32_t seed, const size_t size) {
    UID uid = 0;
    {
      ClientMessage c(Commands::Bridge_DebugMessage);
      uid = c.get_uid();
      c.send_many((uint32_t) kReadbackItem, seed, (uint32_t) size);
    }
    if (DeviceBridge::waitForCommand(Commands::Bridge_Response, 0, nullptr, true, uid) != Result::Success) {
      fail("no readback item response", seed);
      return;
    }
    void* p = nullptr;
    const size_t n = DeviceBridge::get_data(&p);
    const bool ok = p != nullptr && n == size && checkPattern((const uint8_t*) p, n, seed) &&
                    DeviceBridge::get_data() == kTrailer;
    DeviceBridge::pop_front();
    if (!ok) {
      fail("readback item content", seed);
    }
  }

  void sendRandomItem(const uint32_t seed, const size_t size, std::vector<uint8_t>& buf) {
    buf.resize(std::max<size_t>(size, 1));
    fillPattern(buf.data(), size, seed);
    ClientMessage c(Commands::Bridge_DebugMessage);
    c.send_many((uint32_t) kRandomItem, seed, (uint32_t) size);
    c.send_data((uint32_t) size, buf.data());
    c.send_data(kTrailer);
  }

  void sendStateRecords(const uint32_t seed, const uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
      StateMessage c(Commands::IDirect3DDevice9Ex_SetRenderState, 0x1234);
      c.send_many(seed, i);
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    printf("usage: %s <server exe> <client ring bytes> <server ring bytes> [commands] [seed]\n", argv[0]);
    return 2;
  }
  const std::string serverExe = argv[1];
  const uint32_t clientRing = (uint32_t) strtoul(argv[2], nullptr, 0);
  const uint32_t serverRing = (uint32_t) strtoul(argv[3], nullptr, 0);
  const uint32_t numCommands = argc > 4 ? (uint32_t) strtoul(argv[4], nullptr, 0) : 400;
  const uint32_t rngSeed = argc > 5 ? (uint32_t) strtoul(argv[5], nullptr, 0) : 12345;

  bridge_util::Config::init(bridge_util::Config::App::Client);
  GlobalOptions::init();
  // STREAM_TEST_LOGDIR names a directory for bridge32.log and bridge64.log.
  if (const char* logDir = getenv("STREAM_TEST_LOGDIR")) {
    dxvk::util::RtxFileSys::init(logDir);
    Logger::init();
  }
  BridgeState::setServerState(BridgeState::ProcessState::Running);
  DeviceBridge::init("Device", clientRing, 3000, 3000, serverRing, 10, 25);
  DeviceBridge::setStateBatching(true);

  // The server maps the same channels by the GUID.
  std::string cmd = "\"" + serverExe + "\" " + gUniqueIdentifier.toString() + " " + std::to_string(clientRing) + " " +
                    std::to_string(serverRing);
  STARTUPINFOA si = { sizeof(si) };
  PROCESS_INFORMATION pi = {};
  if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
    printf("could not start the server: %u\n", GetLastError());
    return 2;
  }

  const size_t maxItem = DeviceBridge::maxItemBytes();
  const size_t ringBytes = DeviceBridge::getWriterChannel().data->get_total_size() * sizeof(uint32_t);
  const size_t serverRingBytes = serverRing;
  g_bigBytes = std::min<size_t>(ringBytes * 5 / 2 + 7, 200u << 20);
  printf("client ring %zu bytes, server ring %u bytes, item %zu bytes, big %zu bytes\n",
         ringBytes, serverRing, maxItem, g_bigBytes);
  fflush(stdout);

  std::vector<uint8_t> buf;
  std::mt19937 rng(rngSeed);
  uint32_t seed = 1;

  // Fixed cases first: every path once with a payload larger than its ring.
  sendRows(seed++, 4096 * 4, (uint32_t) (g_bigBytes / (4096 * 4)) + 1, buf);
  sendRows(seed++, 12345, (uint32_t) (g_bigBytes / 12345) + 3, buf);
  sendRows(seed++, (uint32_t) maxItem + 13, 5, buf);  // rows larger than one item
  sendBigItem(seed++, g_bigBytes, buf);
  sendStagedBlob(seed++, g_bigBytes - 3);
  sendManyWords(seed++, (uint32_t) (ringBytes / 4) * 2 + 11);
  sendKeptPointer(seed++, g_bigBytes + 1, buf);
  requestReadbackRows(seed++, 3840 * 4, (uint32_t) (serverRingBytes * 3 / (3840 * 4)) + 1);
  requestReadbackItem(seed++, serverRingBytes * 2 + 5);
  sendStateRecords(seed++, 70000);

  // Then random commands of every kind and size, so batches end and wrap at many different ring positions.
  for (uint32_t i = 0; i < numCommands && gbBridgeRunning; ++i) {
    const uint32_t s = seed++;
    switch (rng() % 9) {
    case 0: sendRows(s, 1 + rng() % 70000, 1 + rng() % 300, buf); break;
    case 1: sendBigItem(s, 1 + rng() % (maxItem * 3), buf); break;
    case 2: sendStagedBlob(s, 1 + rng() % (maxItem * 3)); break;
    case 3: sendManyWords(s, rng() % 70000); break;
    case 4: sendKeptPointer(s, 1 + rng() % (maxItem * 3), buf); break;
    case 5: requestReadbackRows(s, 1 + rng() % 20000, 1 + rng() % 200); break;
    case 6: requestReadbackItem(s, 1 + rng() % (maxItem * 3)); break;
    case 7: sendStateRecords(s, rng() % 5000); break;
    default: sendRandomItem(s, rng() % 4096, buf); break;
    }
  }
  {
    ClientMessage c(Commands::Bridge_DebugMessage);
    c.send_many((uint32_t) kTerminate, seed);
  }

  if (WaitForSingleObject(pi.hProcess, 600000) != WAIT_OBJECT_0) {
    printf("FAIL: the server did not finish\n");
    TerminateProcess(pi.hProcess, 3);
    return 1;
  }
  DWORD serverExit = 1;
  GetExitCodeProcess(pi.hProcess, &serverExit);
  printf("client: %d failures, bridge %s, server exit code %u\n", g_failures, gbBridgeRunning ? "running" : "stopped",
         serverExit);
  return (g_failures == 0 && gbBridgeRunning && serverExit == 0) ? 0 : 1;
}

#elif defined(REMIX_BRIDGE_SERVER)

namespace {
  // Some commands are read slowly, so the client fills the ring and waits for space.
  void maybeStall(std::mt19937& rng) {
    if (rng() % 8 == 0) {
      Sleep(rng() % 3);
    }
  }

  bool expectTrailer() {
    return DeviceBridge::get_data() == kTrailer;
  }

  void handle(const Header& header, std::mt19937& rng, bool& done, uint32_t& stateSeq) {
    const uint32_t uid = DeviceBridge::get_data();
    if (header.command == Commands::Bridge_StatePacket) {
      while (gbBridgeRunning && DeviceBridge::get_data_pos() != header.dataOffset) {
        const uint32_t id = DeviceBridge::get_data();
        const uint32_t seed = DeviceBridge::get_data();
        const uint32_t i = DeviceBridge::get_data();
        if (id != Commands::IDirect3DDevice9Ex_SetRenderState || header.pHandle != 0x1234) {
          fail("state record", seed);
          return;
        }
        if (i != 0 && i != stateSeq + 1) {
          fail("state record order", seed);
        }
        stateSeq = i;
      }
      return;
    }
    const uint32_t kind = DeviceBridge::get_data();
    const uint32_t seed = DeviceBridge::get_data();
    switch (kind) {
    case kRows:
    {
      const uint32_t rowSize = DeviceBridge::get_data();
      const uint32_t numRows = DeviceBridge::get_data();
      const uint64_t total = uint64_t(rowSize) * numRows;
      uint64_t received = 0;
      while (received < total) {
        void* p = nullptr;
        const size_t n = DeviceBridge::get_data_view(&p);
        if (p == nullptr || n == 0 || n % rowSize != 0 || n > total - received) {
          fail("rows blob size", seed);
          return;
        }
        if (!checkPattern((const uint8_t*) p, n, seed, received)) {
          fail("rows content", seed);
        }
        received += n;
        maybeStall(rng);
      }
      if (!expectTrailer()) {
        fail("rows trailer", seed);
      }
      break;
    }
    case kBigItem:
    case kStagedBlob:
    case kRandomItem:
    {
      const uint32_t size = DeviceBridge::get_data();
      void* p = nullptr;
      const uint32_t n = DeviceBridge::get_data(&p);
      if (n != size || (size != 0 && (p == nullptr || !checkPattern((const uint8_t*) p, n, seed)))) {
        fail("item content", seed);
      }
      if (!expectTrailer()) {
        fail("item trailer", seed);
      }
      maybeStall(rng);
      break;
    }
    case kManyWords:
    {
      const uint32_t count = DeviceBridge::get_data();
      for (uint32_t i = 0; i < count; ++i) {
        if (DeviceBridge::get_data() != seed * 2654435761u + i) {
          fail("word content", seed);
          return;
        }
        if ((i & 0xFFFF) == 0) {
          maybeStall(rng);
        }
      }
      if (!expectTrailer()) {
        fail("words trailer", seed);
      }
      break;
    }
    case kKeptPointer:
    {
      void* pSmall = nullptr;
      const uint32_t smallSize = DeviceBridge::get_data(&pSmall);
      void* pBig = nullptr;
      const uint32_t bigSize = DeviceBridge::get_data(&pBig);
      maybeStall(rng);
      if (pSmall == nullptr || smallSize != 37 || !checkPattern((const uint8_t*) pSmall, smallSize, seed ^ 0x5555)) {
        fail("kept pointer content", seed);
      }
      if (pBig == nullptr || !checkPattern((const uint8_t*) pBig, bigSize, seed)) {
        fail("kept pointer big item", seed);
      }
      if (!expectTrailer()) {
        fail("kept pointer trailer", seed);
      }
      break;
    }
    case kReadbackRows:
    {
      const uint32_t rowSize = DeviceBridge::get_data();
      const uint32_t numRows = DeviceBridge::get_data();
      std::vector<uint8_t> row(rowSize);
      ServerMessage c(Commands::Bridge_Response, uid);
      const uint32_t rowsPerBlob = (uint32_t) std::max<size_t>(1, DeviceBridge::maxItemBytes() / rowSize);
      for (uint32_t r = 0; r < numRows;) {
        const uint32_t blobRows = std::min(rowsPerBlob, numRows - r);
        uint8_t* const pBlob = c.begin_data_blob(size_t(blobRows) * rowSize);
        fillPattern(pBlob, size_t(blobRows) * rowSize, seed, uint64_t(r) * rowSize);
        c.end_data_blob();
        r += blobRows;
      }
      c.send_data(kTrailer);
      break;
    }
    case kReadbackItem:
    {
      const uint32_t size = DeviceBridge::get_data();
      std::vector<uint8_t> item(size);
      fillPattern(item.data(), size, seed);
      ServerMessage c(Commands::Bridge_Response, uid);
      c.send_data(size, item.data());
      c.send_data(kTrailer);
      break;
    }
    case kTerminate:
      done = true;
      break;
    default:
      fail("unknown kind", seed);
      break;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    return 2;
  }
  std::wstring guid(argv[1], argv[1] + strlen(argv[1]));
  LPWSTR pGuid = guid.data();
  gUniqueIdentifier.setGuid(&pGuid);
  const uint32_t clientRing = (uint32_t) strtoul(argv[2], nullptr, 0);
  const uint32_t serverRing = (uint32_t) strtoul(argv[3], nullptr, 0);

  bridge_util::Config::init(bridge_util::Config::App::Server);
  GlobalOptions::init();
  // STREAM_TEST_LOGDIR names a directory for bridge32.log and bridge64.log.
  if (const char* logDir = getenv("STREAM_TEST_LOGDIR")) {
    dxvk::util::RtxFileSys::init(logDir);
    Logger::init();
  }
  DeviceBridge::init("Device", serverRing, 10, 25, clientRing, 3000, 3000);

  std::mt19937 rng(777);
  bool done = false;
  uint32_t commands = 0;
  uint32_t stateSeq = 0;
  while (!done && gbBridgeRunning && DeviceBridge::waitForCommand() == Result::Success) {
    const Header header = DeviceBridge::pop_front();
    DeviceBridge::beginReadCommand(header, false);
    handle(header, rng, done, stateSeq);
    if (!DeviceBridge::endReadCommand()) {
      fail("data out of step after a command", 0);
      break;
    }
    DeviceBridge::releaseReadData();
    ++commands;
  }
  printf("server: %u commands, %d failures, bridge %s\n", commands, g_failures, gbBridgeRunning ? "running" : "stopped");
  fflush(stdout);
  return (done && g_failures == 0 && gbBridgeRunning) ? 0 : 1;
}

#endif
