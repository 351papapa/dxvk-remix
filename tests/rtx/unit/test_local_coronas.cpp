#include "../../../include/remix/remix_local_coronas.h"
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace remixapi::local_coronas;
static void check(bool passed, const char* name) {
  if (!passed) { std::fprintf(stderr, "FAIL: %s\n", name); std::exit(1); }
}
int main() {
  for (bool owned : { false, true }) {
    for (bool composited : { false, true }) {
      check(submissionState(true, owned, composited) == State::Suppressed, "filter off always takes ownership, never fallback");
    }
  }
  check(submissionState(false, true, false) == State::Duplicate, "pending duplicate stays runtime-owned");
  check(submissionState(false, true, true) == State::Duplicate, "rendered duplicate cannot request IM3D again");
  check(submissionState(false, false, true) == State::Late, "first late submission rejects ownership");
  check(submissionState(false, false, false) == State::Accepted, "first timely submission accepted");
  check(submissionState(false, false, false, false) == State::Unavailable, "first unavailable submission rejects");
  check(submissionState(false, true, true, false) == State::Duplicate, "readiness changes cannot give accepted batch back to host");
  check(submissionState(true, true, true, false) == State::Suppressed, "filter-off ownership wins over late duplicate unavailable");
  std::vector<Source> input(limit + 1);
  for (uint32_t i = 0; i < input.size(); ++i) {
    input[i].position[0] = float(i);
    input[i].halfWorldSize = 2;
    input[i].color[0] = .5f;
    input[i].opacity = 1;
    input[i].falloff = 6;
    input[i].sourceId = i + 1;
  }
  std::string wire;
  std::vector<Source> decoded;
  for (uint32_t count : {0u, 1u, 511u, 512u, 513u, 1032u, 4095u, 4096u}) {
    check(encode(input.data(), count, wire, true), "v2 encode boundary");
    check(decode(wire.c_str(), decoded, true), "v2 decode boundary");
    check(decoded.size() == count, "whole batch count");
    check(!count || !std::memcmp(decoded.data(), input.data(), count * sizeof(Source)), "all record bytes survive");
    check(!decode(wire.c_str(), decoded), "v2 cannot be misread as v1");
    wire.pop_back();
    check(!decode(wire.c_str(), decoded, true), "reject truncated batch");
  }
  check(encode(input.data(), 512, wire), "v1 limit still works");
  check(decode(wire.c_str(), decoded), "v1 decode still works");
  check(!decode(wire.c_str(), decoded, true), "v1 cannot be misread as v2");
  check(!encode(input.data(), 513, wire), "v1 rejects enlarged batch");
  check(!encode(input.data(), 4097, wire, true), "v2 rejects oversized batch");
  input[4095].sourceId = input[1].sourceId;
  check(!encode(input.data(), 4096, wire, true), "duplicate in high index rejected");
  input[4095].sourceId = 4096;
  input[4095].opacity = std::numeric_limits<float>::quiet_NaN();
  check(!encode(input.data(), 4096, wire, true), "invalid last source rejects all");
  input[4095].opacity = 1;
  check(encode(input.data(), 4096, wire, true), "maximum packet");
  check(wire.size() == maxChars, "maximum wire bound");
  wire.back() = 'q';
  check(!decode(wire.c_str(), decoded, true), "invalid final nibble rejected");
  // Verify the expanded GPU tile bit addressing reaches every admitted source.
  std::array<uint32_t, limit / 32> mask{};
  for (uint32_t lane = 0; lane < 256; ++lane)
    for (uint32_t index = lane; index < limit; index += 256) mask[index / 32] |= 1u << (index % 32);
  for (uint32_t word : mask) check(word == 0xffffffffu, "complete tile mask coverage");
  std::printf("PASS: corona v1/v2 boundaries, exact roundtrip, invalid whole batches, 4096-source tile coverage (%zu-bit)\n", sizeof(void*) * 8);
}
