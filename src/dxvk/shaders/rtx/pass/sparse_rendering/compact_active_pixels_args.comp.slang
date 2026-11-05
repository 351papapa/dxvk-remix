// Writes the active-pixel list's indirect dispatch args from the entry count the compaction pass accumulated.
// The count is clamped to the list capacity first, so consumers never read past the entries the compaction pass wrote.

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/sparse_rendering/compact_active_pixels_binding_indices.h"
#include "rtx/pass/sparse_rendering/sparse_rendering.slangh"

// Inputs/Outputs

layout(binding = COMPACT_ACTIVE_PIXELS_BINDING_ACTIVE_PIXEL_LIST_INPUT_OUTPUT)
RWStructuredBuffer<uint> ActivePixelList;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
  const uint activePixelCount = min(ActivePixelList[COMPACT_ACTIVE_PIXELS_LIST_COUNT_SLOT], cb.sparseRenderingArgs.activePixelListCapacity);

  ActivePixelList[0] = (activePixelCount + COMPACT_ACTIVE_PIXELS_CONSUMER_GROUP_SIZE - 1) / COMPACT_ACTIVE_PIXELS_CONSUMER_GROUP_SIZE;
  ActivePixelList[1] = 1;
  ActivePixelList[2] = 1;
  ActivePixelList[COMPACT_ACTIVE_PIXELS_LIST_COUNT_SLOT] = activePixelCount;
}
