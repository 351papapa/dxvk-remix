// Adapted from xoxor4d/dxvk-remix, commit 7f5210e1a7cd1356bf0271230f82e6132b08ad8c.
// Modified for the runtime's rain-map subsystem. See ThirdPartyLicenses.txt for the original zlib/libpng license.
#pragma once

#include "rtx/pass/common_binding_indices.h"

#define RAIN_MAP_BINDING_HEIGHT_OUTPUT 130
#define RAIN_MAP_BINDING_HEIGHT_INPUT  131
#define RAIN_MAP_BINDING_MAP_OUTPUT    132

#if RAIN_MAP_BINDING_HEIGHT_OUTPUT <= COMMON_MAX_BINDING
#error "Rain map bindings overlap the common ray tracing bindings."
#endif

#if RAIN_MAP_BINDING_MAP_OUTPUT >= BINDING_ATMOSPHERE_MIN
#error "Rain map bindings overlap the sky resource bindings."
#endif
