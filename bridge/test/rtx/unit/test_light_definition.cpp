#include "util_remixapi_light_update.h"
#include <cstdio>
#include <cstdlib>

static unsigned checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
int main() {
  using namespace remixapi::light_update;
  remixapi_LightInfo info {};
  CHECK(!validDefinition(nullptr));
  CHECK(!validDefinition(&info));
  info.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO;
  CHECK(!validDefinition(&info));
  const remixapi_StructType types[] = {
    REMIXAPI_STRUCT_TYPE_LIGHT_INFO_SPHERE_EXT, REMIXAPI_STRUCT_TYPE_LIGHT_INFO_RECT_EXT,
    REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISK_EXT, REMIXAPI_STRUCT_TYPE_LIGHT_INFO_CYLINDER_EXT,
    REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISTANT_EXT, REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DOME_EXT,
    REMIXAPI_STRUCT_TYPE_LIGHT_INFO_USD_EXT };
  for (const auto type : types) {
    Header shape { type, nullptr };
    info.pNext = &shape;
    CHECK(validDefinition(&info));
    shape.pNext = &shape;
    CHECK(!validDefinition(&info));
    Header other { type, nullptr };
    shape.pNext = &other;
    CHECK(!validDefinition(&info));
  }
  Header unknown { REMIXAPI_STRUCT_TYPE_LIGHT_INFO, nullptr };
  info.pNext = &unknown;
  CHECK(!validDefinition(&info));
  CHECK(std::strcmp(serverQueryKey(versionKey), serverVersionKey) == 0);
  CHECK(std::strcmp(serverQueryKey(statusKey), serverStatusKey) == 0);
  CHECK(std::strcmp(versionKey, serverVersionKey) != 0);
  CHECK(std::strcmp(serverVersionKey, runtimeVersionKey) != 0);
  CHECK(std::strcmp(serverQueryKey("ordinary.key"), "ordinary.key") == 0);
  std::printf("light definition validation: %u checks passed\n", checks);
}
