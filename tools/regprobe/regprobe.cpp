// regprobe: create a Vulkan compute pipeline from a SPIR-V file and print the driver's executable statistics
// (register count, local memory = spills, ...). Descriptor layouts are reflected from the SPIR-V itself.
// Usage: regprobe <file.spv> [more.spv ...] | regprobe --rt <rgen.spv> [miss.spv] [chit.spv] | regprobe --driver
// Compute lines also print Workgroup=XxYxZ (SPIR-V LocalSize) and Driver=<major.minor>.
// --driver prints the driver and device only (tools/regprobe/regcheck.py keys its cache on it).
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <fstream>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, (int)r_); return false; } } while (0)

static std::vector<uint32_t> readFile(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint32_t> w(n / 4);
  f.read((char*)w.data(), n);
  return w;
}

struct Binding { uint32_t set, binding; VkDescriptorType type; uint32_t count; };

// Minimal SPIR-V reflection: descriptor bindings + push constant presence.
static bool reflect(const std::vector<uint32_t>& w, std::vector<Binding>& out, uint32_t& pushSize) {
  if (w.size() < 5 || w[0] != 0x07230203) return false;
  std::map<uint32_t, std::vector<uint32_t>> types;  // id -> instruction words
  std::map<uint32_t, uint32_t> constants;           // id -> value
  std::map<uint32_t, uint32_t> setOf, bindingOf;
  std::map<uint32_t, bool> isBlock, isBufferBlock;
  std::vector<std::pair<uint32_t, uint32_t>> vars;  // (resultType ptr id, var id) + storage class kept in types
  std::map<uint32_t, uint32_t> varStorage, varType;
  pushSize = 0;
  for (size_t i = 5; i < w.size();) {
    uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
    if (len == 0) break;
    const uint32_t* ins = &w[i];
    switch (op) {
      case 71: // OpDecorate
        if (ins[2] == 34) setOf[ins[1]] = ins[3];
        else if (ins[2] == 33) bindingOf[ins[1]] = ins[3];
        else if (ins[2] == 2) isBlock[ins[1]] = true;
        else if (ins[2] == 3) isBufferBlock[ins[1]] = true;
        break;
      case 43: // OpConstant
        constants[ins[2]] = ins[3];
        break;
      case 59: // OpVariable
        varType[ins[2]] = ins[1];
        varStorage[ins[2]] = ins[3];
        break;
      default:
        if ((op >= 19 && op <= 39) || op == 5341 /*AccelStructKHR*/ || op == 4472 /*RayQueryKHR*/)
          types[ins[1]] = std::vector<uint32_t>(ins, ins + len);
        break;
    }
    i += len;
  }
  for (auto& [var, ptrType] : varType) {
    uint32_t sc = varStorage[var];
    if (sc == 9) { pushSize = 256; continue; } // PushConstant
    if (!setOf.count(var) || !bindingOf.count(var)) continue;
    auto& ptr = types[ptrType];          // OpTypePointer: [op, id, sc, type]
    if (ptr.size() < 4) continue;
    uint32_t t = ptr[3], count = 1;
    for (;;) {
      auto& ti = types[t];
      uint32_t top = ti.empty() ? 0 : (ti[0] & 0xffff);
      if (top == 28) { count *= constants[ti[3]]; t = ti[2]; }        // OpTypeArray
      else if (top == 29) { count = 4096; t = ti[2]; }                // OpTypeRuntimeArray
      else break;
    }
    auto& ti = types[t];
    uint32_t top = ti.empty() ? 0 : (ti[0] & 0xffff);
    VkDescriptorType dt;
    if (top == 5341) dt = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    else if (top == 26) dt = VK_DESCRIPTOR_TYPE_SAMPLER;
    else if (top == 27) dt = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    else if (top == 25) { // OpTypeImage: [op, id, sampledType, dim, depth, arrayed, ms, sampled, format]
      bool buf = ti[3] == 5;
      bool storage = ti[7] == 2;
      dt = buf ? (storage ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
               : (storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    }
    else if (sc == 12) dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;          // StorageBuffer
    else if (sc == 2) dt = isBufferBlock[t] ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    else { fprintf(stderr, "  skip var %u (type op %u, sc %u)\n", var, top, sc); continue; }
    out.push_back({setOf[var], bindingOf[var], dt, count});
  }
  return true;
}

struct Ctx { VkInstance inst; VkPhysicalDevice phys; VkDevice dev; PFN_vkGetPipelineExecutablePropertiesKHR props; PFN_vkGetPipelineExecutableStatisticsKHR stats;
             uint32_t driverVersion; std::string deviceName; };

// NVIDIA packs the driver version as 10.8.8.6 bits (major.minor.secondary.tertiary).
static std::string driverString(uint32_t v) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%u.%02u", (v >> 22) & 0x3ffu, (v >> 14) & 0xffu);
  return buf;
}

static bool init(Ctx& c) {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  CHECK(vkCreateInstance(&ici, nullptr, &c.inst));
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(c.inst, &n, nullptr);
  std::vector<VkPhysicalDevice> pds(n);
  vkEnumeratePhysicalDevices(c.inst, &n, pds.data());
  c.phys = VK_NULL_HANDLE;
  for (auto p : pds) { VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(p, &pr); if (pr.vendorID == 0x10de) { c.phys = p; c.driverVersion = pr.driverVersion; c.deviceName = pr.deviceName; fprintf(stderr, "device: %s\n", pr.deviceName); break; } }
  if (!c.phys) { fprintf(stderr, "no NVIDIA device\n"); return false; }

  // Enable every feature the device supports (shaders use int8/16, fp16, scalar layout, ray query, ...)
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pe{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, &pe};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, &rq};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rtp};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR pf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR, &as};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT, &pf};
  VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &af};
  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
  VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
  vkGetPhysicalDeviceFeatures2(c.phys, &f2);
  if (!pe.pipelineExecutableInfo) { fprintf(stderr, "pipelineExecutableInfo unsupported\n"); return false; }
  rtp.rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE;
  rtp.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE;
  as.accelerationStructureCaptureReplay = VK_FALSE;
  v12.bufferDeviceAddressCaptureReplay = VK_FALSE;

  const char* exts[] = {
    VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME,
    VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
  };
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
  dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = sizeof(exts) / sizeof(exts[0]); dci.ppEnabledExtensionNames = exts;
  CHECK(vkCreateDevice(c.phys, &dci, nullptr, &c.dev));
  c.props = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(c.dev, "vkGetPipelineExecutablePropertiesKHR");
  c.stats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(c.dev, "vkGetPipelineExecutableStatisticsKHR");
  return true;
}

// Execution model of the first OpEntryPoint (5 = GLCompute, 5313 raygen, 5314 intersection, 5315 any-hit, 5316 closest-hit, 5317 miss)
static uint32_t executionModel(const std::vector<uint32_t>& w) {
  for (size_t i = 5; i < w.size();) {
    uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
    if (len == 0) break;
    if (op == 15) return w[i + 1];
    i += len;
  }
  return 0;
}

// Workgroup size from OpExecutionMode LocalSize (17) of the module; 0 when absent (ray tracing stages).
static void workgroupSize(const std::vector<uint32_t>& w, uint32_t (&wg)[3]) {
  wg[0] = wg[1] = wg[2] = 0;
  for (size_t i = 5; i < w.size();) {
    uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
    if (len == 0) break;
    if (op == 16 && len >= 6 && w[i + 2] == 17) { wg[0] = w[i + 3]; wg[1] = w[i + 4]; wg[2] = w[i + 5]; return; }
    i += len;
  }
}

static bool makeLayout(Ctx& c, const std::vector<std::vector<uint32_t>>& codes, VkPipelineLayout& pl, std::vector<VkDescriptorSetLayout>& layouts) {
  std::map<uint32_t, std::map<uint32_t, Binding>> merged;
  uint32_t pushSize = 0;
  for (auto& code : codes) {
    std::vector<Binding> binds;
    uint32_t ps = 0;
    if (!reflect(code, binds, ps)) return false;
    pushSize = std::max(pushSize, ps);
    for (auto& b : binds) merged[b.set][b.binding] = b;
  }
  std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> sets;
  std::map<uint32_t, std::vector<VkDescriptorBindingFlags>> flags;
  for (auto& [s, m] : merged) for (auto& [bi, b] : m) {
    VkDescriptorSetLayoutBinding lb{};
    lb.binding = b.binding; lb.descriptorType = b.type; lb.descriptorCount = b.count; lb.stageFlags = VK_SHADER_STAGE_ALL;
    sets[s].push_back(lb);
    flags[s].push_back(b.count == 4096 ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0);
  }
  uint32_t maxSet = sets.empty() ? 0 : sets.rbegin()->first + 1;
  layouts.resize(maxSet);
  for (uint32_t s = 0; s < maxSet; ++s) {
    VkDescriptorSetLayoutBindingFlagsCreateInfo fci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    fci.bindingCount = (uint32_t)flags[s].size(); fci.pBindingFlags = flags[s].data();
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &fci};
    lci.bindingCount = (uint32_t)sets[s].size(); lci.pBindings = sets[s].data();
    CHECK(vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &layouts[s]));
  }
  VkPushConstantRange pcr{VK_SHADER_STAGE_ALL, 0, pushSize};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = maxSet; plci.pSetLayouts = layouts.data();
  plci.pushConstantRangeCount = pushSize ? 1 : 0; plci.pPushConstantRanges = &pcr;
  CHECK(vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl));
  return true;
}

static void printStats(Ctx& c, VkPipeline pipe, const std::string& label) {
  VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR}; pi.pipeline = pipe;
  uint32_t ne = 0; c.props(c.dev, &pi, &ne, nullptr);
  std::vector<VkPipelineExecutablePropertiesKHR> eps(ne, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  c.props(c.dev, &pi, &ne, eps.data());
  if (ne == 0) printf("%s executables=0\n", label.c_str());
  for (uint32_t e = 0; e < ne; ++e) {
    VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR}; ei.pipeline = pipe; ei.executableIndex = e;
    uint32_t ns = 0; c.stats(c.dev, &ei, &ns, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    c.stats(c.dev, &ei, &ns, st.data());
    printf("%s [%s]", label.c_str(), eps[e].name);
    for (auto& s : st) {
      uint64_t v = s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR ? s.value.u64 : (uint64_t)s.value.i64;
      if (s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR) v = s.value.b32;
      printf(" %s=%llu", s.name, (unsigned long long)(v & 0xffffffffull));
    }
    printf("\n");
  }
}

// Ray tracing pipeline from raygen / miss / closest-hit modules (one group each, in the order given).
static bool probeRt(Ctx& c, const std::vector<const char*>& paths) {
  std::vector<std::vector<uint32_t>> codes;
  for (auto p : paths) codes.push_back(readFile(p));
  VkPipelineLayout pl; std::vector<VkDescriptorSetLayout> layouts;
  if (!makeLayout(c, codes, pl, layouts)) return false;
  std::vector<VkShaderModule> mods;
  std::vector<VkPipelineShaderStageCreateInfo> stages;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
  std::string label;
  for (size_t i = 0; i < codes.size(); ++i) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = codes[i].size() * 4; smci.pCode = codes[i].data();
    VkShaderModule sm; CHECK(vkCreateShaderModule(c.dev, &smci, nullptr, &sm)); mods.push_back(sm);
    uint32_t em = executionModel(codes[i]);
    VkShaderStageFlagBits stage = em == 5313 ? VK_SHADER_STAGE_RAYGEN_BIT_KHR : em == 5316 ? VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR :
                                  em == 5317 ? VK_SHADER_STAGE_MISS_BIT_KHR : em == 5315 ? VK_SHADER_STAGE_ANY_HIT_BIT_KHR : VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages.push_back({VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, stage, sm, "main"});
    VkRayTracingShaderGroupCreateInfoKHR g{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    g.generalShader = g.closestHitShader = g.anyHitShader = g.intersectionShader = VK_SHADER_UNUSED_KHR;
    if (stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) { g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR; g.closestHitShader = (uint32_t)i; }
    else { g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR; g.generalShader = (uint32_t)i; }
    groups.push_back(g);
    std::string n = paths[i]; n = n.substr(n.find_last_of("\\/") + 1);
    label += (label.empty() ? "" : "+") + n;
  }
  VkRayTracingPipelineCreateInfoKHR rci{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  rci.stageCount = (uint32_t)stages.size(); rci.pStages = stages.data();
  rci.groupCount = (uint32_t)groups.size(); rci.pGroups = groups.data();
  rci.maxPipelineRayRecursionDepth = 1;
  rci.layout = pl;
  auto create = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(c.dev, "vkCreateRayTracingPipelinesKHR");
  VkPipeline pipe;
  CHECK(create(c.dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rci, nullptr, &pipe));
  printStats(c, pipe, label);
  vkDestroyPipeline(c.dev, pipe, nullptr);
  for (auto m : mods) vkDestroyShaderModule(c.dev, m, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
  for (auto l : layouts) vkDestroyDescriptorSetLayout(c.dev, l, nullptr);
  return true;
}

static bool probe(Ctx& c, const char* path) {
  auto code = readFile(path);
  std::vector<Binding> binds;
  uint32_t pushSize = 0;
  if (!reflect(code, binds, pushSize)) { fprintf(stderr, "%s: not SPIR-V\n", path); return false; }
  if (executionModel(code) != 5) {
    std::vector<const char*> one{path};
    return probeRt(c, one);
  }

  std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> sets;
  std::map<uint32_t, std::vector<VkDescriptorBindingFlags>> flags;
  for (auto& b : binds) {
    VkDescriptorSetLayoutBinding lb{};
    lb.binding = b.binding; lb.descriptorType = b.type; lb.descriptorCount = b.count; lb.stageFlags = VK_SHADER_STAGE_ALL;
    sets[b.set].push_back(lb);
    flags[b.set].push_back(b.count == 4096 ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0);
  }
  uint32_t maxSet = sets.empty() ? 0 : sets.rbegin()->first + 1;
  std::vector<VkDescriptorSetLayout> layouts(maxSet);
  for (uint32_t s = 0; s < maxSet; ++s) {
    VkDescriptorSetLayoutBindingFlagsCreateInfo fci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    fci.bindingCount = (uint32_t)flags[s].size(); fci.pBindingFlags = flags[s].data();
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &fci};
    lci.bindingCount = (uint32_t)sets[s].size(); lci.pBindings = sets[s].data();
    CHECK(vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &layouts[s]));
  }
  VkPushConstantRange pcr{VK_SHADER_STAGE_ALL, 0, pushSize};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = maxSet; plci.pSetLayouts = layouts.data();
  plci.pushConstantRangeCount = pushSize ? 1 : 0; plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl));

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = code.size() * 4; smci.pCode = code.data();
  VkShaderModule sm;
  CHECK(vkCreateShaderModule(c.dev, &smci, nullptr, &sm));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"};
  cpci.layout = pl;
  VkPipeline pipe;
  CHECK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

  VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR}; pi.pipeline = pipe;
  uint32_t ne = 0; c.props(c.dev, &pi, &ne, nullptr);
  std::vector<VkPipelineExecutablePropertiesKHR> eps(ne, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  c.props(c.dev, &pi, &ne, eps.data());
  for (uint32_t e = 0; e < ne; ++e) {
    VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR}; ei.pipeline = pipe; ei.executableIndex = e;
    uint32_t ns = 0; c.stats(c.dev, &ei, &ns, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    c.stats(c.dev, &ei, &ns, st.data());
    printf("%s", path);
    for (auto& s : st) {
      uint64_t v = s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR ? s.value.u64 : (uint64_t)s.value.i64;
      if (s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR) v = s.value.b32;
      printf(" %s=%llu", s.name, (unsigned long long)(v & 0xffffffffull));
    }
    uint32_t wg[3];
    workgroupSize(code, wg);
    printf(" Workgroup=%ux%ux%u Driver=%s\n", wg[0], wg[1], wg[2], driverString(c.driverVersion).c_str());
  }
  vkDestroyPipeline(c.dev, pipe, nullptr);
  vkDestroyShaderModule(c.dev, sm, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
  for (auto l : layouts) vkDestroyDescriptorSetLayout(c.dev, l, nullptr);
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: regprobe <file.spv> ... | regprobe --rt <rgen.spv> [miss.spv] [chit.spv]\n"); return 2; }
  Ctx c{};
  if (!init(c)) return 1;
  if (std::string(argv[1]) == "--driver") {
    printf("driver=%s device=%s\n", driverString(c.driverVersion).c_str(), c.deviceName.c_str());
    vkDestroyDevice(c.dev, nullptr);
    vkDestroyInstance(c.inst, nullptr);
    return 0;
  }
  int rc = 0;
  if (std::string(argv[1]) == "--rt") {
    std::vector<const char*> paths(argv + 2, argv + argc);
    rc = probeRt(c, paths) ? 0 : 1;
  } else
  for (int i = 1; i < argc; ++i) if (!probe(c, argv[i])) rc = 1;
  vkDestroyDevice(c.dev, nullptr);
  vkDestroyInstance(c.inst, nullptr);
  return rc;
}
