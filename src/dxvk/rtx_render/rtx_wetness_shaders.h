#pragma once
#include "rtx_water.h"
#include "rtx/pass/wetness_args.h"

#define WETNESS_SHADER_CODE_INNER(code) RTX_WETNESS_SHADER_##code
#define WETNESS_SHADER_CODE(code) WETNESS_SHADER_CODE_INNER(code)
#define GET_WETNESS_ENABLED_SHADER(stage, className, code) GET_SHADER_VARIANT(stage, className, code)
#define GET_WETNESS_SHADER_VARIANT(stage, className, code) \
  (WetnessOptions::enabled() || WaterSystem::needsWetnessShader() \
    ? GET_WETNESS_ENABLED_SHADER(stage, className, WETNESS_SHADER_CODE(code)) \
    : GET_SHADER_VARIANT(stage, className, code))

#define WETNESS_SHADER_SOURCE(className, stage, code) \
public: \
  static const uint32_t* getStaticCodeData() { return code; } \
  static size_t getStaticCodeSize() { return sizeof(code); } \
  static const char* getName() { return #code; } \
  static const VkShaderStageFlagBits getStage() { return stage; } \
  static Rc<DxvkShader> getShader() { return GET_WETNESS_SHADER_VARIANT(stage, className, code); } \
  struct Ctor { Ctor() { if (getStage() == VK_SHADER_STAGE_COMPUTE_BIT) AutoShaderPipelinePrewarmer::registerComputeShaderForPrewarm([]{ return className::getShader(); }); }}; \
  static Ctor ctor;
