# NUMOS-derived shader portions: Copyright 2026 Kim2091 (Sparkles)
# SPDX-License-Identifier: Apache-2.0
# Remix Plus modifications add scene visibility and finite source-path transport.

"""Build host cloud shaders with scene visibility without changing the NUMOS submodule.

The generated overlay retains NUMOS's notices. Exact anchors make a NUMOS source
change fail the build for review instead of silently dropping the shadow hooks.
"""
import ast
import pathlib
import struct
import subprocess
import importlib.util
import sys


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'NUMOS scene hook needs review: {old[:100]!r}')
    return source.replace(old, new, 1)


def fnv1a(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return value


SCENE_VISIBILITY = r'''
// Remix Plus modification: scene shadows on direct cloud light and diffraction.
#include "numos/util/instance_definitions.h"
layout(binding = 30)
RaytracingAccelerationStructure cloudSceneTlas;

static float g_cloudSceneForwardNumeratorLog = -1e30f;
static float g_cloudSceneForwardDenominatorLog = -1e30f;
static float g_cloudSceneStepFraction = 0.5f;
static float g_cloudSceneTailTau = 0.0f;
static vec3 g_cloudSceneLastDirect = vec3(0.0f);
static vec3 g_cloudSceneLastAmbient = vec3(0.0f);

float cloudSceneSunVisibility(vec3 sampleKm, vec3 sunDirYUp, AtmosphereArgs args) {
  RayDesc ray;
  ray.Origin = args.cameraPosition + numosKmYUpToWorld(sampleKm - args.cameraWorldPosYUpKm, args);
  ray.Direction = normalize(atmosphereYUpToWorld(sunDirYUp, args.isZUp != 0u, args.flipUpAxis != 0u));
  ray.TMin = max(args.worldUnitsPerKm * 1e-7f, 1e-5f);
  ray.TMax = max(args.worldUnitsPerKm * 1000.0f, ray.TMin);
  static const uint kFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
    | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<kFlags> query;
  query.TraceRayInline(cloudSceneTlas, kFlags, OBJECT_MASK_OPAQUE, ray);
  query.Proceed();
  return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0f : 1.0f;
}

// Independent log sums keep the ratio valid even when absolute transmission
// is below float32 range but a high forward fraction still carries visible light.
float cloudSceneLogAdd(float a, float b) {
  const float hi = max(a, b);
  return hi + log(1.0f + exp(min(a, b) - hi));
}

void cloudSceneAccumulateForward(float deltaTau, float tauView, float tauSun,
                                 float visibility, AtmosphereArgs args) {
  if (args.cloudForwardPeakWeight <= 0.0f || deltaTau <= 0.0f) {
    return;
  }
  const float logWeight = log(deltaTau) + args.cloudForwardPeakWeight * tauView;
  g_cloudSceneForwardDenominatorLog = cloudSceneLogAdd(g_cloudSceneForwardDenominatorLog, logWeight);
  if (visibility > 0.0f) {
    const float logSource = logWeight - max(tauView + tauSun, 0.0f) + log(visibility);
    g_cloudSceneForwardNumeratorLog = cloudSceneLogAdd(g_cloudSceneForwardNumeratorLog, logSource);
  }
}

float cloudSceneAureoleScale(float visibleTau) {
  if (g_cloudSceneForwardDenominatorLog <= -1e29f || g_cloudSceneForwardNumeratorLog <= -1e29f) {
    return 0.0f;
  }
  // The complete homogeneous column has tauView + tauSun = visibleTau, giving 1.
  // A finite camera segment retains the attenuation between its end and the sun.
  return exp(min(visibleTau + g_cloudSceneForwardNumeratorLog - g_cloudSceneForwardDenominatorLog, 0.0f));
}
'''


SUN_OPTICAL_DEPTH = r'''
// Fresh incident depth for a forward sample skipped by the ordinary lighting LOD.
// This mirrors the direct light's sun path, without its sky/horizon light paths.
float cloudSceneSunOpticalDepth(vec3 samplePos, float typeLocal, float coverageLocal,
                                float tSampleKm, CloudShadeContext ctx, CloudSlab slab,
                                AtmosphereArgs args) {
  float dSun;
  float softUnused;
#ifdef CLOUD_SHADOW_MAP_AVAILABLE
  if (args.cloudShadowLive != 0u && slab.primary) {
    const CloudSunRay ray = cloudSunRay(samplePos, ctx, slab.altitudeKm, slab.thicknessKm, args);
#if CLOUD_SUN_BEER_MAP
    if (tSampleKm >= kCloudBeerMapNearKm) {
      dSun = cloudBeerMapRead(AtmosphereCloudBeerMap, cloudShadowLookup(samplePos, 0u, args), ray.w).x;
    } else
#endif
    {
      dSun = sampleCloudOpticalDepthLive(samplePos, ctx.sunDirYUp, 0, kCloudSunDetailKm,
                                         kCloudSunDetailTaps, kCloudSunDetailKm, 0, 0.0f,
                                         typeLocal, coverageLocal, ctx, slab, args, softUnused);
      dSun += cloudShadowMapOpticalDepth(samplePos, ray, kCloudSunDetailKm, args).x;
    }
  } else
#endif
  {
    const bool deck = !slab.primary;
    dSun = sampleCloudOpticalDepthLive(samplePos, ctx.sunDirYUp, deck ? 2 : 8, kCloudLightMarchMaxKm,
                                       deck ? 2 : kCloudSunDetailTaps, kCloudSunDetailKm,
                                       deck ? 2 : kCloudSunFarTaps, kCloudSunFarKm,
                                       typeLocal, coverageLocal, ctx, slab, args, softUnused) * slab.densityScale;
  }
  if (!slab.primary) {
    dSun += cloudPrimarySunOpticalDepth(samplePos, ctx, args).x;
  } else if (args.cloudLayer2Enable != 0u) {
    dSun += cloudSlabOpticalDepthFrom(samplePos, ctx.sunDirYUp, 4, 2,
                                      ctx, cloudSecondSlab(args), args).x;
  }
  return max(dSun * args.cloudDensity, 0.0f);
}
'''



def safe_spirv_opt_flags():
    # Read the established host recipe without executing its command-line parser.
    recipe = pathlib.Path(__file__).with_name('compile_shaders.py')
    module = ast.parse(recipe.read_text(encoding='utf-8'))
    matches = [node for node in module.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'spirvOptPasses'
                       for target in node.targets)]
    if len(matches) != 1:
        raise ValueError('Host SPIR-V optimization recipe changed; review the cloud shader build')
    passes = ast.literal_eval(matches[0].value)
    if 'combine-access-chains' in passes:
        raise ValueError('Cloud shader recipe must retain the host NonUniform access-chain workaround')
    return [f'--{item}' for item in passes] + [
        '--strip-debug', '--strip-nonsemantic', '--preserve-bindings', '--preserve-spec-constants',
        '--target-env=vulkan1.2', '--scalar-block-layout']


def specialization_constants(path):
    data = path.read_bytes()
    words = struct.unpack('<%dI' % (len(data) // 4), data)
    decorated, definitions, types = {}, {}, {}
    index = 5
    while index < len(words):
        count, opcode = words[index] >> 16, words[index] & 0xffff
        operands = words[index + 1:index + count]
        if opcode == 71 and operands[1] == 1:  # OpDecorate, SpecId
            decorated[operands[0]] = operands[2]
        elif opcode in (20, 21, 22):  # scalar bool, int, float
            types[operands[0]] = (opcode, *operands[1:])
        elif opcode in (48, 49, 50, 51, 52):
            definitions[operands[1]] = (opcode, operands[0], operands[2:])
        index += count
    result = []
    for ident, spec_id in sorted(decorated.items(), key=lambda item: item[1]):
        opcode, type_id, operands = definitions[ident]
        if opcode not in (48, 49, 50) or type_id not in types:
            raise ValueError('Cloud composite/expression specialization constant requires review')
        result.append((spec_id, types[type_id], opcode, operands))
    return result


def optimize_scene_shader(target, reflection, spirvopt, spirvval):
    before_local, before_push, before_bindings = reflection.reflect(target)
    before_specs = specialization_constants(target)
    optimized = target.with_suffix('.optimized.spv')
    subprocess.run([str(spirvopt), *safe_spirv_opt_flags(), str(target), '-o', str(optimized)], check=True)
    subprocess.run([str(spirvval), '--target-env', 'vulkan1.2', '--scalar-block-layout', str(optimized)], check=True)
    after_local, after_push, after_bindings = reflection.reflect(optimized)
    # OpName is debug information; every numerical descriptor property remains ABI.
    before_abi = (before_local, before_push, [item[:-1] for item in before_bindings])
    after_abi = (after_local, after_push, [item[:-1] for item in after_bindings])
    if before_abi != after_abi or before_specs != specialization_constants(optimized):
        raise ValueError(f'Cloud shader optimization changed ABI or specialization constants: {target}')
    optimized.replace(target)


def build_scene_shaders(numos, slangc, out, spirvopt=None, spirvval=None):
    if bool(spirvopt) != bool(spirvval):
        raise ValueError('Cloud shader optimization requires both spirv-opt and spirv-val')
    numos, slangc, out = map(pathlib.Path, (numos, slangc, out))
    overlay = out / 'scene_source'
    lib = overlay / 'numos' / 'lib'
    lib.mkdir(parents=True, exist_ok=True)
    source = (numos / 'shaders/numos/lib/cloud_march_common.slangh').read_text(encoding='utf-8')
    source = replace_once(source, '// SPDX-License-Identifier: Apache-2.0\n',
        '// SPDX-License-Identifier: Apache-2.0\n' + SCENE_VISIBILITY + '\n')
    source = replace_once(source, '// density only, no light, with the march\'s own density field (cloudMarchDensity, the same span and',
        '// density and incident light, with the march\'s own density field (cloudMarchDensity, the same span and')
    source = replace_once(source, '// Optical depth (args.cloudDensity applied) of one crossing from tFromKm on, density only:',
        '// Optical depth and incident forward transport over one crossing from tFromKm on:')
    source = replace_once(source, 'bool integrateCloudSample(float tSampleKm,',
        SUN_OPTICAL_DEPTH + '\nbool integrateCloudSample(float tSampleKm,')
    source = replace_once(source, '  vec3 surfaceLight;\n  if (refineLighting',
        '  const float sceneVisibility = cloudSceneSunVisibility(samplePos, ctx.sunDirYUp, args);\n'
        '  float forwardSunTau = 0.0f;\n'
        '  vec3 surfaceLight;\n  if (refineLighting')
    source = replace_once(source,
        '    surfaceLight = lighting.direct + lighting.ambient;\n    g_cloudLastLight = surfaceLight;',
        '    g_cloudSceneLastDirect = lighting.direct;\n'
        '    g_cloudSceneLastAmbient = lighting.ambient;\n'
        '    forwardSunTau = max(dSun * args.cloudDensity, 0.0f);\n'
        '    surfaceLight = lighting.direct * sceneVisibility + lighting.ambient;\n'
        '    g_cloudLastLight = surfaceLight;')
    source = replace_once(source, '    surfaceLight = g_cloudLastLight;',
        '    surfaceLight = g_cloudSceneLastDirect * sceneVisibility + g_cloudSceneLastAmbient;\n'
        '    if (args.cloudForwardPeakWeight > 0.0f && sceneVisibility > 0.0f) {\n'
        '      forwardSunTau = cloudSceneSunOpticalDepth(samplePos, typeLocal, coverageLocal,\n'
        '                                                 tSampleKm, ctx, slab, args);\n'
        '    }')
    source = replace_once(source, '  // Composite: cloudColor multiplies all surface-scatter terms.',
        '  const float forwardDeltaTau = effectiveDensity * segLenKm;\n'
        '  const float forwardViewTau = -log(max(viewTransmittance, 1e-30f))\n'
        '    + g_cloudSceneStepFraction * forwardDeltaTau;\n'
        '  cloudSceneAccumulateForward(forwardDeltaTau, forwardViewTau, forwardSunTau, sceneVisibility, args);\n\n'
        '  // Composite: cloudColor multiplies all surface-scatter terms.')
    source = replace_once(source, '      bool sampled = integrateCloudSample(t, stepLen, rayFrac, viewDirYUp, slab,',
        '      g_cloudSceneStepFraction = pixelJitter;\n'
        '      bool sampled = integrateCloudSample(t, stepLen, rayFrac, viewDirYUp, slab,')
    source = replace_once(source, '    integrateCloudSample(tSample, stepKm, rayFrac, viewDirYUp, slab,',
        '    g_cloudSceneStepFraction = hash;\n'
        '    integrateCloudSample(tSample, stepKm, rayFrac, viewDirYUp, slab,')
    source = replace_once(source, '        integrateCloudSample(tSample, tailStepKm, rayFrac, viewDirYUp, slab,',
        '        g_cloudSceneStepFraction = 0.5f;\n'
        '        integrateCloudSample(tSample, tailStepKm, rayFrac, viewDirYUp, slab,')
    source = replace_once(source, '  g_cloudLayer2Weight = 0.0f;\n  g_cloudExitTKm = -1.0f;',
        '  g_cloudLayer2Weight = 0.0f;\n  g_cloudExitTKm = -1.0f;\n'
        '  g_cloudSceneForwardNumeratorLog = -1e30f;\n'
        '  g_cloudSceneForwardDenominatorLog = -1e30f;')
    source = replace_once(source, 'static const uint kCloudAureoleRestTaps = 8u;',
        'static const uint kCloudAureoleRestTaps = 16u;')
    source = replace_once(source, '    const float lenKm = min(segKm, tExit - t);',
        '    const float lenKm = k + 1u == kCloudAureoleRestTaps ? tExit - t : min(segKm, tExit - t);')
    source = replace_once(source,
        '    tau += cloudMarchDensity(tSample, lenKm, rayFrac, viewDirYUp, slab,\n'
        '                             typeNoiseEntry, typeNoiseExit, coverageNoiseEntry, coverageNoiseExit,\n'
        '                             ctx, args, samplePos, hf, typeLocal, coverageLocal, sdfUnused)\n'
        '         * args.cloudDensity * lenKm;',
        '    const float deltaTau = cloudMarchDensity(tSample, lenKm, rayFrac, viewDirYUp, slab,\n'
        '                                              typeNoiseEntry, typeNoiseExit, coverageNoiseEntry, coverageNoiseExit,\n'
        '                                              ctx, args, samplePos, hf, typeLocal, coverageLocal, sdfUnused)\n'
        '                         * args.cloudDensity * lenKm;\n'
        '    if (deltaTau > 0.0f) {\n'
        '      const float visibility = cloudSceneSunVisibility(samplePos, ctx.sunDirYUp, args);\n'
        '      const float sunTau = visibility > 0.0f\n'
        '        ? cloudSceneSunOpticalDepth(samplePos, typeLocal, coverageLocal, tSample, ctx, slab, args) : 0.0f;\n'
        '      cloudSceneAccumulateForward(deltaTau, g_cloudSceneTailTau + tau + jitter * deltaTau, sunTau, visibility, args);\n'
        '    }\n'
        '    tau += deltaTau;')
    source = replace_once(source,
        '  if (g_cloudExitTKm >= 0.0f && dot(aureole, lumWeights) > 0.02f * dot(cloudRadiance, lumWeights)) {',
        '  // Forward transport needs the remainder even after the ordinary march became opaque.\n'
        '  if (g_cloudExitTKm >= 0.0f) {')
    source = replace_once(source,
        '    for (uint c = g_cloudExitCrossing; c < crossingCount && tau < 24.0f; ++c) {',
        '    for (uint c = g_cloudExitCrossing; c < crossingCount; ++c) {')
    source = replace_once(source,
        '      tau += cloudCrossingRestOpticalDepth(viewDirYUp, slabs[layer], int(crossing[c] & 1u), tMinClamp, tMaxClamp,',
        '      g_cloudSceneTailTau = tau;\n'
        '      tau += cloudCrossingRestOpticalDepth(viewDirYUp, slabs[layer], int(crossing[c] & 1u), tMinClamp, tMaxClamp,')
    source = replace_once(source, '  return min(aureole, vec3(6.0e4f));',
        '  return min(aureole, vec3(6.0e4f)) * cloudSceneAureoleScale(tau);')
    (lib / 'cloud_march_common.slangh').write_text(source, encoding='utf-8')
    entry = overlay / 'cloud_render.comp.slang'
    entry.write_bytes((numos / 'shaders/numos/passes/cloud_render.comp.slang').read_bytes())
    flags = ['-entry', 'main', '-target', 'spirv', '-zero-initialize', '-emit-spirv-directly',
             '-D__SLANG__', '-matrix-layout-column-major', '-Wno-30081', '-fvk-use-scalar-layout']
    spec = importlib.util.spec_from_file_location('numos_pass_reflection', numos / 'tools/gen_pass_table.py')
    reflection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reflection)
    for name, defines in [('cloud_render', []), ('cloud_render_fast', ['-DCLOUD_SUN_BEER_MAP=1'])]:
        original = out / (name + '.comp.spv')
        target = out / (name + '.comp.scene.spv')
        subprocess.run([str(slangc), str(entry), '-I', str(overlay), '-I', str(numos / 'shaders'),
                        *flags, *defines, '-o', str(target)], check=True)
        payload = target.read_bytes()
        if len(payload) % 4 or struct.unpack_from('<I', payload)[0] != 0x07230203:
            raise ValueError(f'Invalid scene SPIR-V: {target}')
        local, push, bindings = reflection.reflect(original)
        scene_local, scene_push, scene_bindings = reflection.reflect(target)
        expected = bindings + [(30, 0, 'AccelerationStructure', 'None', 0, 'cloudSceneTlas')]
        if (local, push, sorted(expected)) != (scene_local, scene_push, scene_bindings):
            raise ValueError(f'NUMOS scene shader ABI changed: {target}')
        if spirvopt:
            optimize_scene_shader(target, reflection, spirvopt, spirvval)
            payload = target.read_bytes()
        # Original ABI hash, then final payload hash; checked by the runtime before use.
        target.with_suffix('.meta').write_text(
            f'{fnv1a(original.read_bytes()):016x} {fnv1a(payload):016x}\n', encoding='ascii')
        print(f'Built scene visibility: {target.name}')


if __name__ == '__main__':
    build_scene_shaders(*sys.argv[1:])
