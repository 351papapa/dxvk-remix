# Copyright (c) 2026 Remix Plus contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate host-owned, ABI-compatible NUMOS haze overlays and three authenticated sidecars.

emit-overlay NUMOS OVERLAY prepares headers before the host composite is compiled.
build NUMOS SLANGC OUTPUT [--originals DIR] [--overlay DIR]
      [--spirv-opt TOOL --spirv-val TOOL] builds the complete compatible pass family.

The dependency is read-only. Modified copies retain its copyright and carry an explicit notice.
The host composite must search OVERLAY before NUMOS/shaders. Its dual decoder also reads stock
17-word lists, allowing the runtime to fall back to the entire original pass family atomically.
"""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import struct
import subprocess
import time

from numos_scene_shaders import fnv1a, optimize_scene_shader, replace_once, specialization_constants


SOURCE_HASHES = {
    'lib/aerial_perspective_lights.slangh': 'c42d983c7b9f32081bb3813952df8dffdccae80d428a30546bfd8783e2af6f69',
    'passes/aerial_perspective_light_cull.comp.slang': '3ead6e44a7c8a25e9530378f9e511afbc5b455eb4309e11a7bf6eecd342d28ec',
    'passes/aerial_perspective_froxel_lighting.comp.slang': '4e9b224c23303277de88bfadaac6502daeed6c8c4db09b5bf52d6d3282023a98',
}
LAYOUT_VERSION = 1
NOTICE = ('// Modified by Remix Plus (2026): complete grouped bitset light lists, shared dual-format\n'
          '// iteration and conservative cone rejection. Original NUMOS sampling and shadow estimator retained.\n')


def write_if_changed(path, source):
    """Avoid recompiling every host shader when the generated include is byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = source.encode('utf-8')
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)


def checked_source(numos, relative):
    source = (numos / 'shaders/numos' / relative).read_text(encoding='utf-8')
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    if digest != SOURCE_HASHES[relative]:
        raise ValueError(f'NUMOS haze source changed; review overlay anchors and bounds: {relative} ({digest})')
    return replace_once(source, '// SPDX-License-Identifier: Apache-2.0\n',
                        '// SPDX-License-Identifier: Apache-2.0\n' + NOTICE)



def split_haze_tap_geometry(source):
    # Keep the original function for the froxel passes. Only the per-pixel airlight orders
    # its pure geometry before medium evaluation, skipping exactly the original zero branches.
    start = source.index('vec3 aerialPerspectiveLightRadiance(')
    end = source.index('\n}\n', start) + 2
    radiance = source[start:end]
    split = radiance.index('  const ApPhase phase =')
    geometry = radiance[radiance.index('{') + 1:split]
    if geometry.count('return vec3(0.0f);') != 2:
        raise ValueError('NUMOS light zero-contribution branches changed; review the tap split')
    geometry = geometry.replace('return vec3(0.0f);', 'return false;')
    geometry = ('// Exact original geometry, evaluated before medium only to skip its proven-zero branches.\n'
        'struct ApHazeTapGeometry {\n  vec3 directionToLight;\n  float solidAngle;\n  float shaping;\n  float window;\n};\n\n'
        'bool apHazeEvaluateTapGeometry(AerialPerspectiveLight light, vec3 worldPosition, out ApHazeTapGeometry geometry) {\n'
        '  geometry.directionToLight = vec3(0.0f);\n'
        '  geometry.solidAngle = 0.0f;\n'
        '  geometry.shaping = 0.0f;\n'
        '  geometry.window = 0.0f;\n' + geometry +
        '  geometry.directionToLight = directionToLight;\n'
        '  geometry.solidAngle = solidAngle;\n'
        '  geometry.shaping = shaping;\n'
        '  geometry.window = window;\n'
        '  return true;\n}\n\n')
    finish = radiance[split:]
    for name in ['directionToLight', 'solidAngle', 'shaping', 'window']:
        finish = re.sub(r'\b' + name + r'\b', 'geometry.' + name, finish)
    finish = ('vec3 apHazeFinishTapRadiance(AerialPerspectiveLight light, ApHazeTapGeometry geometry,\n'
              '                              vec3 viewDirWorld, ApPhaseMix phaseMix, AtmosphereArgs args) {\n' + finish + '\n\n')
    source = replace_once(source, 'static const uint kApAirlightTaps = 8u;',
                          geometry + finish + 'static const uint kApAirlightTaps = 8u;')
    return replace_once(source,
        '      const ApMediumSample m = apSampleMedium(p, args);\n'
        '      const vec3 radiance = aerialPerspectiveLightRadiance(light, p, dirWorld, apPhaseMixAt(m), args);',
        '      ApHazeTapGeometry geometry;\n'
        '      if (!apHazeEvaluateTapGeometry(light, p, geometry)) {\n'
        '        continue;\n'
        '      }\n'
        '      const ApMediumSample m = apSampleMedium(p, args);\n'
        '      const vec3 radiance = apHazeFinishTapRadiance(light, geometry, dirWorld, apPhaseMixAt(m), args);')


def add_finite_spot_beam(source, clip_source):
    # The last float of the packed light is zero for ordinary lights. A positive value selects a
    # finite spot beam and supplies its minimum scattering coefficient in inverse kilometres.
    source = replace_once(source,
        'bool apHazeEvaluateTapGeometry(AerialPerspectiveLight light, vec3 worldPosition, out ApHazeTapGeometry geometry)',
        'bool apHazeEvaluateTapGeometry(AerialPerspectiveLight light, vec3 worldPosition, bool narrowBeam, out ApHazeTapGeometry geometry)')
    source = replace_once(source,
        '  const float solidAngle = 2.0f * kPi * (1.0f - sqrt(max(0.0f, 1.0f - sinThetaSq)));\n\n'
        '  geometry.directionToLight = directionToLight;',
        '  const float cosTheta = sqrt(max(0.0f, 1.0f - sinThetaSq));\n'
        "  // Rationalize the beam's solid angle so a distant small emitter does not round to zero.\n"
        '  const float solidAngle = 2.0f * kPi * (narrowBeam ? sinThetaSq / (1.0f + cosTheta) : 1.0f - cosTheta);\n\n'
        '  geometry.directionToLight = directionToLight;')
    source = replace_once(source, 'vec3 apLocalLightAirlight(', clip_source + '\nvec3 apLocalLightAirlight(')
    source = replace_once(source,
        '    const float a = max(startWorld, t0 - light.influenceRadius);\n'
        '    const float b = min(endWorld, t0 + light.influenceRadius);',
        '    float a = max(startWorld, t0 - light.influenceRadius);\n'
        '    float b = min(endWorld, t0 + light.influenceRadius);')
    source = replace_once(source,
        '    if (!apHazeRayMayContribute(light, toLight, dirWorld, a, b)) {',
        '    const bool narrowBeam = light.padLight1 > 0.0f;\n'
        '    const bool contributes = narrowBeam\n'
        '      ? apClipSpotBeam(light, toLight, dirWorld, t0, a, b)\n'
        '      : apHazeRayMayContribute(light, toLight, dirWorld, a, b);\n'
        '    if (!contributes) {')
    return replace_once(source,
        '      if (!apHazeEvaluateTapGeometry(light, p, geometry)) {\n'
        '        continue;\n'
        '      }\n'
        '      const ApMediumSample m = apSampleMedium(p, args);',
        '      if (!apHazeEvaluateTapGeometry(light, p, narrowBeam, geometry)) {\n'
        '        continue;\n'
        '      }\n'
        '      ApMediumSample m = apSampleMedium(p, args);\n'
        '      if (narrowBeam) {\n'
        "        // Add just enough neutral scattering for this light's shaft. Camera transmittance below\n"
        '        // still uses the global medium, so this never changes scene fog or other lights.\n'
        '        const float meanScattering = dot(m.atmosphere.scattering, vec3(1.0f / 3.0f)) + m.fogScattering;\n'
        '        m.fogScattering += max(light.padLight1 - meanScattering, 0.0f);\n'
        '      }')


def generate_haze_overlay(numos, overlay):
    numos, overlay = map(pathlib.Path, (numos, overlay))
    # Validate every dependency before publishing any changed generated include.
    lights, cull, froxel = [checked_source(numos, relative) for relative in SOURCE_HASHES]
    shader_dir = pathlib.Path(__file__).resolve().parent.parent / 'src/dxvk/shaders/numos_haze'
    helper = (shader_dir / 'numos_haze_helpers.slangh').read_text(encoding='utf-8')
    beam_clip = (shader_dir / 'numos_spot_beam.slangh').read_text(encoding='utf-8')
    lights = replace_once(lights, '#include "numos/lib/aerial_perspective_light.h"',
                          '#include "numos/lib/aerial_perspective_light.h"\n'
                          '#include "numos/lib/numos_haze_helpers.slangh"')
    lights = replace_once(lights,
        '  const uint count = min(clusters[offset], AERIAL_PERSPECTIVE_MAX_LIGHTS_PER_CLUSTER);',
        '  ApHazeLightIterator lightIterator = apHazeLightIterator(clusters, offset, args.apLocalLightCount);')
    lights = replace_once(lights,
        '  for (uint slot = 0u; slot < count; ++slot) {\n'
        '    const AerialPerspectiveLight light = lights[clusters[offset + 1u + slot]];',
        '  uint lightIndex;\n'
        '  while (apHazeNextLight(clusters, lightIterator, lightIndex)) {\n'
        '    const AerialPerspectiveLight light = lights[lightIndex];')
    # Compute the expensive regularized distance only after rejecting zero-contribution lights.
    h_line = '    const float h = max(length(toLight - dirWorld * t0), max(light.equivalentRadius, 1e-4f));\n'
    lights = replace_once(lights, h_line, '')
    lights = replace_once(lights, '    const float thetaA = atan((a - t0) / h);',
        '    if (!apHazeRayMayContribute(light, toLight, dirWorld, a, b)) {\n'
        '      continue;\n'
        '    }\n'
        + h_line + '    const float thetaA = atan((a - t0) / h);')
    lights = add_finite_spot_beam(split_haze_tap_geometry(lights), beam_clip)

    cull = replace_once(cull,
        '  const uint lightCount = args.apLocalLightCount;\n'
        '  if (lightCount == 0u) {\n'
        '    aerialPerspectiveLightClusters[clusterOffset] = 0u;\n'
        '    return;\n'
        '  }',
        '  const uint lightCount = args.apLocalLightCount;\n'
        '  const uint groupSize = apHazeGroupSize(lightCount);\n'
        '  uint groupWords[16];\n'
        '  [unroll]\n'
        '  for (uint word = 0u; word < 16u; ++word) {\n'
        '    groupWords[word] = 0u;\n'
        '  }\n'
        '  if (lightCount == 0u) {\n'
        '    aerialPerspectiveLightClusters[clusterOffset] = kApHazeBitsetMarker;\n'
        '    [unroll]\n'
        '    for (uint word = 0u; word < 16u; ++word) {\n'
        '      aerialPerspectiveLightClusters[clusterOffset + 1u + word] = 0u;\n'
        '    }\n'
        '    return;\n'
        '  }')
    cull = replace_once(cull, 'lightIndex < lightCount && count < AERIAL_PERSPECTIVE_MAX_LIGHTS_PER_CLUSTER',
                         'lightIndex < lightCount')
    cull = replace_once(cull, 'outside = outside || dot(rel, planes[k]) < -light.influenceRadius;',
                         'outside = outside || apHazeOutsidePlane(light, rel, planes[k]);')
    cull = replace_once(cull,
        '        aerialPerspectiveLightClusters[clusterOffset + 1u + count] = lightIndex;',
        '        const uint group = lightIndex / groupSize;\n'
        '        groupWords[group >> 5u] |= 1u << (group & 31u);')
    cull = replace_once(cull, '    aerialPerspectiveLightClusters[clusterOffset] = count;',
        '    [unroll]\n'
        '    for (uint word = 0u; word < 16u; ++word) {\n'
        '      aerialPerspectiveLightClusters[clusterOffset + 1u + word] = groupWords[word];\n'
        '    }\n'
        '    aerialPerspectiveLightClusters[clusterOffset] = kApHazeBitsetMarker | count;')
    cull = replace_once(cull, '  const vec3 centerWorld = args.cameraPosition + center;',
        '  const vec3 centerWorld = args.cameraPosition + center;\n'
        '  // Surface depth/guard can move a lighting sample in front of this slice. Cover\n'
        '  // the convex hull of the original cluster sphere and camera for the new cone test.\n'
        '  const float centerDistance = length(center);\n'
        '  const float coneRadius = centerDistance > radius ? 0.5f * (centerDistance + radius) : radius;\n'
        '  const vec3 coneCenter = centerDistance > radius\n'
        '    ? center * ((centerDistance - radius) / (2.0f * centerDistance)) : center;\n'
        '  const vec3 coneCenterWorld = args.cameraPosition + coneCenter;')
    cull = replace_once(cull,
        '    const vec3 offset = light.position - centerWorld;\n'
        '    const float reach = radius + light.influenceRadius;\n'
        '    if (dot(offset, offset) > reach * reach) {',
        '    const vec3 offset = light.position - centerWorld;\n'
        '    const float reach = radius + light.influenceRadius;\n'
        '    const float tolerance = 2e-5f * (1.0f + length(offset) + abs(reach));\n'
        '    if (length(offset) > reach + tolerance ||\n'
        '        apHazeOutsideConeSphere(light, coneCenterWorld - light.position, coneRadius)) {')
    cull = replace_once(cull,
        '    aerialPerspectiveLightClusters[clusterOffset + 1u + written] = lightIndex;\n\n'
        '    // The CPU sorts the list by importance, so stopping at the cap keeps the brightest lights.\n'
        '    ++written;\n'
        '    if (written >= AERIAL_PERSPECTIVE_MAX_LIGHTS_PER_CLUSTER) {\n'
        '      break;\n'
        '    }',
        '    const uint group = lightIndex / groupSize;\n'
        '    groupWords[group >> 5u] |= 1u << (group & 31u);\n'
        '    ++written;')
    cull = replace_once(cull, '  aerialPerspectiveLightClusters[clusterOffset] = written;',
        '  [unroll]\n'
        '  for (uint word = 0u; word < 16u; ++word) {\n'
        '    aerialPerspectiveLightClusters[clusterOffset + 1u + word] = groupWords[word];\n'
        '  }\n'
        '  aerialPerspectiveLightClusters[clusterOffset] = kApHazeBitsetMarker | written;')
    froxel = replace_once(froxel,
        '    const uint count = min(aerialPerspectiveLightClusters[clusterOffset], AERIAL_PERSPECTIVE_MAX_LIGHTS_PER_CLUSTER);\n'
        '    if (count != 0u) {',
        '    ApHazeLightIterator lightIterator = apHazeLightIterator(aerialPerspectiveLightClusters, clusterOffset, args.apLocalLightCount);\n'
        '    if ((aerialPerspectiveLightClusters[clusterOffset] & ~kApHazeBitsetMarker) != 0u) {')
    froxel = replace_once(froxel,
        '      for (uint slot = 0u; slot < count; ++slot) {\n'
        '        const uint lightIndex = aerialPerspectiveLightClusters[clusterOffset + 1u + slot];',
        '      uint lightIndex;\n'
        '      while (apHazeNextLight(aerialPerspectiveLightClusters, lightIterator, lightIndex)) {')

    outputs = {
        'numos/lib/aerial_perspective_lights.slangh': lights,
        'numos/lib/numos_haze_helpers.slangh': helper,
        'aerial_perspective_light_cull.comp.slang': cull,
        'aerial_perspective_froxel_lighting.comp.slang': froxel,
    }
    for relative, text in outputs.items():
        write_if_changed(overlay / relative, text)
    return {relative: hashlib.sha256(text.encode('utf-8')).hexdigest() for relative, text in outputs.items()}


def build_haze_shaders(numos, slangc, out, spirvopt=None, spirvval=None, overlay=None, originals=None):
    if bool(spirvopt) != bool(spirvval):
        raise ValueError('Haze optimization requires both spirv-opt and spirv-val')
    numos, slangc, out = map(pathlib.Path, (numos, slangc, out))
    out.mkdir(parents=True, exist_ok=True)
    overlay = pathlib.Path(overlay) if overlay else out / 'haze_source'
    originals = pathlib.Path(originals) if originals else out
    generated = generate_haze_overlay(numos, overlay)
    spec = importlib.util.spec_from_file_location('numos_haze_pass_reflection', numos / 'tools/gen_pass_table.py')
    reflection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reflection)
    flags = ['-entry', 'main', '-target', 'spirv', '-zero-initialize', '-emit-spirv-directly',
             '-D__SLANG__', '-matrix-layout-column-major', '-Wno-30081', '-fvk-use-scalar-layout']
    variants = [
        ('aerial_perspective_light_cull', 'aerial_perspective_light_cull', []),
        ('aerial_perspective_froxel_lighting', 'aerial_perspective_froxel_lighting', []),
        ('aerial_perspective_froxel_lighting_traced', 'aerial_perspective_froxel_lighting', ['-DAP_TRACE_SCENE']),
    ]
    validation = {'layoutVersion': LAYOUT_VERSION, 'sourceHashes': SOURCE_HASHES,
                  'generatedHashes': generated, 'optimized': bool(spirvopt), 'variants': []}
    for name, source, defines in variants:
        original = originals / (name + '.comp.spv')
        target = out / (name + '.comp.haze.spv')
        # Do not leave a stale, apparently usable metadata record beside a failed compile.
        metadata = target.with_suffix('.meta')
        if metadata.exists():
            metadata.unlink()
        start = time.monotonic()
        subprocess.run([str(slangc), str(overlay / (source + '.comp.slang')),
                        '-I', str(overlay), '-I', str(numos / 'shaders'),
                        *flags, *defines, '-o', str(target)], check=True)
        compile_seconds = time.monotonic() - start
        payload = target.read_bytes()
        if len(payload) % 4 or struct.unpack_from('<I', payload)[0] != 0x07230203:
            raise ValueError(f'Invalid haze SPIR-V: {target}')
        expected, actual = reflection.reflect(original), reflection.reflect(target)
        # All numerical descriptor properties and local/push sizes must be unchanged.
        abi = lambda value: (value[0], value[1], [binding[:-1] for binding in value[2]])
        if abi(expected) != abi(actual) or specialization_constants(original) != specialization_constants(target):
            raise ValueError(f'NUMOS haze shader ABI changed: {target}')
        raw_size = len(payload)
        start = time.monotonic()
        if spirvopt:
            optimize_scene_shader(target, reflection, spirvopt, spirvval)
            payload = target.read_bytes()
        optimize_seconds = time.monotonic() - start
        metadata.write_text(f'{fnv1a(original.read_bytes()):016x} {fnv1a(payload):016x} {LAYOUT_VERSION}\n', encoding='ascii')
        validation['variants'].append({
            'name': name, 'sha256': hashlib.sha256(payload).hexdigest(), 'originalBytes': original.stat().st_size,
            'rawBytes': raw_size, 'optimizedBytes': len(payload), 'compileSeconds': compile_seconds,
            'optimizeSeconds': optimize_seconds, 'abi': abi(actual), 'metadata': metadata.read_text().strip(),
        })
        print(f'Built complete haze list: {target.name}', flush=True)
    write_if_changed(out / 'haze-validation.json', json.dumps(validation, indent=2) + '\n')
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    emit = sub.add_parser('emit-overlay')
    emit.add_argument('numos')
    emit.add_argument('overlay')
    build = sub.add_parser('build')
    build.add_argument('numos')
    build.add_argument('slangc')
    build.add_argument('out')
    build.add_argument('--overlay')
    build.add_argument('--originals')
    build.add_argument('--spirv-opt')
    build.add_argument('--spirv-val')
    args = parser.parse_args()
    if args.command == 'emit-overlay':
        generate_haze_overlay(args.numos, args.overlay)
    else:
        build_haze_shaders(args.numos, args.slangc, args.out, args.spirv_opt, args.spirv_val, args.overlay, args.originals)


if __name__ == '__main__':
    main()
