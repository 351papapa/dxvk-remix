"""Builds and installs NUMOS's compute shaders for the runtime (submodules/numos).

  build:   numos_shaders.py build <numos dir> <slangc> <spirv out dir> <stamp file> [<spirv-opt> <spirv-val>]
  install: numos_shaders.py install <numos dir> <spirv dir> <destination dir>

build compiles every NUMOS pass with Remix's slangc through NUMOS's own tools/build_shaders.py, then
checks the result against the pass table compiled into the library (src/generated/passes.inc): the
runtime refuses any SPIR-V whose hash differs from the table, so a mismatch fails the build here
instead of leaving a black sky at run time.

install copies the SPIR-V to <destination>/numos/spirv and the night sky data to
<destination>/numos/nightsky, where the runtime loads them from beside d3d9.dll.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

from numos_scene_shaders import build_scene_shaders
from numos_haze_shaders import build_haze_shaders


def check_original_passes(numos, out):
    # Sidecars share the install directory but are not entries in the NUMOS table.
    with tempfile.TemporaryDirectory(prefix='numos-originals-', dir=out) as directory:
        originals = pathlib.Path(directory)
        if originals.resolve().parent != out.resolve():
            raise ValueError('NUMOS pass-check directory escaped the build output')
        for source in out.glob('*.comp.spv'):
            shutil.copy2(source, originals / source.name)
        subprocess.run([sys.executable, str(numos / 'tools/gen_pass_table.py'), '--spirv', str(originals),
                        '--out', str(numos / 'src/generated/passes.inc'), '--check'], check=True)


def build(numos, slangc, out, stamp, spirvopt=None, spirvval=None):
    tools = numos / 'tools'
    subprocess.run([sys.executable, str(tools / 'build_shaders.py'), '--slangc', str(slangc), '--out', str(out),
                    '--jobs', '4'], check=True)
    check_original_passes(numos, out)
    build_scene_shaders(numos, slangc, out, spirvopt, spirvval)
    build_haze_shaders(numos, slangc, out, spirvopt, spirvval)
    pathlib.Path(stamp).write_text('\n'.join(sorted(p.name for p in out.glob('*.spv'))) + '\n')


def install(numos, spirv, dest):
    for src, sub, pattern in ((spirv, 'spirv', '*.spv'), (spirv, 'spirv', '*.meta'), (numos / 'assets' / 'nightsky', 'nightsky', '*')):
        dst = dest / 'numos' / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob(pattern):
            if f.is_file():
                shutil.copy2(f, dst / f.name)
    legal = dest / 'numos' / 'licenses'
    legal.mkdir(parents=True, exist_ok=True)
    for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(numos / name, legal / name)
    for source in (numos / 'licenses').glob('*'):
        if source.is_file():
            shutil.copy2(source, legal / source.name)


def main():
    mode, numos = sys.argv[1], pathlib.Path(sys.argv[2])
    if mode == 'build':
        if len(sys.argv) not in (6, 8):
            sys.exit('build expects numos, slangc, output, stamp, and optional spirv-opt/spirv-val')
        build(numos, pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]), sys.argv[5],
              *(sys.argv[6:8]))
    elif mode == 'install':
        install(numos, pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))
    else:
        sys.exit(f'unknown mode {mode}')


if __name__ == '__main__':
    main()
