"""Builds and installs NUMOS's compute shaders for the runtime (submodules/numos).

  build:   numos_shaders.py build <numos dir> <slangc> <spirv out dir> <stamp file>
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


def build(numos, slangc, out, stamp):
    tools = numos / 'tools'
    subprocess.run([sys.executable, str(tools / 'build_shaders.py'), '--slangc', str(slangc), '--out', str(out),
                    '--jobs', '4'], check=True)
    subprocess.run([sys.executable, str(tools / 'gen_pass_table.py'), '--spirv', str(out),
                    '--out', str(numos / 'src' / 'generated' / 'passes.inc'), '--check'], check=True)
    pathlib.Path(stamp).write_text('\n'.join(sorted(p.name for p in out.glob('*.spv'))) + '\n')


def install(numos, spirv, dest):
    for src, sub, pattern in ((spirv, 'spirv', '*.spv'), (numos / 'assets' / 'nightsky', 'nightsky', '*')):
        dst = dest / 'numos' / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob(pattern):
            if f.is_file():
                shutil.copy2(f, dst / f.name)


def main():
    mode, numos = sys.argv[1], pathlib.Path(sys.argv[2])
    if mode == 'build':
        build(numos, pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]), sys.argv[5])
    elif mode == 'install':
        install(numos, pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))
    else:
        sys.exit(f'unknown mode {mode}')


if __name__ == '__main__':
    main()
