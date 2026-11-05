from pathlib import Path
import ast
import os
import re
import sys
from types import SimpleNamespace

root=Path(__file__).resolve().parents[3]
work=Path(sys.argv[1])
source=root/'scripts-common/compile_shaders.py'
ns={'os':os,'re':re,'args':SimpleNamespace(output=str(work))}
tree=ast.parse(source.read_text())
for node in tree.body:
    if isinstance(node,ast.FunctionDef) or (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='shaderTypeSuffixes' for t in node.targets)):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
ns['writeVariantMatrixHeader']=lambda *args: None
fixtures=work/'feature-axis-tests'
fixtures.mkdir(exist_ok=True)
explicit=fixtures/'explicit.comp.slang'
explicit.write_text('//!feature wetness ENABLE_SURFACE_WETNESS\n//!variant plain.comp\n//!variant alternate.comp\n//!> ALTERNATE=1\n//!end-variants\n')
assert ns['parseShaderVariants'](str(explicit)) == [
    ['plain.comp'],['alternate.comp','ALTERNATE=1'],
    ['plain_wetness.comp','ENABLE_SURFACE_WETNESS'],
    ['alternate_wetness.comp','ALTERNATE=1','ENABLE_SURFACE_WETNESS']]
matrix=fixtures/'matrix.slang'
matrix.write_text('//!feature wetness ENABLE_SURFACE_WETNESS\n//!variant-matrix matrix\n//!> axis mode compute.comp COMPUTE\n//!> axis mode ray.rgen RAY\n//!> axis quality -\n//!> axis quality full FULL\n//!> order mode quality\n//!end-variants\n')
variants=ns['parseShaderVariants'](str(matrix))
assert len(variants)==8
assert variants[:4] == ns['parseBaseShaderVariants'](str(matrix))
assert variants[-1] == ['matrix_ray_full_wetness.rgen','RAY','FULL','ENABLE_SURFACE_WETNESS']
names=set(); enabled=0
for file in (root/'src/dxvk/shaders/rtx').rglob('*.slang'):
    if '//!feature wetness ' not in file.read_text(): continue
    base=ns['parseBaseShaderVariants'](str(file))
    variants=ns['parseShaderVariants'](str(file))
    assert variants[:len(base)]==base
    assert len(variants)==2*len(base)
    for variant in variants:
        assert variant[0] not in names, variant[0]
        names.add(variant[0])
        enabled += 'ENABLE_SURFACE_WETNESS' in variant
assert enabled==465, enabled
print(f'Feature-axis tests passed: {enabled} enabled variants, unchanged disabled names and defines, no collisions.')
