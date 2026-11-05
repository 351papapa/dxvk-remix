dxbc_src = files([
  'dxbc_analysis.cpp',
  'dxbc_chunk_isgn.cpp',
  'dxbc_chunk_shex.cpp',
  'dxbc_common.cpp',
  'dxbc_compiler.cpp',
  'dxbc_defs.cpp',
  'dxbc_decoder.cpp',
  'dxbc_header.cpp',
  'dxbc_module.cpp',
  'dxbc_names.cpp',
  'dxbc_options.cpp',
  'dxbc_reader.cpp',
  'dxbc_util.cpp',
])

# dxvk_shader_include_path: dxbc pulls in rtx_materials.h through the dxvk
# headers, and that includes 'rtx/utility/shared_constants.h', which lives
# under src/dxvk/shaders. The d3d9 frontend gets that path already; dxbc
# never needed it while enable_d3d11 was off.
# /wd4146: dxbc_compiler.cpp uses the standard 'x & -x' lowest-set-bit idiom
# on a uint32_t. Well-defined, deliberate, and upstream DXVK code we would
# rather not diverge from -- but the build is /WX, so MSVC's C4146 stops it.
dxbc_lib = static_library('dxbc', dxbc_src,
  include_directories : [ dxvk_include_path, dxvk_shader_include_path, usd_include_paths, include_directories('../dxvk') ],
  cpp_args            : dxvk_compiler.get_supported_arguments(['/wd4146']),
  override_options    : ['cpp_std='+dxvk_cpp_std])

dxbc_dep = declare_dependency(
  link_with           : [ dxbc_lib ],
  include_directories : [ dxvk_include_path, dxvk_shader_include_path, usd_include_paths, include_directories('../dxvk') ])
