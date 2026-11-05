dxgi_res = wrc_generator.process('version.rc')

dxgi_src = files([
  'dxgi_adapter.cpp',
  'dxgi_enums.cpp',
  'dxgi_factory.cpp',
  'dxgi_format.cpp',
  'dxgi_main.cpp',
  'dxgi_monitor.cpp',
  'dxgi_options.cpp',
  'dxgi_output.cpp',
  'dxgi_swapchain.cpp',
])

# Compiled into the unified module; what ships here is a forwarder.
# See src/d3d11/meson.build.
dxgi_dll = shared_library('dxgi'+dll_ext, files('dxgi_forward.cpp'),
  name_prefix         : '',
  install             : true,
  override_options    : ['cpp_std='+dxvk_cpp_std])
