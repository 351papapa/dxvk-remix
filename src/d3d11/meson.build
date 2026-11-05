d3d11_res = wrc_generator.process('version.rc')

dxgi_common_src = [
  '../dxgi/dxgi_format.cpp',
  '../dxgi/dxgi_monitor.cpp',
  '../dxgi/dxgi_swapchain.cpp',
]

d3d10_src = files([
  '../d3d10/d3d10_blend.cpp',
  '../d3d10/d3d10_buffer.cpp',
  '../d3d10/d3d10_depth_stencil.cpp',
  '../d3d10/d3d10_device.cpp',
  '../d3d10/d3d10_input_layout.cpp',
  '../d3d10/d3d10_multithread.cpp',
  '../d3d10/d3d10_query.cpp',
  '../d3d10/d3d10_rasterizer.cpp',
  '../d3d10/d3d10_sampler.cpp',
  '../d3d10/d3d10_texture.cpp',
  '../d3d10/d3d10_util.cpp',
  '../d3d10/d3d10_view_dsv.cpp',
  '../d3d10/d3d10_view_rtv.cpp',
  '../d3d10/d3d10_view_srv.cpp',
])

d3d11_src = files([
  'd3d11_annotation.cpp',
  'd3d11_blend.cpp',
  'd3d11_buffer.cpp',
  'd3d11_class_linkage.cpp',
  'd3d11_cmdlist.cpp',
  'd3d11_context.cpp',
  'd3d11_context_def.cpp',
  'd3d11_context_ext.cpp',
  'd3d11_context_imm.cpp',
  'd3d11_cuda.cpp',
  'd3d11_depth_stencil.cpp',
  'd3d11_device.cpp',
  'd3d11_enums.cpp',
  'd3d11_gdi.cpp',
  'd3d11_initializer.cpp',
  'd3d11_input_layout.cpp',
  'd3d11_interop.cpp',
  'd3d11_main.cpp',
  'd3d11_options.cpp',
  'd3d11_query.cpp',
  'd3d11_rasterizer.cpp',
  'd3d11_resource.cpp',
  'd3d11_sampler.cpp',
  'd3d11_shader.cpp',
  'd3d11_state.cpp',
  'd3d11_state_object.cpp',
  'd3d11_swapchain.cpp',
  'd3d11_texture.cpp',
  'd3d11_util.cpp',
  'd3d11_video.cpp',
  'd3d11_view_dsv.cpp',
  'd3d11_view_rtv.cpp',
  'd3d11_view_srv.cpp',
  'd3d11_view_uav.cpp',
])

d3d11_shaders = files([
  'shaders/d3d11_video_blit_frag.frag',
  'shaders/d3d11_video_blit_vert.vert',
])

# The D3D11 sources are compiled into the unified module (see
# src/d3d9/meson.build), not into a DLL of their own. Two DLLs that each
# statically link libdxvk get a private copy of every global in it, so they
# could never share a DxvkDevice -- which is the whole point of building this
# frontend. What ships as d3d11.dll is an export forwarder.
d3d11_compiled_shaders = glsl_generator.process(d3d11_shaders)

d3d11_dll = shared_library('d3d11'+dll_ext, files('d3d11_forward.cpp'),
  name_prefix         : '',
  install             : true,
  override_options    : ['cpp_std='+dxvk_cpp_std])
