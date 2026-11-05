# Frontends. Compiled into the module of the API they intercept; the backend (src/dxvk)
# never includes anything from here, which the check below enforces at configure time.

run_command(python_interpreter, global_src_root_norm + '/scripts-common/check_frontend_deps.py', global_src_root_norm,
  check : true)

# Shared translator toolkit (frontends/translator/).
frontend_translator_src = files([
  'translator/camera_classifier.cpp',
  'translator/camera_classifier.h',
  'translator/change_test.h',
  'translator/decal_rank.h',
  'translator/draw_key.h',
  'translator/fog_order.h',
  'translator/frame_batch.h',
  'translator/instance_matcher.cpp',
  'translator/instance_matcher.h',
  'translator/lifetime.h',
  'translator/material_translator.cpp',
  'translator/material_translator.h',
  'translator/resource_stamp.h',
  'translator/sky_rules.cpp',
  'translator/sky_rules.h',
])

# The Remix API's D3D9 device and color grading exports. The API-only build has them too, since API hosts create
# their D3D9 device through it.
frontend_api_src = files([
  'api/api_color_grade.cpp',
  'api/api_d3d9_device.cpp',
])

# The Remix API's D3D11 host, built with the D3D11 module.
frontend_api_d3d11_src = files([
  'api/api_d3d11_host.cpp',
])

# The D3D9 frontend's draw record. D3D9Rtx fills it in both builds, so the API-only build compiles it too.
frontend_d3d9_record_src = files([
  'd3d9/d3d9_draw_call.cpp',
  'd3d9/d3d9_draw_call.h',
])

# D3D9 frontend, built into the D3D9 module with the toolkit.
frontend_d3d9_src = files([
  'd3d9/d3d9_camera.cpp',
  'd3d9/d3d9_camera.h',
  'd3d9/d3d9_draw_capture.h',
  'd3d9/d3d9_frontend.cpp',
  'd3d9/d3d9_frontend.h',
  'd3d9/d3d9_raster_replay.cpp',
  'd3d9/d3d9_raster_replay.h',
]) + frontend_translator_src
