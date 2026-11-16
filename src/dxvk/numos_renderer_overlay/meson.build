numos_renderer_overlay_script = files('../../../scripts-common/numos_renderer_overlay.py')
numos_renderer_overlay_headers = include_directories('.')
numos_renderer_overlay_sources = []

foreach name : ['renderer.cpp', 'renderer_api.cpp', 'renderer_impl.h']
  source = configure_file(
    input: files('../../../submodules/numos/src/' + name, name + '.patch'),
    output: name,
    command: [python, numos_renderer_overlay_script, '@INPUT0@', '@INPUT1@', '@OUTPUT@'],
  )
  numos_renderer_overlay_sources += [source]
endforeach

subdir('numos')
