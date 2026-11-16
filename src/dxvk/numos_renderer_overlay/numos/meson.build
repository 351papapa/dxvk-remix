configure_file(
  input: files('../../../../submodules/numos/include/numos/renderer.h', 'renderer.h.patch'),
  output: 'renderer.h',
  command: [python, numos_renderer_overlay_script, '@INPUT0@', '@INPUT1@', '@OUTPUT@'],
)
