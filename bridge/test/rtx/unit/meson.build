
tests = []

# Command History Test
if (cpu_family == 'x86_64')
cmd_history_exe = executable(
    'command_history',  files('test_cmd_queue.cpp'),
    include_directories : util_include_path,
    dependencies : util_dep,
    win_subsystem : 'console')
test('command_history', cmd_history_exe, timeout : 5)
tests += cmd_history_exe
endif

# Remix API Serializing Test
if (cpu_family == 'x86_64')
    name_suff = 'x64'
else
    name_suff = 'x86'
endif
write_name = 'test_remix_api_write'
write_exe = write_name + '_'  + name_suff
tests += executable(
    write_exe,  files(write_name + '.cpp'),
    include_directories : [util_include_path, public_include_path ],
    dependencies : util_dep,
    win_subsystem : 'console',)
read_name = 'test_remix_api_read'
read_exe = read_name + '_'  + name_suff
tests += executable(
    read_exe,  files(read_name + '.cpp'),
    include_directories : [util_include_path, public_include_path ],
    dependencies : util_dep,
    win_subsystem : 'console')
# Bridge streaming test: the x86 build is the client and starts the x64 build as the server.
tests += executable(
    'test_bridge_stream_' + name_suff,  files('test_bridge_stream.cpp'),
    include_directories : [util_include_path, public_include_path ],
    dependencies : util_dep,
    win_subsystem : 'console')
if (cpu_family == 'x86_64')
python = find_program('python')
remix_api_test_script = join_paths(meson.current_source_dir(), 'test_remix_api.py')
remix_api_test_artifacts_dir = join_paths(meson.global_build_root(), 'test_artifacts', 'remix_api')
variant = 'x64_2_x86'
test('remix_api__' + variant,
     python,
     args : [remix_api_test_script,
             '--' + variant,
             '--executablesDir', meson.current_build_dir(),
             '--artifactDir', remix_api_test_artifacts_dir ])
variant = 'x86_2_x64'
test('remix_api__' + variant,
     python,
     args : [remix_api_test_script,
             '--' + variant,
             '--executablesDir', meson.current_build_dir(),
             '--artifactDir', remix_api_test_artifacts_dir ])
endif
gpu_payload_exe = executable('test_gpu_instancing_payload_' + name_suff, files('test_gpu_instancing_payload.cpp'),
    include_directories : [util_include_path, public_include_path],
    win_subsystem : 'console')
test('gpu_instancing_payload', gpu_payload_exe)
tests += gpu_payload_exe

retained_error_exe = executable('test_retained_error_' + name_suff, files('test_retained_error.cpp'),
    include_directories : [util_include_path, public_include_path],
    win_subsystem : 'console')
test('retained_error', retained_error_exe)
tests += retained_error_exe

skinning_payload_exe = executable('test_mesh_skinning_payload_' + name_suff, files('test_mesh_skinning_payload.cpp'),
    include_directories : [util_include_path, public_include_path],
    dependencies : util_dep,
    win_subsystem : 'console')
test('mesh_skinning_payload', skinning_payload_exe)
tests += skinning_payload_exe

alias_target('unit_tests', tests)

