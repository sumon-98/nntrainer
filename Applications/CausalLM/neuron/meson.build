# nntrainer/Applications/CausalLM/neuron/meson.build
# Smoke-test executable for the Neuron backend.

neuron_app_sources = [
  'nntrainer_neuron_smoke.cpp',
]

neuron_smoke_exe = executable(
  'nntrainer_neuron_smoke',
  neuron_app_sources,
  dependencies: [ccapi_dep, nntrainer_dep],
  include_directories: [nntrainer_inc],
  install: true,
  install_dir: nntrainer_bindir,
)

# Host-side unit test for NeuronKvCacheManager: pure bookkeeping logic, no
# NPU/.dla/device involved, so this builds and runs as a plain native
# executable (unlike nntrainer_neuron_smoke, which needs an Android target).
neuron_kv_cache_manager_test_exe = executable(
  'neuron_kv_cache_manager_test',
  ['neuron_kv_cache_manager_test.cpp'],
  install: false,
)
test('neuron_kv_cache_manager_test', neuron_kv_cache_manager_test_exe)
