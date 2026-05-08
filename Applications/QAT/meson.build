qat_layer_src = [
  meson.current_source_dir() / 'qat_fc_layer.cpp',
]

qat_inc = include_directories('.')

# Export the QAT layer as a linkable dependency for other Applications
qat_layer_lib = static_library('qat_layer',
  qat_layer_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: qat_inc,
)

qat_layer_dep = declare_dependency(
  link_with: qat_layer_lib,
  include_directories: qat_inc,
)

# MNIST QAT test executable (standalone)
executable('nntrainer_qat',
  ['main_qat_mnist.cpp'],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep, qat_layer_dep],
  include_directories: qat_inc,
  install: true,
  install_dir: application_install_dir
)
