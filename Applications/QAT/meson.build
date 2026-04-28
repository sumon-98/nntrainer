qat_src = [
  'qat_fc_layer.cpp',
  'main_qat_mnist.cpp'
]

qat_inc = include_directories('.')

executable('nntrainer_qat',
  qat_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: qat_inc,
  install: true,
  install_dir: application_install_dir
)
