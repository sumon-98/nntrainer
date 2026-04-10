lora_example_sources = [
  'main.cpp',
  'mnist_loader.cpp',
]

lora_run_sources = [
  'lora_run.cpp',
  'mnist_loader.cpp',
]

lora_example_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  lora_example_dependencies += [gtest_dep]
endif

e = executable('lora_example',
  lora_example_sources,
  include_directories: [include_directories('.')],
  dependencies: lora_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

# Inference-only executable
e2 = executable('lora_run',
  lora_run_sources,
  include_directories: [include_directories('.')],
  dependencies: lora_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
