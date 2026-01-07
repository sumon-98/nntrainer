# Common dependencies for all ONNX examples
onnx_example_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  onnx_example_dependencies += [gtest_dep]
endif

# fcModel example
fcModel_sources = [
  'fcModel/main.cpp',
]

executable('nntrainer_onnx_fcModel',
  fcModel_sources,
  include_directories: [include_directories('.'), include_directories('fcModel')],
  dependencies: onnx_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

# llama example
llama_sources = [
  'llama/main.cpp',
]

executable('nntrainer_onnx_llama',
  llama_sources,
  include_directories: [include_directories('.'), include_directories('llama')],
  dependencies: onnx_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

# qwen3 single-token example
qwen3_single_token_sources = [
  'qwen3/single-token/main.cpp',
]

executable('nntrainer_onnx_qwen3_single_token',
  qwen3_single_token_sources,
  include_directories: [include_directories('.'), include_directories('qwen3'), include_directories('qwen3/single-token')],
  dependencies: onnx_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

# qwen3 multi-token example
qwen3_multi_token_sources = [
  'qwen3/multi-token/main.cpp',
]

executable('nntrainer_onnx_qwen3_multi_token',
  qwen3_multi_token_sources,
  include_directories: [include_directories('.'), include_directories('qwen3'), include_directories('qwen3/multi-token')],
  dependencies: onnx_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)

fc_model_executable = executable('nntrainer_onnx_fc_model',
  fc_model_sources,
  include_directories: [include_directories('.')],
  dependencies: onnx_example_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
