# QAT Quantized Inference Benchmark
# All three benchmarks use NNTrainer Tensor API directly (no qat_layer_dep)

# Modes A/B/C: Per-tensor-affine INT8 (latency + accuracy)
executable('nntrainer_qat_benchmark',
  ['qat_benchmark.cpp'],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  install: true,
  install_dir: application_install_dir
)

# Modes D/E: GGML-quantized SIMD inference (latency + accuracy)
executable('nntrainer_qat_benchmark_ggml',
  ['qat_benchmark_ggml.cpp'],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  install: true,
  install_dir: application_install_dir
)

# Forced-scale comparison: per-tensor vs per-block accuracy only
executable('nntrainer_qat_benchmark_forced_scale',
  ['qat_benchmark_forced_scale.cpp'],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  install: true,
  install_dir: application_install_dir
)

# Comprehensive MNIST POC: 5-mode comparison (FP32, Q6_K, QAT, QAT+Q6_K, forced)
executable('nntrainer_qat_mnist_poc',
  ['qat_mnist_full_poc.cpp'],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep, qat_layer_dep],
  include_directories: [qat_inc],
  install: true,
  install_dir: application_install_dir
)
