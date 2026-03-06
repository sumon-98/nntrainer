// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   main.cpp
 * @date   3 Jan 2026
 * @brief  dynamic onnx example using nntrainer-onnx-api
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#ifdef ENABLE_ONNX_INTERPRETER
#include <onnx_interpreter.h>
#include <unordered_map>
#endif
#include <optimizer.h>
#include <util_func.h>

int main() {
  auto model = ml::train::createModel();
#ifdef ENABLE_ONNX_INTERPRETER
  try {
    std::string path = "/workspace/nntrainer/Applications/ONNX/python/qwen3/multi-token-kv-cache/prefill/prefill.onnx";

    nntrainer::ONNXInterpreter interp;

    // Example user-provided mapping of symbolic dimension names to values.
    std::unordered_map<std::string, int> dim_map = {{"past_seq", 256}};

    // This step should be done before 'deserialize'
    interp.setDimParamMap(dim_map);

    auto graph = interp.deserialize(path);

    // add layers produced by the interpreter into the model
    for (auto &node : graph) {
      model->addLayer(node);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }
#endif
  try {
    model->compile();
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  return 0;
}
