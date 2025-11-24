// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 * @file   main.cpp
 * @date   14 October 2025
 * @brief  onnx example using nntrainer-onnx-api
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <fstream>
#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>

void saveToRaw(const float *data, size_t size, const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Error: Cannot open file " << filename << " for writing.\n";
    return;
  }

  out.write(reinterpret_cast<const char *>(data), size * sizeof(float));
  out.close();

  std::cout << std::endl << ".bin generated successfully !";
}

int main() {
  auto model = ml::train::createModel();

  std::cout << "--------------------------------------Create Model "
               "Done--------------------------------------"
            << std::endl;
  try {
    std::string path =
      "/workspace/nntrainer/Applications/ONNX/python/qwen3/qwen3_model.onnx";
    model->load(path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Load Model "
               "Done--------------------------------------"
            << std::endl;

// Set FP16 tensor type BEFORE compilation for mixed precision
#ifdef ENABLE_FP16
  std::cout << "--------------------------------------Converting model to "
               "FP16--------------------------------------"
            << std::endl;
  model->setProperty({"model_tensor_type=FP16-FP16"});
  model->setProperty({"loss_scale=17768"});
#endif

  try {
    model->compile(ml::train::ExecutionMode::INFERENCE);
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Compile Model "
               "Done--------------------------------------"
            << std::endl;
  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Initialize Model "
               "Done--------------------------------------"
            << std::endl;
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  std::cout << "--------------------------------------Summarize Model "
               "Done--------------------------------------"
            << std::endl;

  std::string weight_path = "/workspace/nntrainer/Applications/ONNX/python/qwen3/bins/";
  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
#ifdef ENABLE_FP16
    std::cout << "--------------------------------------FP32 weights loaded and "
                 "automatically converted to FP16--------------------------------------"
              << std::endl;
#else
    std::cout << "--------------------------------------Loading weights "
                 "Done--------------------------------------"
              << std::endl;
#endif
  } catch (std::exception &e) {
    std::cerr << "Error during loading weights: " << e.what() << "\n";
    return 1;
  }

  // Convert input data to FP16 if model is using FP16
#ifdef ENABLE_FP16
  _FP16 *input = new _FP16[1];
  _FP16 *sin = new _FP16[128];
  _FP16 *cos = new _FP16[128];
  _FP16 *epsilon = new _FP16[1];

  input[0] = static_cast<_FP16>(52);

  for (int i = 0; i < 128; i++) {
    sin[i] = static_cast<_FP16>(0);
    cos[i] = static_cast<_FP16>(1);
  }
  epsilon[0] = static_cast<_FP16>(1e-6);

  std::vector<_FP16 *> in;

  in.push_back(epsilon);
  in.push_back(sin);
  in.push_back(cos);
  in.push_back(input);
#else
  float *input = new float[1];
  float *sin = new float[128];
  float *cos = new float[128];
  float *epsilon = new float[1];

  input[0] = 52;

  for (int i = 0; i < 128; i++) {
    sin[i] = 0;
    cos[i] = 1;
  }
  epsilon[0] = 1e-6;

  std::vector<float *> in;

  in.push_back(epsilon);
  in.push_back(sin);
  in.push_back(cos);
  in.push_back(input);
#endif

  // The inference API expects float* regardless of tensor type
  // NNTrainer handles the conversion internally
  std::vector<float *> float_in;
#ifdef ENABLE_FP16
  // Convert FP16 inputs back to float for the API
  float_in.push_back(reinterpret_cast<float *>(epsilon));
  float_in.push_back(reinterpret_cast<float *>(sin));
  float_in.push_back(reinterpret_cast<float *>(cos));
  float_in.push_back(reinterpret_cast<float *>(input));
#else
  float_in = in;
#endif

  auto ans = model->inference(1, float_in);

  std::cout << "-------------------------------------------Inference "
               "Done--------------------------------------------"
            << std::endl;

  for (auto it : ans) {
    saveToRaw(it, 151936,
              "../../../../Applications/ONNX/jni/nntrainer_logits.bin");
  }

  return 0;
}
