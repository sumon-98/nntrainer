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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>

void loadFromRaw(float *data, size_t size, const std::string &filename) {

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file: " << filename << std::endl;
    return;
  }

  file.read(reinterpret_cast<char *>(data), size * sizeof(float));
  std::streamsize bytesRead = file.gcount();

  if (bytesRead != size * sizeof(float)) {
    std::cerr << "Warning: Expected " << size * sizeof(float)
              << " bytes, but read " << bytesRead << " bytes.\n";
  }

  file.close();
  return;
}

void saveToRaw(float *data, size_t size, const std::string &filename) {
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
               
  // Timing model loading
  auto model_load_start = std::chrono::high_resolution_clock::now();
  try {
    std::string path =
      "../../../../Applications/ONNX/python/qwen3/multi-token/qwen3_model.onnx";
    model->load(path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }
  auto model_load_end = std::chrono::high_resolution_clock::now();
  auto model_load_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      model_load_end - model_load_start);
  double model_load_time = model_load_duration.count() / 1000.0; // Convert to milliseconds

  std::cout << "--------------------------------------Load Model "
               "Done--------------------------------------"
            << std::endl;
  
  // Timing model compilation
  auto compile_start = std::chrono::high_resolution_clock::now();
  try {
    model->compile(ml::train::ExecutionMode::INFERENCE);
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }
  auto compile_end = std::chrono::high_resolution_clock::now();
  auto compile_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      compile_end - compile_start);
  double compile_time = compile_duration.count() / 1000.0; // Convert to milliseconds

  std::cout << "--------------------------------------Compile Model "
               "Done--------------------------------------"
            << std::endl;
  
  // Timing model initialization
  auto init_start = std::chrono::high_resolution_clock::now();
  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }
  auto init_end = std::chrono::high_resolution_clock::now();
  auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      init_end - init_start);
  double init_time = init_duration.count() / 1000.0; // Convert to milliseconds

  std::cout << "--------------------------------------Initialize Model "
               "Done--------------------------------------"
            << std::endl;
  
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  std::cout << "--------------------------------------Summarize Model "
               "Done--------------------------------------"
            << std::endl;

  // Timing weight loading
  auto weight_load_start = std::chrono::high_resolution_clock::now();
  std::string weight_path =
    "../../../../Applications/ONNX/python/qwen3/multi-token/bins/";
  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (std::exception &e) {
    std::cerr << "Error during loading weights: " << e.what() << "\n";
    return 1;
  }
  auto weight_load_end = std::chrono::high_resolution_clock::now();
  auto weight_load_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      weight_load_end - weight_load_start);
  double weight_load_time = weight_load_duration.count() / 1000.0; // Convert to milliseconds

  std::cout << "--------------------------------------Loading weights "
               "Done--------------------------------------"
            << std::endl;

  const int max_embedding_length = 256;
  const int tokens_to_be_generated = 20;
  const int num_vocab = 151936;
  int curr_len = 0;

  float *input = new float[max_embedding_length];
  float *sin = new float[max_embedding_length * 128];
  float *cos = new float[max_embedding_length * 128];
  float *epsilon = new float[1];

  // Loading inputs
  loadFromRaw(
    input, max_embedding_length,
    "../../../../Applications/ONNX/python/qwen3/multi-token/input_tokens.bin");

  for (int i = 0; i < max_embedding_length; i++) {
    if (input[i] == 151643)
      break;
    ++curr_len;
  }

  // Loading rotary embeddings
  loadFromRaw(sin, max_embedding_length * 128,
              "../../../../Applications/ONNX/python/qwen3/multi-token/"
              "rotary_embeddings_sine.bin");
  loadFromRaw(cos, max_embedding_length * 128,
              "../../../../Applications/ONNX/python/qwen3/multi-token/"
              "rotary_embeddings_cosine.bin");

  epsilon[0] = 1e-6;

  // Timing for multi-token generation
  std::vector<double> token_times;
  token_times.reserve(tokens_to_be_generated);
  
  auto total_start = std::chrono::high_resolution_clock::now();
  
  // Pre-allocate vector for finding max element to avoid repeated allocations
  std::vector<float> output_buffer(num_vocab);
  
  for (int i = 0; i < tokens_to_be_generated; i++) {
    auto token_start = std::chrono::high_resolution_clock::now();
    float *output = model->inference(1, {epsilon, sin, cos, input})[0];
    auto token_end = std::chrono::high_resolution_clock::now();
    
    auto token_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        token_end - token_start);
    token_times.push_back(token_duration.count() / 1000.0); // Convert to milliseconds
    
    output = output + (int)(curr_len - 1) * (num_vocab);
    // Copy output to buffer for thread-safe max_element operation
    std::copy(output, output + num_vocab, output_buffer.begin());
    float token_id =
      std::distance(output_buffer.begin(), std::max_element(output_buffer.begin(), output_buffer.end()));
    input[curr_len] = token_id;
    curr_len += 1;
    
    std::cout << "Token " << (i + 1) << " generated in " 
              << token_times.back() << " ms" << std::endl;
  }
  
  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      total_end - total_start);
  
  // Report timing statistics
  double total_time = total_duration.count() / 1000.0; // Convert to milliseconds
  double avg_time = total_time / tokens_to_be_generated;
  
  // Find min and max times
  double min_time = *std::min_element(token_times.begin(), token_times.end());
  double max_time = *std::max_element(token_times.begin(), token_times.end());
  
  std::cout << "\n=== Timing Report ===" << std::endl;
  std::cout << "Total time for " << tokens_to_be_generated 
            << " tokens: " << total_time << " ms" << std::endl;
  std::cout << "Average time per token: " << avg_time << " ms" << std::endl;
  std::cout << "Min time for a token: " << min_time << " ms" << std::endl;
  std::cout << "Max time for a token: " << max_time << " ms" << std::endl;
  std::cout << "Tokens per second: " << (1000.0 / avg_time) << std::endl;
  
  // Print all timing information at the end
  std::cout << "\n=== Complete Timing Report ===" << std::endl;
  std::cout << "Model loading time: " << model_load_time << " ms" << std::endl;
  std::cout << "Model compilation time: " << compile_time << " ms" << std::endl;
  std::cout << "Model initialization time: " << init_time << " ms" << std::endl;
  std::cout << "Weight loading time: " << weight_load_time << " ms" << std::endl;
  std::cout << "Total time for " << tokens_to_be_generated 
            << " tokens: " << total_time << " ms" << std::endl;
  std::cout << "Average time per token: " << avg_time << " ms" << std::endl;
  std::cout << "Tokens per second: " << (1000.0 / avg_time) << std::endl;

  saveToRaw(
    input, curr_len,
    "../../../../Applications/ONNX/python/qwen3/multi-token/output_tokens.bin");

  return 0;
}
