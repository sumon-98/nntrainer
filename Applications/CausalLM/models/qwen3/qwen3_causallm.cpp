/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	qwen3_causallm.cpp
 * @date	23 July 2025
 * @brief	This defines a qwen3 causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <llm_util.hpp>
#include <model.h>
#include <qwen3_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <reshaped_rms_norm.h>

namespace causallm {

std::vector<LayerHandle> Qwen3Transformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {
  
  std::cout << "Entered qwen3_causallm.cpp's createAttention function." << std::endl;
    
  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  if (isLoRATarget("wq")) {
    q_params.push_back(withKey("lora_rank", LORA_RANK));
    q_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  }
  layers.push_back(createLayer("fully_connected", q_params));

  // Q-reshaped-norm layer
  // q_norm(q_proj.view(hidden_shape))
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("input_layers", Q),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("trainable", "false")};
  layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  if (isLoRATarget("wk")) {
    k_params.push_back(withKey("lora_rank", LORA_RANK));
    k_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  }
  layers.push_back(createLayer("fully_connected", k_params));

  // K-reshaped-norm layer
  // k_norm(k_proj.view(hidden_shape))
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm), withKey("input_layers", K),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("trainable", "false")};
  layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  if (isLoRATarget("wv")) {
    v_params.push_back(withKey("lora_rank", LORA_RANK));
    v_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  }
  layers.push_back(createLayer("fully_connected", v_params));

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", SLIDING_WINDOW),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q_norm, K_norm, V}),
    withKey("trainable", "false")};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  if (isLoRATarget("wo")) {
    o_params.push_back(withKey("lora_rank", LORA_RANK));
    o_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  }
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

void Qwen3Transformer::registerCustomLayers() {
  ///
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Qwen3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen3Transformer::registerCustomLayers();
}

// LoRA Debugging
void Qwen3CausalLM::exportWeightsToFile(const std::string& filename) const {
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }
  
  outfile << std::fixed << std::setprecision(6);
  outfile << "=== Detailed Model Weights Export ===" << std::endl;
  outfile << "Export Time: " << std::time(nullptr) << std::endl;
  outfile << "=====================================" << std::endl << std::endl;
  
  if (!model) {
    outfile << "Error: Model is not initialized" << std::endl;
    outfile.close();
    return;
  }
  
  // Use the forEachLayer method to iterate through all layers
  model->forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &rc, void *user_data) {
      std::ofstream &outfile = *static_cast<std::ofstream*>(user_data);
      
      outfile << "Layer: " << layer.getName() << " (Type: " << layer.getType() << ")\n";
      
      try {
        // Get weights using the layer interface
        std::vector<float*> weights;
        std::vector<ml::train::TensorDim> weight_dims;
        layer.getWeights(weights, weight_dims);
        
        for (size_t i = 0; i < weights.size(); ++i) {
          if (weights[i] && weight_dims[i].getFeatureLen() > 0) {
            outfile << "  Weight " << i << " (Name: " << layer.getWeightName(i) 
                    << ", Dim: " << weight_dims[i].getFeatureLen() << "): ";
            
            // Print first 10 and last 10 values for large weights
            size_t len = weight_dims[i].getFeatureLen();
            size_t print_count = std::min(len, (size_t)20);
            
            for (size_t j = 0; j < std::min(print_count/2, len); ++j) {
              outfile << weights[i][j] << " ";
            }
            
            if (len > print_count) {
              outfile << "... ";
              for (size_t j = len - print_count/2; j < len; ++j) {
                outfile << weights[i][j] << " ";
              }
            } else {
              for (size_t j = print_count/2; j < len; ++j) {
                outfile << weights[i][j] << " ";
              }
            }
            outfile << "\n";
          }
        }
      } catch (const std::exception& e) {
        outfile << "  Error accessing weights: " << e.what() << "\n";
      }
      outfile << "\n";
    },
    &outfile
  );
  
  outfile.close();
  std::cout << "Detailed model weights exported to: " << filename << std::endl;
}

} // namespace causallm
