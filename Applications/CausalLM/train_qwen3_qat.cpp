// SPDX-License-Identifier: Apache-2.0
/**
 * @file   train_qwen3_qat.cpp
 * @brief  Qwen3 Weight-Only QAT (Quantization Aware Training) — Full Model
 *
 * This script performs full model training of Qwen3 with weight-only QAT.
 * All fully_connected layers (wq, wk, wv, wo, ffn_up, ffn_gate, ffn_down)
 * are replaced with qat_fully_connected layers that apply fake quantization
 * to weights during the forward pass while leaving activations in FP32.
 *
 * Based on main_lora_train.cpp (full model training).
 *
 * Usage:
 *   ./build/Applications/CausalLM/nntr_qwen3_qat \
 *       <model_dir> <train_data.txt> \
 *       [--lr <float>] [--epochs <int>] [--output <path>] \
 *       [--max_samples <int>] [--skip_weights]
 */

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <causal_lm.h>
#include <lora_train.h>
#include <transformer.h>
#include <factory.h>

#include "json.hpp"
#include "qwen3_causallm.h"

#include <dataset.h>
#include <model.h>
#include <profiler.h>

// QAT layer
#include <app_context.h>
#include <engine.h>
#include <qat_fc_layer.h>

using json = nlohmann::json;

// =============================================================================
// Qwen3 QAT Transformer — Overrides FC layers to use QAT FC
// =============================================================================
namespace causallm {

/**
 * @brief Qwen3 Transformer subclass that replaces all fully_connected layers
 *        with qat_fully_connected layers for weight-only QAT.
 *
 * This follows the same override pattern as Qwen3Transformer, which itself
 * overrides Transformer::createAttention() to add Q/K norm layers.
 */
class Qwen3QATTransformer : public Qwen3CausalLM {
public:
  Qwen3QATTransformer(json &cfg, json &generation_cfg, json &nntr_cfg)
    : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
      CausalLM(cfg, generation_cfg, nntr_cfg),
      Qwen3Transformer(cfg, generation_cfg, nntr_cfg),
      Qwen3CausalLM(cfg, generation_cfg, nntr_cfg) {}

  ~Qwen3QATTransformer() = default;

  /**
   * @brief Override createAttention to use qat_fully_connected for wq/wk/wv/wo
   */
  std::vector<LayerHandle> createAttention(
    const int layer_id, int seq_len, int n_heads, int head_dim,
    std::string query_name, std::string key_name,
    std::string value_name) override;

  /**
   * @brief Override createMlp to use qat_fully_connected for ffn_up/gate/down
   */
  std::vector<LayerHandle> createMlp(
    const int layer_id, int dim, int hidden_dim,
    std::string input_name) override;

  /**
   * @brief Register custom layers including QAT FC
   */
  void registerCustomLayers() override;
};

// ---------------------------------------------------------------------------
// Attention: wq, wk, wv, wo → qat_fully_connected
// Attention core, Q/K norms → unchanged
// ---------------------------------------------------------------------------
std::vector<LayerHandle> Qwen3QATTransformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q layer — QAT FC
  layers.push_back(createLayer("qat_fully_connected", {
    withKey("name", Q),
    withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"),
    withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")}));

  // Q-reshaped-norm layer (unchanged — no learnable matmul weights)
  layers.push_back(createLayer("reshaped_rms_norm", {
    withKey("name", Q_norm),
    withKey("input_layers", Q),
    withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))}));

  // K layer — QAT FC
  layers.push_back(createLayer("qat_fully_connected", {
    withKey("name", K),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"),
    withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")}));

  // K-reshaped-norm layer (unchanged)
  layers.push_back(createLayer("reshaped_rms_norm", {
    withKey("name", K_norm),
    withKey("input_layers", K),
    withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))}));

  // V layer — QAT FC
  layers.push_back(createLayer("qat_fully_connected", {
    withKey("name", V),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"),
    withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")}));

  // Attention core layer (unchanged — no learnable weights)
  layers.push_back(createLayer("mha_core", {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", SLIDING_WINDOW),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q_norm, K_norm, V})}));

  // O layer — QAT FC
  layers.push_back(createLayer("qat_fully_connected", {
    withKey("name", O),
    withKey("unit", DIM),
    withKey("disable_bias", "true"),
    withKey("input_layers", A),
    withKey("weight_initializer", "ones")}));

  return layers;
}

// ---------------------------------------------------------------------------
// MLP: ffn_up, ffn_gate, ffn_down → qat_fully_connected
// SwiGLU → unchanged
// ---------------------------------------------------------------------------
std::vector<LayerHandle> Qwen3QATTransformer::createMlp(
  const int layer_id, int dim, int hidden_dim, std::string input_name) {

  std::vector<LayerHandle> layers;

  // ffn_up — QAT FC
  layers.push_back(createLayer("qat_fully_connected", {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("unit", hidden_dim),
    withKey("disable_bias", "true"),
    withKey("input_layers", input_name),
    withKey("weight_initializer", "ones")}));

  // ffn_gate — QAT FC
  layers.push_back(createLayer("qat_fully_connected", {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
    withKey("unit", hidden_dim),
    withKey("disable_bias", "true"),
    withKey("input_layers", input_name),
    withKey("weight_initializer", "ones")}));

  // swiglu (unchanged — stateless activation)
  layers.push_back(createLayer("swiglu", {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_ffn_gate," +
            "layer" + std::to_string(layer_id) + "_ffn_up")}));

  // ffn_down — QAT FC
  layers.push_back(createLayer("qat_fully_connected", {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
    withKey("unit", dim),
    withKey("disable_bias", "true"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
    withKey("weight_initializer", "ones")}));

  return layers;
}

void Qwen3QATTransformer::registerCustomLayers() {
  // Register all standard Qwen3 layers first
  Qwen3CausalLM::registerCustomLayers();

  // Then register the QAT FC layer
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<nntrainer::QATFullyConnectedLayer>);
    std::cout << "[QAT] Registered qat_fully_connected layer" << std::endl;
  } catch (std::invalid_argument &e) {
    std::cerr << "[QAT] Layer registration note: " << e.what() << std::endl;
  }
}

} // namespace causallm

// =============================================================================
// Main — Full Model QAT Training
// =============================================================================
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <train_data.txt>"
                 " [--lr <float>] [--epochs <int>]"
                 " [--output <path>] [--max_samples <int>] [--skip_weights]"
              << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string train_data_path = argv[2];
  float lr = 1e-5f;           // Lower LR for QAT than normal training
  unsigned int epochs = 1;
  std::string output_path = "qat_weights.bin";
  int max_samples = -1;
  bool skip_weights = false;

  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--lr" && i + 1 < argc) {
      lr = std::atof(argv[++i]);
    } else if (arg == "--epochs" && i + 1 < argc) {
      epochs = std::atoi(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--max_samples" && i + 1 < argc) {
      max_samples = std::atoi(argv[++i]);
    } else if (arg == "--skip_weights") {
      skip_weights = true;
    }
  }

  try {
    auto profiler_listener =
      std::make_shared<nntrainer::profile::GenericProfileListener>();
    PROFILE_BEGIN(profiler_listener);

    std::string config_path = model_dir + "/config.json";
    std::string gen_config_path = model_dir + "/generation_config.json";
    std::string nntr_config_path = model_dir + "/nntr_config.json";

    auto cfg = causallm::LoadJsonFile(config_path);
    auto gen_cfg = causallm::LoadJsonFile(gen_config_path);
    auto nntr_cfg = causallm::LoadJsonFile(nntr_config_path);

    std::cout << "=== Qwen3 Weight-Only QAT Training (Full Model) ==="
              << std::endl;
    std::cout << "Model dir:    " << model_dir << std::endl;
    std::cout << "Train data:   " << train_data_path << std::endl;
    std::cout << "Learning rate: " << lr << std::endl;
    std::cout << "Epochs:       " << epochs << std::endl;
    std::cout << "Output:       " << output_path << std::endl;
    std::cout << "Max samples:  "
              << (max_samples > 0 ? std::to_string(max_samples) : "all")
              << std::endl;
    std::cout << "Skip weights: " << (skip_weights ? "yes" : "no")
              << std::endl;

    // Use QAT-aware model (swaps FC → QAT FC)
    auto model = std::make_unique<causallm::Qwen3QATTransformer>(
      cfg, gen_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Failed to allocate Qwen3QATTransformer" << std::endl;
      return 1;
    }

    model->initializeForTraining(lr, epochs);

    if (!skip_weights) {
      std::string weight_path =
        model_dir + "/" +
        nntr_cfg["model_file_name"].get<std::string>();
      std::cout << "Loading initial weights from: " << weight_path
                << std::endl;
      model->load_weight(weight_path);
    } else {
      std::cout << "Skipping weight loading (random initialization)."
                << std::endl;
    }

    // Setup tokenizer
    std::string tokenizer_path = "";
    if (nntr_cfg.contains("tokenizer_file")) {
      tokenizer_path = nntr_cfg["tokenizer_file"].get<std::string>();
    }
    if (tokenizer_path.empty()) {
      tokenizer_path = model_dir + "/tokenizer.json";
      std::cout << "tokenizer_file not set, using: " << tokenizer_path
                << std::endl;
    }
    auto tokenizer_blob = causallm::LoadBytesFromFile(tokenizer_path);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);

    unsigned int seq_len = nntr_cfg["init_seq_len"].get<unsigned int>();
    unsigned int vocab_size = cfg["vocab_size"].get<unsigned int>();

    causallm::TrainingDataGenerator data_gen(
      tokenizer.get(), seq_len, vocab_size);
    data_gen.loadTextFile(train_data_path);

    if (max_samples > 0 &&
        (unsigned int)max_samples < data_gen.getNumSamples()) {
      std::cout << "Limiting training samples from "
                << data_gen.getNumSamples() << " to " << max_samples
                << std::endl;
      data_gen.limitSamples(max_samples);
    }

    std::cout << "Training samples: " << data_gen.getNumSamples() << std::endl;

    if (data_gen.getNumSamples() == 0) {
      std::cerr << "Error: Not enough training data" << std::endl;
      return 1;
    }

    auto dataset_train = std::shared_ptr<ml::train::Dataset>(
      ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                               causallm::TrainingDataGenerator::dataCb,
                               &data_gen));

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);

    std::cout << "\n=== Starting Qwen3 QAT full model training ===" << std::endl;
    auto train_start = std::chrono::steady_clock::now();

    // // Callbacks
    // auto iter_cb = [](void *user_data) -> bool {
    //   auto m = static_cast<causallm::Qwen3QATTransformer *>(user_data);
    //   std::cout << "  [Step] Training Loss: " << m->getTrainingLoss()
    //             << std::endl;
    //   return false;
    // };

    // auto epoch_cb = [](void *user_data) {
    //   auto m = static_cast<causallm::Qwen3QATTransformer *>(user_data);
    //   std::cout << "[Epoch Done] Training Loss: " << m->getTrainingLoss()
    //             << std::endl;
    // };

    // model->train({}, iter_cb, model.get(), epoch_cb, model.get());
    model->train();

    auto train_end = std::chrono::steady_clock::now();
    double elapsed_sec =
      std::chrono::duration<double>(train_end - train_start).count();

    std::cout << "\nQAT Training completed in " << elapsed_sec << " seconds."
              << std::endl;

    // Memory profile
    std::cout << "\n=== NNTrainer Memory Profile (QAT Training) ==="
              << std::endl;
    PROFILE_END(profiler_listener);
    std::cout << "================================================\n"
              << std::endl;

    try {
      model->save_weight(output_path);
      std::cout << "QAT weights saved to: " << output_path << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Warning: Could not save weights: " << e.what()
                << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
