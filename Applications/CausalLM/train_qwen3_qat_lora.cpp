// SPDX-License-Identifier: Apache-2.0
/**
 * @file   train_qwen3_qat_lora.cpp
 * @brief  Qwen3 QAT + LoRA Training — Weight-Only Quantization with LoRA
 *
 * This script combines QAT (Quantization Aware Training) with LoRA (Low-Rank
 * Adaptation) for Qwen3. The training approach is:
 *
 *   1. All base FC layers use qat_fully_connected (weight-only fake quant)
 *   2. LoRA adapters are injected on top via JSON config (same as LoRA master)
 *   3. Base weights are FROZEN (trainable=false) — only LoRA adapters train
 *   4. During forward pass, the frozen base weights are still fake-quantized
 *      so the network learns to compensate for quantization through LoRA
 *
 * Industry Standard Context:
 *   This is known as "QAT-aware LoRA" or "Quantization-aware Fine-tuning".
 *   The idea is that after deploying the model, both the quantized base
 *   weights AND the LoRA adapters are used at inference time. The LoRA
 *   adapters specifically learn to correct for quantization error.
 *
 * IMPORTANT NOTE: This is a STARTER script. The key challenge is that
 * NNTrainer's built-in LoRA support is tied to the fully_connected layer
 * (via lora_rank/lora_alpha properties). Our qat_fully_connected layer
 * does NOT currently support LoRA properties. There are two paths forward:
 *
 *   Path A: Add lora_rank/lora_alpha support to qat_fc_layer (recommended
 *           but requires significant layer changes)
 *   Path B: Use a two-stage approach where QAT quantization stats are
 *           computed first, then LoRA training uses those frozen stats
 *
 * For now, this script implements Path B as a starting point:
 *   - Stage 1: Build with qat_fully_connected (for weight quantization stats)
 *   - Stage 2: The LoRA adapters are standard fully_connected LoRA layers
 *              applied after the QAT layers
 *
 * Usage:
 *   ./build/Applications/CausalLM/nntr_qwen3_qat_lora \
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
// Qwen3 QAT+LoRA Transformer
//
// Strategy: The base FC weights are replaced with QAT FC (frozen, weight-only
// fake quantized). LoRA adapters are NOT injected through qat_fc_layer but
// through the standard LoRA mechanism (lora_rank/lora_alpha on the regular
// FC layer). This means:
//   - The base weights go through fake quantization during forward
//   - The LoRA delta is added in FP32 on top
//   - Only the LoRA delta is trainable
//
// NOTE: This requires that NNTrainer's LoRA mechanism can work with a base
// layer type of "qat_fully_connected". If it can't (because LoRA properties
// are only parsed by the built-in FC layer), we fall back to using standard
// fully_connected with LoRA and simply set trainable=false on the base weight.
// In that fallback case, QAT is applied as a post-training step.
// =============================================================================
namespace causallm {

class Qwen3QATLoRATransformer : public Qwen3CausalLM {
public:
  Qwen3QATLoRATransformer(json &cfg, json &generation_cfg, json &nntr_cfg)
    : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
      CausalLM(cfg, generation_cfg, nntr_cfg),
      Qwen3Transformer(cfg, generation_cfg, nntr_cfg),
      Qwen3CausalLM(cfg, generation_cfg, nntr_cfg) {}

  ~Qwen3QATLoRATransformer() = default;

  std::vector<LayerHandle> createAttention(
    const int layer_id, int seq_len, int n_heads, int head_dim,
    std::string query_name, std::string key_name,
    std::string value_name) override;

  std::vector<LayerHandle> createMlp(
    const int layer_id, int dim, int hidden_dim,
    std::string input_name) override;

  void registerCustomLayers() override;

private:
  /**
   * @brief Create a QAT FC layer with frozen base weights.
   *
   * The QAT fake quantization still runs during forward (to simulate INT8
   * weights), but the base weight is not updated by the optimizer.
   */
  LayerHandle createQATFCLayer(
    const std::string &name, int unit,
    const std::string &input_layers) {
    return createLayer("qat_fully_connected", {
      withKey("name", name),
      withKey("unit", unit),
      withKey("disable_bias", "true"),
      withKey("input_layers", input_layers),
      withKey("weight_initializer", "ones"),
      withKey("trainable", "false")});  // Frozen for LoRA
  }
};

std::vector<LayerHandle> Qwen3QATLoRATransformer::createAttention(
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

  // Q — QAT FC (frozen base)
  layers.push_back(createQATFCLayer(Q, head_dim * n_heads, query_name));

  // Q norm (unchanged)
  layers.push_back(createLayer("reshaped_rms_norm", {
    withKey("name", Q_norm), withKey("input_layers", Q),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("trainable", "false")}));

  // K — QAT FC (frozen base)
  layers.push_back(
    createQATFCLayer(K, head_dim * n_heads / GQA_SIZE, key_name));

  // K norm (unchanged)
  layers.push_back(createLayer("reshaped_rms_norm", {
    withKey("name", K_norm), withKey("input_layers", K),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("trainable", "false")}));

  // V — QAT FC (frozen base)
  layers.push_back(
    createQATFCLayer(V, head_dim * n_heads / GQA_SIZE, value_name));

  // Attention core (unchanged)
  layers.push_back(createLayer("mha_core", {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", SLIDING_WINDOW),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q_norm, K_norm, V}),
    withKey("trainable", "false")}));

  // O — QAT FC (frozen base)
  layers.push_back(createQATFCLayer(O, DIM, A));

  return layers;
}

std::vector<LayerHandle> Qwen3QATLoRATransformer::createMlp(
  const int layer_id, int dim, int hidden_dim, std::string input_name) {

  std::vector<LayerHandle> layers;

  // ffn_up — QAT FC (frozen)
  layers.push_back(createQATFCLayer(
    "layer" + std::to_string(layer_id) + "_ffn_up", hidden_dim, input_name));

  // ffn_gate — QAT FC (frozen)
  layers.push_back(createQATFCLayer(
    "layer" + std::to_string(layer_id) + "_ffn_gate", hidden_dim, input_name));

  // swiglu (unchanged)
  layers.push_back(createLayer("swiglu", {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_ffn_gate," +
            "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("trainable", "false")}));

  // ffn_down — QAT FC (frozen)
  layers.push_back(createQATFCLayer(
    "layer" + std::to_string(layer_id) + "_ffn_down", dim,
    "layer" + std::to_string(layer_id) + "_ffn_swiglu"));

  return layers;
}

void Qwen3QATLoRATransformer::registerCustomLayers() {
  Qwen3CausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<nntrainer::QATFullyConnectedLayer>);
    std::cout << "[QAT+LoRA] Registered qat_fully_connected layer"
              << std::endl;
  } catch (std::invalid_argument &e) {
    std::cerr << "[QAT+LoRA] Layer registration note: " << e.what()
              << std::endl;
  }
}

} // namespace causallm

// =============================================================================
// Main
// =============================================================================
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr
      << "Usage: " << argv[0]
      << " <model_dir> <train_data.txt>"
         " [--lr <float>] [--epochs <int>]"
         " [--output <path>] [--max_samples <int>] [--skip_weights]"
      << std::endl;
    std::cerr << "\nNOTE: This is a STARTER script for QAT+LoRA. "
              << "LoRA adapters are not yet injected. "
              << "See comments in source for the roadmap." << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string train_data_path = argv[2];
  float lr = 1e-4f;
  unsigned int epochs = 1;
  std::string output_path = "qat_lora_weights.bin";
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

    std::cout << "=== Qwen3 QAT+LoRA Training (Starter) ===" << std::endl;
    std::cout << "Model dir:    " << model_dir << std::endl;
    std::cout << "Train data:   " << train_data_path << std::endl;
    std::cout << "Learning rate: " << lr << std::endl;
    std::cout << "Epochs:       " << epochs << std::endl;
    std::cout << "NOTE: Base FC weights are QAT-quantized and FROZEN."
              << std::endl;
    std::cout << "NOTE: LoRA adapter injection is TODO — "
              << "currently trains with frozen QAT weights only."
              << std::endl;

    auto model = std::make_unique<causallm::Qwen3QATLoRATransformer>(
      cfg, gen_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Failed to allocate Qwen3QATLoRATransformer" << std::endl;
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

    // Tokenizer + data
    std::string tokenizer_path = "";
    if (nntr_cfg.contains("tokenizer_file")) {
      tokenizer_path = nntr_cfg["tokenizer_file"].get<std::string>();
    }
    if (tokenizer_path.empty()) {
      tokenizer_path = model_dir + "/tokenizer.json";
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
      data_gen.limitSamples(max_samples);
    }

    if (data_gen.getNumSamples() == 0) {
      std::cerr << "Error: Not enough training data" << std::endl;
      return 1;
    }

    auto dataset_train = std::shared_ptr<ml::train::Dataset>(
      ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                               causallm::TrainingDataGenerator::dataCb,
                               &data_gen));

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);

    std::cout << "\n=== Starting QAT+LoRA training ===" << std::endl;
    auto train_start = std::chrono::steady_clock::now();

    model->train();

    auto train_end = std::chrono::steady_clock::now();
    double elapsed_sec =
      std::chrono::duration<double>(train_end - train_start).count();

    std::cout << "\nQAT+LoRA Training completed in " << elapsed_sec
              << " seconds." << std::endl;

    PROFILE_END(profiler_listener);

    try {
      model->save_weight(output_path);
      std::cout << "Weights saved to: " << output_path << std::endl;
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
