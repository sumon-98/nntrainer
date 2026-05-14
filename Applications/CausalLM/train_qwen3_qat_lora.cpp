// SPDX-License-Identifier: Apache-2.0
/**
 * @file   train_qwen3_qat_lora.cpp
 * @brief  Qwen3 LoRA QAT Training — QAT on LoRA adapters for INT8 deployment
 *
 * This script trains Qwen3 with LoRA adapters that are fake-quantized (QAT).
 * The key architecture:
 *
 *   1. Base FC weights are FROZEN and used as vanilla FP32 (no quantization)
 *   2. LoRA adapters (A, B) are TRAINABLE and FAKE-QUANTIZED to INT8
 *   3. The LoRA adapters learn to work under INT8 quantization noise
 *   4. At deployment: base weights stay FP32, LoRA adapters go to INT8
 *
 * The math per layer:
 *   output = input * W_frozen
 *          + input * fakeQuant(A) * fakeQuant(B) * (alpha/rank)
 *          + bias
 *
 * Usage:
 *   ./build/Applications/CausalLM/nntr_qwen3_qat_lora \
 *       <model_dir> <train_data.txt> \
 *       [--lr <float>] [--epochs <int>] [--output <path>] \
 *       [--max_samples <int>] [--skip_weights] \
 *       [--lora_rank <int>] [--lora_alpha <int>]
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

// For withKey() and createLayer()
#include <llm_util.hpp>
#include <layer.h>

using json = nlohmann::json;
using ml::train::createLayer;

// =============================================================================
// Qwen3 QAT+LoRA Transformer
// =============================================================================
namespace causallm {

class Qwen3QATLoRATransformer : public Qwen3CausalLM {
public:
  Qwen3QATLoRATransformer(json &cfg, json &generation_cfg, json &nntr_cfg,
                           int lora_rank, int lora_alpha)
    : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
      Qwen3CausalLM(cfg, generation_cfg, nntr_cfg),
      lora_rank_(lora_rank),
      lora_alpha_(lora_alpha) {}

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
  int lora_rank_;
  int lora_alpha_;

  /**
   * @brief Create a QAT FC layer with LoRA enabled.
   * Base weight is frozen; only LoRA adapters are trainable.
   */
  LayerHandle createQATLoRAFCLayer(
    const std::string &name, int unit,
    const std::string &input_layers) {
    std::vector<std::string> params = {
      withKey("name", name),
      withKey("unit", unit),
      withKey("disable_bias", "true"),
      withKey("input_layers", input_layers),
      withKey("weight_initializer", "ones"),
      withKey("lora_rank", lora_rank_),
      withKey("lora_alpha", lora_alpha_)};
    return createLayer("qat_fully_connected", params);
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

  // Q — QAT FC + LoRA
  layers.push_back(createQATLoRAFCLayer(Q, head_dim * n_heads, query_name));

  // Q norm (frozen, no matmul weights)
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm),
    withKey("input_layers", Q),
    withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("trainable", "false")};
  layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

  // K — QAT FC + LoRA
  layers.push_back(
    createQATLoRAFCLayer(K, head_dim * n_heads / GQA_SIZE, key_name));

  // K norm (frozen)
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm),
    withKey("input_layers", K),
    withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("trainable", "false")};
  layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

  // V — QAT FC + LoRA
  layers.push_back(
    createQATLoRAFCLayer(V, head_dim * n_heads / GQA_SIZE, value_name));

  // Attention core (frozen)
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

  // O — QAT FC + LoRA
  layers.push_back(createQATLoRAFCLayer(O, DIM, A));

  return layers;
}

std::vector<LayerHandle> Qwen3QATLoRATransformer::createMlp(
  const int layer_id, int dim, int hidden_dim, std::string input_name) {

  std::vector<LayerHandle> layers;

  // ffn_up — QAT FC + LoRA
  layers.push_back(createQATLoRAFCLayer(
    "layer" + std::to_string(layer_id) + "_ffn_up", hidden_dim, input_name));

  // ffn_gate — QAT FC + LoRA
  layers.push_back(createQATLoRAFCLayer(
    "layer" + std::to_string(layer_id) + "_ffn_gate", hidden_dim, input_name));

  // swiglu (frozen, stateless)
  std::vector<std::string> swiglu_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_ffn_gate," +
            "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("trainable", "false")};
  layers.push_back(createLayer("swiglu", swiglu_params));

  // ffn_down — QAT FC + LoRA
  layers.push_back(createQATLoRAFCLayer(
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
         " [--lora_rank <int>] [--lora_alpha <int>]"
      << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string train_data_path = argv[2];
  float lr = 1e-4f;
  unsigned int epochs = 1;
  std::string output_path = "qat_lora_weights.bin";
  int max_samples = -1;
  bool skip_weights = false;
  int lora_rank = 8;      // Default LoRA rank
  int lora_alpha = 16;    // Default LoRA alpha

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
    } else if (arg == "--lora_rank" && i + 1 < argc) {
      lora_rank = std::atoi(argv[++i]);
    } else if (arg == "--lora_alpha" && i + 1 < argc) {
      lora_alpha = std::atoi(argv[++i]);
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

    std::cout << "=== Qwen3 QAT + LoRA Training ===" << std::endl;
    std::cout << "Model dir:    " << model_dir << std::endl;
    std::cout << "Train data:   " << train_data_path << std::endl;
    std::cout << "Learning rate: " << lr << std::endl;
    std::cout << "Epochs:       " << epochs << std::endl;
    std::cout << "LoRA rank:    " << lora_rank << std::endl;
    std::cout << "LoRA alpha:   " << lora_alpha << std::endl;
    std::cout << "LoRA scaling: " << (float)lora_alpha / lora_rank << std::endl;
    std::cout << "Output:       " << output_path << std::endl;
    std::cout << "Max samples:  "
              << (max_samples > 0 ? std::to_string(max_samples) : "all")
              << std::endl;
    std::cout << "Skip weights: " << (skip_weights ? "yes" : "no")
              << std::endl;
    std::cout << "\nBase FC weights: FROZEN, vanilla FP32 (no quantization)"
              << std::endl;
    std::cout << "LoRA adapters:   TRAINABLE + FAKE-QUANTIZED (INT8 simulation)"
              << std::endl;

    // =========================================================================
    // CRITICAL: Inject lora_rank/lora_alpha into nntr_cfg BEFORE constructing
    // the model. The base Transformer::setupParameters() reads these from JSON
    // to set LORA_RANK > 0, which triggers freezing of ALL non-FC layers:
    //   - embedding0          (Transformer::constructModel)
    //   - output_norm         (Transformer::constructModel)
    //   - lm_head             (Transformer::constructModel)
    //   - attention_norm      (Transformer::createTransformerDecoderBlock)
    //   - decoder_add         (Transformer::createTransformerDecoderBlock)
    //   - ffn_norm            (Transformer::createTransformerDecoderBlock)
    //   - decoder_output      (Transformer::createTransformerDecoderBlock)
    //
    // Without this, LORA_RANK == 0 and those layers remain trainable!
    // This follows the same pattern as train_qwen3_lora_master.cpp.
    // =========================================================================
    nntr_cfg["lora_rank"] = lora_rank;
    nntr_cfg["lora_alpha"] = lora_alpha;
    // We don't set lora_target here because our QAT+LoRA subclass overrides
    // createAttention/createMlp directly — the base class isLoRATarget() is
    // not used for layer type selection in our overrides. But LORA_RANK > 0
    // is still needed to trigger the freezing guards.

    auto model = std::make_unique<causallm::Qwen3QATLoRATransformer>(
      cfg, gen_cfg, nntr_cfg, lora_rank, lora_alpha);
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

    std::cout << "\n=== Starting QAT+LoRA training ===" << std::endl;
    auto train_start = std::chrono::steady_clock::now();

    model->train();

    auto train_end = std::chrono::steady_clock::now();
    double elapsed_sec =
      std::chrono::duration<double>(train_end - train_start).count();

    std::cout << "\nQAT+LoRA Training completed in " << elapsed_sec
              << " seconds." << std::endl;

    std::cout << "\n=== NNTrainer Memory Profile (QAT+LoRA) ===" << std::endl;
    PROFILE_END(profiler_listener);
    std::cout << "=============================================\n" << std::endl;

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
