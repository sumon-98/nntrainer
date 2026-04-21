// SPDX-License-Identifier: Apache-2.0

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

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <train_data.txt> [--lr <float>] [--epochs <int>]"
                 " [--output <path>] [--max_samples <int>] [--skip_weights]"
              << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string train_data_path = argv[2];
  float lr = 1e-4f;
  unsigned int epochs = 1;
  std::string output_path = "model_weights.bin";
  int max_samples = -1;       // -1 = use all samples
  bool skip_weights = false;  // skip loading pre-trained weights

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
    // Setup built-in NNTrainer memory profiler (active only with -Denable-profile=true)
    auto profiler_listener = std::make_shared<nntrainer::profile::GenericProfileListener>();
    PROFILE_BEGIN(profiler_listener);

    std::string config_path = model_dir + "/config.json";
    std::string gen_config_path = model_dir + "/generation_config.json";
    std::string nntr_config_path = model_dir + "/nntr_config.json";

    auto cfg = causallm::LoadJsonFile(config_path);
    auto gen_cfg = causallm::LoadJsonFile(gen_config_path);
    auto nntr_cfg = causallm::LoadJsonFile(nntr_config_path);

    std::cout << "=== Qwen3 LoRA Training Master ===" << std::endl;
    std::cout << "Model dir: " << model_dir << std::endl;
    std::cout << "Train data: " << train_data_path << std::endl;

    // =========================================================================
    // HARDCODED LORA INJECTION
    // By modifying the JSON object directly in C++ memory BEFORE passing it to
    // the model builder, we programmatically force Qwen3 to use LoRA on all 
    // relevant linear projections!
    // =========================================================================
    std::cout << "\n[LoRA Master] Programmatically injecting LoRA configurations..." << std::endl;
    
    nntr_cfg["lora_rank"] = 8;
    nntr_cfg["lora_alpha"] = 16;
    
    // Qwen3's transformer.cpp uses fc_layer exclusively for:
    // Attention: wq, wk, wv, wo
    // MLP: ffn_up, ffn_down, ffn_gate
    nntr_cfg["lora_target"] = json::array({"wq", "wk", "wv", "wo", "ffn_up", "ffn_down", "ffn_gate"});
    
    std::cout << "[LoRA Master] LoRA Rank & Alpha explicitly forced to: " << nntr_cfg["lora_rank"] << ", " << nntr_cfg["lora_alpha"] << std::endl;
    std::cout << "[LoRA Master] LoRA Targets configured: " << nntr_cfg["lora_target"].dump() << "\n" << std::endl;

    // Use Factory if wanted or initialize Qwen3 directly:
    auto model = std::make_unique<causallm::Qwen3CausalLM>(cfg, gen_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Failed to allocate Qwen3CausalLM" << std::endl;
      return 1;
    }

    model->initializeForTraining(lr, epochs);

    if (!skip_weights && nntr_cfg.contains("model_file_name")) {
      std::string weight_path = model_dir + "/" + nntr_cfg["model_file_name"].get<std::string>();
      std::cout << "Loading initial weights from: " << weight_path << std::endl;
      model->load_weight(weight_path);
    } else {
      std::cout << "Skipping weight loading (using random initialization)." << std::endl;
    }

    std::string tokenizer_path = "";
    if (nntr_cfg.contains("tokenizer_file")) {
      tokenizer_path = nntr_cfg["tokenizer_file"].get<std::string>();
    }
    if (tokenizer_path.empty()) {
      tokenizer_path = model_dir + "/tokenizer.json";
      std::cout << "tokenizer_file not set, using: " << tokenizer_path << std::endl;
    }
    auto tokenizer_blob = causallm::LoadBytesFromFile(tokenizer_path);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);

    unsigned int seq_len = nntr_cfg["init_seq_len"].get<unsigned int>();
    unsigned int vocab_size = cfg["vocab_size"].get<unsigned int>();
    
    causallm::TrainingDataGenerator data_gen(tokenizer.get(), seq_len, vocab_size);
    data_gen.loadTextFile(train_data_path);

    if (max_samples > 0 && (unsigned int)max_samples < data_gen.getNumSamples()) {
      std::cout << "Limiting training samples from " << data_gen.getNumSamples()
                << " to " << max_samples << std::endl;
      data_gen.limitSamples(max_samples);
    }

    if (data_gen.getNumSamples() == 0) {
      std::cerr << "Error: Not enough training data (need > 0 lines)" << std::endl;
      return 1;
    }

    auto dataset_train = std::shared_ptr<ml::train::Dataset>(ml::train::createDataset(
        ml::train::DatasetType::GENERATOR, causallm::TrainingDataGenerator::dataCb, &data_gen));
    
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);

    std::cout << "\n=== Starting Qwen3 LoRA full model training ===" << std::endl;
    auto train_start = std::chrono::steady_clock::now();
    
    std::cout << "\n=== NNTrainer Model Summary (Checking Trainable vs Frozen Layers) ===" << std::endl;
    model->summarize(std::cout, ML_TRAIN_SUMMARY_TENSOR);
    std::cout << "==================================================================\n" << std::endl;

    // For checking if only LoRA layers are being updated or not
    std::cout << "\n=== Saving model weights BEFORE training ===" << std::endl;
    model->exportWeightsToFile("model_weights_before_training_LORA.txt");

    model->train();
    auto train_end = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration<double>(train_end - train_start).count();

    std::cout << "\nTraining completed in " << elapsed_sec << " seconds." << std::endl;
    // For checking if only LoRA layers are being updated or not
    std::cout << "\n=== Saving model weights AFTER training ===" << std::endl;
    model->exportWeightsToFile("model_weights_after_training_LORA.txt");

    model->save_weight(output_path);
    std::cout << "LoRA Weights saved to: " << output_path << std::endl;

    // Print memory profiling report (only produces output with -Denable-profile=true)
    std::cout << "\n=== NNTrainer Memory Profile Report (LoRA Training) ===" << std::endl;
    PROFILE_END(profiler_listener);
    std::cout << "======================================================\n" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
