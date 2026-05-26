// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qwen3_quantized_inference.cpp
 * @brief  Qwen3 Quantized Inference POC
 *
 * Demonstrates quantized Qwen3 0.6B inference using NNTrainer.
 *
 * Two modes:
 *   Mode 1: FP32 baseline inference (model_tensor_type = "FP32-FP32")
 *   Mode 2: Q4_0 quantized inference (model_tensor_type = "Q4_0-FP32")
 *
 * The Q4_0 path loads the same FP32 .bin weight file but quantizes FC
 * layer weights on-the-fly during model->load(). Embedding and LM-head
 * stay in the dtype specified by embedding_dtype / lmhead_dtype.
 *
 * Usage:
 *   ./build/Applications/CausalLM/nntr_qwen3_quant_infer \
 *       <model_dir> [test_prompt] [--q4_0] [--q4_0_fp16] [--both] \
 *       [--max_tokens <N>] [--sst2 <path_to_sst2_train.txt>]
 *
 * Example (FP32 only):
 *   ./nntr_qwen3_quant_infer ./res/qwen3/qwen3-0.6b
 *
 * Example (Q4_0 only):
 *   ./nntr_qwen3_quant_infer ./res/qwen3/qwen3-0.6b --q4_0
 *
 * Example (both, compare):
 *   ./nntr_qwen3_quant_infer ./res/qwen3/qwen3-0.6b --both
 *
 * Example (SST2 eval):
 *   ./nntr_qwen3_quant_infer ./res/qwen3/qwen3-0.6b --both \
 *       --sst2 ./sst2_data/val.txt --max_tokens 32
 */

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <factory.h>

#include "json.hpp"
#include "qwen3_causallm.h"
#include "transformer.h"

using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;

// ─── SST2 sample reader ────────────────────────────────────────────────────
struct SST2Sample {
  std::string sentence;
  std::string label; // "Positive" or "Negative"
};

static std::vector<SST2Sample> loadSST2(const std::string &path,
                                        int max_samples = 20) {
  std::vector<SST2Sample> samples;
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "Warning: Could not open SST2 file: " << path << std::endl;
    return samples;
  }
  std::string line;
  while (std::getline(f, line) &&
         (max_samples <= 0 ||
          (int)samples.size() < max_samples)) {
    // Format: "Sentence: <text> Sentiment: <label>"
    auto sent_pos = line.find("Sentence: ");
    auto label_pos = line.find(" Sentiment: ");
    if (sent_pos != std::string::npos && label_pos != std::string::npos) {
      SST2Sample s;
      s.sentence = line.substr(sent_pos + 10,
                               label_pos - (sent_pos + 10));
      s.label = line.substr(label_pos + 12);
      samples.push_back(s);
    }
  }
  return samples;
}

// ─── Run inference with a specific model_tensor_type ────────────────────────
struct InferenceResult {
  std::string output_text;
  double latency_ms;
  bool success;
};

static InferenceResult runInference(
    const std::string &model_dir,
    const std::string &tensor_type, // e.g. "FP32-FP32" or "Q4_0-FP32"
    const std::string &prompt,
    unsigned int max_tokens) {

  InferenceResult result;
  result.success = false;

  try {
    json cfg = causallm::LoadJsonFile(model_dir + "/config.json");
    json gen_cfg =
        causallm::LoadJsonFile(model_dir + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_dir + "/nntr_config.json");

    // Override tensor type and generation params
    nntr_cfg["model_tensor_type"] = tensor_type;
    if (max_tokens > 0) {
      nntr_cfg["num_to_generate"] = max_tokens;
    }

    // Get weight file path
    std::string weight_file =
        model_dir + "/" + nntr_cfg["model_file_name"].get<std::string>();

    // Create model
    auto model = std::make_unique<causallm::Qwen3CausalLM>(
        cfg, gen_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Failed to create Qwen3CausalLM" << std::endl;
      return result;
    }

    // Initialize for inference
    model->initialize();

    // Load weights (quantization happens here for Q4_0)
    std::cerr << "  Loading weights (" << tensor_type << ")..." << std::flush;
    auto load_start = Clock::now();
    model->load_weight(weight_file);
    auto load_end = Clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(
                         load_end - load_start)
                         .count();
    std::cerr << " done (" << (int)load_ms << " ms)" << std::endl;

    // Run inference
    std::cerr << "  Running inference..." << std::endl;
    auto infer_start = Clock::now();
    model->run(prompt, false, "", "", true);
    auto infer_end = Clock::now();

    result.latency_ms =
        std::chrono::duration<double, std::milli>(infer_end - infer_start)
            .count();
    result.output_text = model->getOutput(0);
    result.success = true;

  } catch (const std::exception &e) {
    std::cerr << "  ERROR: " << e.what() << std::endl;
    result.output_text = "[ERROR] " + std::string(e.what());
  }

  return result;
}

// ─── Main ───────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cerr
        << "Usage: " << argv[0]
        << " <model_dir> [prompt] [--q4_0] [--q4_0_fp16] [--both]"
           " [--max_tokens <N>] [--sst2 <path>] [--sst2_samples <N>]"
        << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string custom_prompt = "";
  bool run_fp32 = true;
  bool run_q4_0 = false;
  bool run_q4_0_fp16 = false;
  unsigned int max_tokens = 0; // 0 = use config default
  std::string sst2_path = "";
  int sst2_samples = 10;

  // Parse args
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--q4_0") {
      run_q4_0 = true;
      run_fp32 = false;
    } else if (arg == "--q4_0_fp16") {
      run_q4_0_fp16 = true;
      run_fp32 = false;
    } else if (arg == "--both") {
      run_fp32 = true;
      run_q4_0 = true;
    } else if (arg == "--max_tokens" && i + 1 < argc) {
      max_tokens = std::atoi(argv[++i]);
    } else if (arg == "--sst2" && i + 1 < argc) {
      sst2_path = argv[++i];
    } else if (arg == "--sst2_samples" && i + 1 < argc) {
      sst2_samples = std::atoi(argv[++i]);
    } else if (arg[0] != '-') {
      custom_prompt = arg;
    }
  }

  // Build prompt
  std::string prompt;
  if (!custom_prompt.empty()) {
    prompt = "<|im_start|>user\n" + custom_prompt +
             "<|im_end|>\n<|im_start|>assistant\n";
  } else {
    // Read from nntr_config's sample_input
    try {
      json nntr_cfg =
          causallm::LoadJsonFile(model_dir + "/nntr_config.json");
      prompt = nntr_cfg["sample_input"].get<std::string>();
    } catch (...) {
      prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    }
  }

  if (max_tokens == 0) max_tokens = 64; // sensible default for benchmarking

  std::cout << "╔══════════════════════════════════════════════════════╗"
            << std::endl;
  std::cout << "║    Qwen3 Quantized Inference Comparison POC         ║"
            << std::endl;
  std::cout << "╠══════════════════════════════════════════════════════╣"
            << std::endl;
  std::cout << "║ Model dir:  " << model_dir << std::endl;
  std::cout << "║ Max tokens: " << max_tokens << std::endl;
  std::cout << "║ Modes:      ";
  if (run_fp32) std::cout << "FP32 ";
  if (run_q4_0) std::cout << "Q4_0 ";
  if (run_q4_0_fp16) std::cout << "Q4_0-FP16 ";
  std::cout << std::endl;
  std::cout << "╚══════════════════════════════════════════════════════╝"
            << std::endl;

  // ─── SST2 evaluation mode ───────────────────────────────────────────
  if (!sst2_path.empty()) {
    auto samples = loadSST2(sst2_path, sst2_samples);
    if (samples.empty()) {
      std::cerr << "No SST2 samples loaded!" << std::endl;
      return 1;
    }
    std::cout << "\n=== SST2 Sentiment Classification ("
              << samples.size() << " samples) ===\n"
              << std::endl;

    // We'll run each mode and count correct predictions
    struct ModeResult {
      std::string name;
      std::string tensor_type;
      int correct;
      double total_latency_ms;
    };

    std::vector<ModeResult> modes;
    if (run_fp32)
      modes.push_back({"FP32", "FP32-FP32", 0, 0.0});
    if (run_q4_0)
      modes.push_back({"Q4_0", "Q4_0-FP32", 0, 0.0});
    if (run_q4_0_fp16)
      modes.push_back({"Q4_0-FP16", "Q4_0-FP16", 0, 0.0});

    for (auto &mode : modes) {
      std::cout << "\n--- Mode: " << mode.name << " (" << mode.tensor_type
                << ") ---" << std::endl;

      for (size_t i = 0; i < samples.size(); ++i) {
        std::string sst2_prompt =
            "<|im_start|>user\nClassify the sentiment of the following "
            "sentence as Positive or Negative. Reply with only one word.\n"
            "Sentence: " +
            samples[i].sentence +
            "<|im_end|>\n<|im_start|>assistant\n";

        auto res = runInference(model_dir, mode.tensor_type,
                                sst2_prompt, 8);
        if (res.success) {
          mode.total_latency_ms += res.latency_ms;

          // Check if output contains the correct label
          std::string out_lower = res.output_text;
          std::transform(out_lower.begin(), out_lower.end(),
                         out_lower.begin(), ::tolower);
          std::string label_lower = samples[i].label;
          std::transform(label_lower.begin(), label_lower.end(),
                         label_lower.begin(), ::tolower);
          bool correct =
              out_lower.find(label_lower) != std::string::npos;
          if (correct) mode.correct++;

          std::cout << "  [" << (i + 1) << "/" << samples.size() << "] "
                    << (correct ? "✓" : "✗") << " Expected: "
                    << samples[i].label << " Got: \""
                    << res.output_text.substr(0, 50) << "\""
                    << " (" << (int)res.latency_ms << " ms)"
                    << std::endl;
        }
      }
    }

    // Print summary
    std::cout << "\n╔══════════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║        SST2 Results Summary                  ║"
              << std::endl;
    std::cout << "╠═════════════╦══════════╦═══════════════════════╣"
              << std::endl;
    std::cout << "║ Mode        ║ Accuracy ║ Avg Latency/sample    ║"
              << std::endl;
    std::cout << "╠═════════════╬══════════╬═══════════════════════╣"
              << std::endl;
    for (auto &mode : modes) {
      double acc = 100.0 * mode.correct / samples.size();
      double avg_lat = mode.total_latency_ms / samples.size();
      printf("║ %-11s ║ %5.1f %% ║ %8.0f ms           ║\n",
             mode.name.c_str(), acc, avg_lat);
    }
    std::cout << "╚═════════════╩══════════╩═══════════════════════╝"
              << std::endl;

    return 0;
  }

  // ─── Single-prompt mode ─────────────────────────────────────────────
  std::cout << "\nPrompt: " << prompt.substr(0, 80)
            << (prompt.size() > 80 ? "..." : "") << "\n"
            << std::endl;

  if (run_fp32) {
    std::cout << "═══════════════════════════════════════════════"
              << std::endl;
    std::cout << "Mode 1: FP32 Baseline" << std::endl;
    std::cout << "═══════════════════════════════════════════════"
              << std::endl;
    auto res =
        runInference(model_dir, "FP32-FP32", prompt, max_tokens);
    std::cout << "\n  Output: " << res.output_text << std::endl;
    std::cout << "  Latency: " << (int)res.latency_ms << " ms"
              << std::endl;
  }

  if (run_q4_0) {
    std::cout << "\n═══════════════════════════════════════════════"
              << std::endl;
    std::cout << "Mode 2: Q4_0 Quantized" << std::endl;
    std::cout << "═══════════════════════════════════════════════"
              << std::endl;
    auto res =
        runInference(model_dir, "Q4_0-FP32", prompt, max_tokens);
    std::cout << "\n  Output: " << res.output_text << std::endl;
    std::cout << "  Latency: " << (int)res.latency_ms << " ms"
              << std::endl;
  }

  if (run_q4_0_fp16) {
    std::cout << "\n═══════════════════════════════════════════════"
              << std::endl;
    std::cout << "Mode 3: Q4_0 + FP16 Activations" << std::endl;
    std::cout << "═══════════════════════════════════════════════"
              << std::endl;
    auto res =
        runInference(model_dir, "Q4_0-FP16", prompt, max_tokens);
    std::cout << "\n  Output: " << res.output_text << std::endl;
    std::cout << "  Latency: " << (int)res.latency_ms << " ms"
              << std::endl;
  }

  std::cout << "\n=== Done ===" << std::endl;
  return 0;
}
