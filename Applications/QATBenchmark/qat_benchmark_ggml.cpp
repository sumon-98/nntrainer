// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_benchmark_ggml.cpp
 * @brief  Post-QAT Quantized Inference Benchmark — Modes D, E
 *
 * Benchmark comparing GGML-quantized inference pipelines:
 *   Mode D: Q6_K weights (layer1) + Q4_0 (layer2) with GGML SIMD
 *   Mode E: Q4_0 weights (both layers) with GGML SIMD
 *
 * Both modes internally quantize FP32 activations to Q8_K/Q8_0 on-the-fly,
 * so the actual dot product is INT×INT with SIMD acceleration.
 *
 * Also runs Mode A (FP32 baseline) for comparison.
 * Uses random initialization — no training needed.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <tensor.h>
#include <tensor_dim.h>
#include <quantizer.h>

using namespace nntrainer;
using Clock = std::chrono::high_resolution_clock;

static constexpr unsigned int INPUT_DIM = 768;
static constexpr unsigned int HIDDEN_DIM = 256;
static constexpr unsigned int OUTPUT_DIM = 32;
static constexpr unsigned int BATCH = 64;
static constexpr int NUM_ITERS = 1000;

static void fillRandom(Tensor &t, float mean, float stddev, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(mean, stddev);
  float *data = t.getData<float>();
  for (unsigned int i = 0; i < t.size(); ++i)
    data[i] = dist(gen);
}

// ─── Mode A: FP32 baseline ─────────────────────────────────────────────────
static void modeA_fp32(const Tensor &input, const Tensor &W1, const Tensor &b1,
                       const Tensor &W2, const Tensor &b2, Tensor &output) {
  Tensor hidden(TensorDim(BATCH, 1, 1, HIDDEN_DIM));
  input.dot(W1, hidden, false, false);
  hidden.add_i(b1);
  hidden.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, hidden);
  hidden.dot(W2, output, false, false);
  output.add_i(b2);
}

// ─── Mode D: Q6_K (layer1) + Q4_0 (layer2) ─────────────────────────────────
static void modeD_ggml_q6k(const Tensor &input,
                           const Tensor &W1_q6k, const Tensor &b1,
                           const Tensor &W2_q4_0, const Tensor &b2,
                           Tensor &output) {
  Tensor hidden(TensorDim(BATCH, 1, 1, HIDDEN_DIM));
  input.dot(W1_q6k, hidden, false, false);
  hidden.add_i(b1);
  hidden.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, hidden);
  hidden.dot(W2_q4_0, output, false, false);
  output.add_i(b2);
}

// ─── Mode E: Q4_0 for both layers ──────────────────────────────────────────
static void modeE_ggml_q4_0(const Tensor &input,
                            const Tensor &W1_q4_0, const Tensor &b1,
                            const Tensor &W2_q4_0, const Tensor &b2,
                            Tensor &output) {
  Tensor hidden(TensorDim(BATCH, 1, 1, HIDDEN_DIM));
  input.dot(W1_q4_0, hidden, false, false);
  hidden.add_i(b1);
  hidden.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, hidden);
  hidden.dot(W2_q4_0, output, false, false);
  output.add_i(b2);
}

// ─── Accuracy comparison ────────────────────────────────────────────────────
static void compareAccuracy(const Tensor &baseline, const Tensor &test,
                            const char *name) {
  const float *ref = baseline.getData<float>();
  const float *tst = test.getData<float>();
  double mse = 0.0, max_err = 0.0;
  for (unsigned i = 0; i < baseline.size(); ++i) {
    double diff = static_cast<double>(ref[i]) - static_cast<double>(tst[i]);
    mse += diff * diff;
    double ad = std::abs(diff);
    if (ad > max_err) max_err = ad;
  }
  mse /= baseline.size();
  std::cerr << "  " << std::setw(24) << name
            << "  MSE=" << std::scientific << std::setprecision(6) << mse
            << "  MaxErr=" << std::fixed << std::setprecision(6) << max_err
            << std::endl;
}

// ─── Main ───────────────────────────────────────────────────────────────────
int main() {
  std::cerr << "=== QAT Quantized Inference Benchmark (Modes D/E — GGML) ==="
            << std::endl;
  std::cerr << "Config: input=" << INPUT_DIM << " hidden=" << HIDDEN_DIM
            << " output=" << OUTPUT_DIM << " batch=" << BATCH
            << " iters=" << NUM_ITERS << std::endl;

  Tensor W1(TensorDim(1, 1, INPUT_DIM, HIDDEN_DIM));
  Tensor W2(TensorDim(1, 1, HIDDEN_DIM, OUTPUT_DIM));
  Tensor b1(TensorDim(1, 1, 1, HIDDEN_DIM));
  Tensor b2(TensorDim(1, 1, 1, OUTPUT_DIM));
  Tensor input(TensorDim(BATCH, 1, 1, INPUT_DIM));

  fillRandom(W1, 0.0f, 0.05f, 42);
  fillRandom(W2, 0.0f, 0.05f, 43);
  fillRandom(b1, 0.0f, 0.01f, 44);
  fillRandom(b2, 0.0f, 0.01f, 45);
  fillRandom(input, 0.0f, 1.0f, 100);

  // ── Quantize weights to GGML formats ──
  std::cerr << "\n--- Quantizing weights ---" << std::endl;

  auto q6k_quantizer = Quantization::createQuantizer(QScheme::Q6_K);
  Tensor W1_q6k = q6k_quantizer->quantize(W1, Tdatatype::Q6_K);
  std::cerr << "  W1 -> Q6_K: " << W1_q6k.bytes() << " bytes (was "
            << W1.bytes() << " FP32, "
            << std::fixed << std::setprecision(1)
            << (float)W1.bytes() / W1_q6k.bytes() << "x compression)"
            << std::endl;

  auto q4_0_quantizer = Quantization::createQuantizer(QScheme::Q4_0);
  Tensor W2_q4_0 = q4_0_quantizer->quantize(W2, Tdatatype::Q4_0);
  std::cerr << "  W2 -> Q4_0: " << W2_q4_0.bytes() << " bytes (was "
            << W2.bytes() << " FP32, "
            << (float)W2.bytes() / W2_q4_0.bytes() << "x compression)"
            << std::endl;

  Tensor W1_q4_0 = q4_0_quantizer->quantize(W1, Tdatatype::Q4_0);
  std::cerr << "  W1 -> Q4_0: " << W1_q4_0.bytes() << " bytes (was "
            << W1.bytes() << " FP32, "
            << (float)W1.bytes() / W1_q4_0.bytes() << "x compression)"
            << std::endl;

  // ── Memory comparison ──
  size_t fp32_bytes = W1.bytes() + W2.bytes();
  size_t modeD_bytes = W1_q6k.bytes() + W2_q4_0.bytes();
  size_t modeE_bytes = W1_q4_0.bytes() + W2_q4_0.bytes();

  std::cerr << "\n--- Memory (weights only) ---" << std::endl;
  std::cerr << "  Mode A (FP32):       " << fp32_bytes << " bytes" << std::endl;
  std::cerr << "  Mode D (Q6K+Q4_0):   " << modeD_bytes << " bytes — "
            << std::fixed << std::setprecision(2)
            << (float)fp32_bytes / modeD_bytes << "x reduction" << std::endl;
  std::cerr << "  Mode E (Q4_0+Q4_0):  " << modeE_bytes << " bytes — "
            << (float)fp32_bytes / modeE_bytes << "x reduction" << std::endl;

  // ── Compute baseline (Mode A) ──
  Tensor out_A(TensorDim(BATCH, 1, 1, OUTPUT_DIM));
  Tensor out_D(TensorDim(BATCH, 1, 1, OUTPUT_DIM));
  Tensor out_E(TensorDim(BATCH, 1, 1, OUTPUT_DIM));
  modeA_fp32(input, W1, b1, W2, b2, out_A);

  // ── Benchmark ──
  std::cerr << "\n--- Latency (" << NUM_ITERS << " iterations) ---"
            << std::endl;
  {
    auto start = Clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
      Tensor tmp(TensorDim(BATCH, 1, 1, OUTPUT_DIM));
      modeA_fp32(input, W1, b1, W2, b2, tmp);
    }
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cerr << "  Mode A (FP32, baseline): " << std::fixed
              << std::setprecision(2) << ms / NUM_ITERS << " ms/iter"
              << std::endl;
  }
  {
    auto start = Clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
      Tensor tmp(TensorDim(BATCH, 1, 1, OUTPUT_DIM));
      modeD_ggml_q6k(input, W1_q6k, b1, W2_q4_0, b2, tmp);
    }
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cerr << "  Mode D (Q6K SIMD):       " << std::fixed
              << std::setprecision(2) << ms / NUM_ITERS << " ms/iter"
              << std::endl;
    modeD_ggml_q6k(input, W1_q6k, b1, W2_q4_0, b2, out_D);
  }
  {
    auto start = Clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
      Tensor tmp(TensorDim(BATCH, 1, 1, OUTPUT_DIM));
      modeE_ggml_q4_0(input, W1_q4_0, b1, W2_q4_0, b2, tmp);
    }
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cerr << "  Mode E (Q4_0 SIMD):      " << std::fixed
              << std::setprecision(2) << ms / NUM_ITERS << " ms/iter"
              << std::endl;
    modeE_ggml_q4_0(input, W1_q4_0, b1, W2_q4_0, b2, out_E);
  }

  std::cerr << "\n--- Accuracy vs FP32 Baseline ---" << std::endl;
  compareAccuracy(out_A, out_D, "Mode D (Q6K+Q4_0 SIMD)");
  compareAccuracy(out_A, out_E, "Mode E (Q4_0+Q4_0 SIMD)");

  std::cerr << "\n=== GGML Benchmark Complete ===" << std::endl;
  return 0;
}
