// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_benchmark.cpp
 * @brief  Post-QAT Quantized Inference Benchmark — Modes A, B, C
 *
 * Standalone benchmark comparing three inference pipelines:
 *   Mode A: FP32 weights × FP32 activations (baseline, uses BLAS sgemm)
 *   Mode B: INT8 weights (dequant to FP32) × FP32 activations (BLAS sgemm)
 *   Mode C: INT8 weights × INT8 activations → INT32 accumulate (naive loop)
 *
 * Uses random initialization with synthetic QAT statistics.
 * No NNTrainer model/training needed — operates directly on Tensor API.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <tensor.h>
#include <tensor_dim.h>

using namespace nntrainer;
using Clock = std::chrono::high_resolution_clock;

// ─── Configuration ──────────────────────────────────────────────────────────
static constexpr unsigned int INPUT_DIM = 768;   // multiple of 256 for GGML
static constexpr unsigned int HIDDEN_DIM = 128;  // multiple of 32
static constexpr unsigned int OUTPUT_DIM = 10;
static constexpr unsigned int BATCH = 64;
static constexpr int NUM_ITERS = 1000;

// ─── Per-tensor-affine INT8 quantization helpers ────────────────────────────

struct QuantParams {
  float scale;
  float zero_point;
  float q_min;
  float q_max;
};

/// Compute per-tensor-affine quantization parameters from min/max
static QuantParams computeQuantParams(float min_val, float max_val) {
  QuantParams p;
  p.q_min = -128.0f;
  p.q_max = 127.0f;
  float range = max_val - min_val;
  if (range < 1e-8f) range = 1e-8f;
  p.scale = range / (p.q_max - p.q_min);
  p.zero_point = p.q_min - std::round(min_val / p.scale);
  p.zero_point = std::max(p.q_min, std::min(p.q_max, p.zero_point));
  return p;
}

/// Quantize a float value to int8 using given params
static inline int8_t quantize_val(float v, const QuantParams &p) {
  float q = std::round(v / p.scale + p.zero_point);
  q = std::max(p.q_min, std::min(p.q_max, q));
  return static_cast<int8_t>(q);
}

/// Dequantize an int8 value back to float
static inline float dequantize_val(int8_t q, const QuantParams &p) {
  return p.scale * (static_cast<float>(q) - p.zero_point);
}

// ─── Utility: fill tensor with random normal values ─────────────────────────
static void fillRandom(Tensor &t, float mean, float stddev, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(mean, stddev);
  float *data = t.getData<float>();
  for (unsigned int i = 0; i < t.size(); ++i)
    data[i] = dist(gen);
}

// ─── Utility: compute statistics ────────────────────────────────────────────
static void getMinMax(const Tensor &t, float &out_min, float &out_max) {
  const float *d = t.getData<float>();
  out_min = d[0];
  out_max = d[0];
  for (unsigned int i = 1; i < t.size(); ++i) {
    if (d[i] < out_min) out_min = d[i];
    if (d[i] > out_max) out_max = d[i];
  }
}

// ─── Mode A: FP32 × FP32 (baseline) ────────────────────────────────────────
static void modeA_fp32(const Tensor &input, const Tensor &W1, const Tensor &b1,
                       const Tensor &W2, const Tensor &b2, Tensor &output) {
  // Layer 1: hidden = input * W1 + b1, then ReLU
  Tensor hidden({BATCH, 1, 1, HIDDEN_DIM});
  input.dot(W1, hidden, false, false);
  hidden.add_i(b1);
  hidden.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, hidden);

  // Layer 2: output = hidden * W2 + b2
  hidden.dot(W2, output, false, false);
  output.add_i(b2);
}

// ─── Mode B: Dequantized-INT8 weights × FP32 activations ───────────────────
static void modeB_dequant_fp32(const Tensor &input,
                               const std::vector<int8_t> &W1q, QuantParams W1p,
                               const Tensor &b1,
                               const std::vector<int8_t> &W2q, QuantParams W2p,
                               const Tensor &b2, Tensor &output) {
  // Dequantize W1 to FP32
  Tensor W1_deq({1, 1, INPUT_DIM, HIDDEN_DIM});
  float *w1d = W1_deq.getData<float>();
  for (size_t i = 0; i < W1q.size(); ++i)
    w1d[i] = dequantize_val(W1q[i], W1p);

  // Layer 1
  Tensor hidden({BATCH, 1, 1, HIDDEN_DIM});
  input.dot(W1_deq, hidden, false, false);
  hidden.add_i(b1);
  hidden.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, hidden);

  // Dequantize W2 to FP32
  Tensor W2_deq({1, 1, HIDDEN_DIM, OUTPUT_DIM});
  float *w2d = W2_deq.getData<float>();
  for (size_t i = 0; i < W2q.size(); ++i)
    w2d[i] = dequantize_val(W2q[i], W2p);

  // Layer 2
  hidden.dot(W2_deq, output, false, false);
  output.add_i(b2);
}

// ─── Mode C: INT8 × INT8 → INT32 (naive matmul) ────────────────────────────

/// Naive INT8 matmul: C[M×N] = A[M×K] * B[K×N], all in int32 accumulator
static void int8_matmul(const int8_t *A, const int8_t *B, int32_t *C,
                        unsigned M, unsigned K, unsigned N) {
  for (unsigned i = 0; i < M; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (unsigned k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(A[i * K + k]) *
               static_cast<int32_t>(B[k * N + j]);
      }
      C[i * N + j] = acc;
    }
  }
}

static void modeC_int8(const Tensor &input, QuantParams act1_p,
                       const std::vector<int8_t> &W1q, QuantParams W1p,
                       const Tensor &b1,
                       QuantParams act2_p,
                       const std::vector<int8_t> &W2q, QuantParams W2p,
                       const Tensor &b2, Tensor &output) {
  const float *in_fp = input.getData<float>();
  unsigned total1 = BATCH * INPUT_DIM;

  // Quantize activations for layer 1
  std::vector<int8_t> act1_q(total1);
  for (unsigned i = 0; i < total1; ++i)
    act1_q[i] = quantize_val(in_fp[i], act1_p);

  // INT8 matmul: act1_q[BATCH×INPUT_DIM] * W1q[INPUT_DIM×HIDDEN_DIM]
  std::vector<int32_t> acc1(BATCH * HIDDEN_DIM);
  int8_matmul(act1_q.data(), W1q.data(), acc1.data(),
              BATCH, INPUT_DIM, HIDDEN_DIM);

  // Rescale to FP32 + bias + ReLU
  float combined_scale1 = act1_p.scale * W1p.scale;
  Tensor hidden({BATCH, 1, 1, HIDDEN_DIM});
  float *hid = hidden.getData<float>();
  const float *b1d = b1.getData<float>();
  for (unsigned i = 0; i < BATCH; ++i) {
    for (unsigned j = 0; j < HIDDEN_DIM; ++j) {
      // Simplified: ignoring zero_point cross-terms for benchmark
      float val = static_cast<float>(acc1[i * HIDDEN_DIM + j]) *
                  combined_scale1 + b1d[j];
      hid[i * HIDDEN_DIM + j] = val > 0.f ? val : 0.f; // ReLU
    }
  }

  // Quantize hidden activations for layer 2
  unsigned total2 = BATCH * HIDDEN_DIM;
  std::vector<int8_t> act2_q(total2);
  for (unsigned i = 0; i < total2; ++i)
    act2_q[i] = quantize_val(hid[i], act2_p);

  // INT8 matmul: act2_q[BATCH×HIDDEN_DIM] * W2q[HIDDEN_DIM×OUTPUT_DIM]
  std::vector<int32_t> acc2(BATCH * OUTPUT_DIM);
  int8_matmul(act2_q.data(), W2q.data(), acc2.data(),
              BATCH, HIDDEN_DIM, OUTPUT_DIM);

  // Rescale to FP32 + bias
  float combined_scale2 = act2_p.scale * W2p.scale;
  float *out = output.getData<float>();
  const float *b2d = b2.getData<float>();
  for (unsigned i = 0; i < BATCH; ++i)
    for (unsigned j = 0; j < OUTPUT_DIM; ++j)
      out[i * OUTPUT_DIM + j] =
        static_cast<float>(acc2[i * OUTPUT_DIM + j]) * combined_scale2 + b2d[j];
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
  std::cerr << "  " << std::setw(20) << name
            << "  MSE=" << std::scientific << std::setprecision(6) << mse
            << "  MaxErr=" << std::fixed << std::setprecision(6) << max_err
            << std::endl;
}

// ─── Main ───────────────────────────────────────────────────────────────────
int main() {
  std::cerr << "=== QAT Quantized Inference Benchmark (Modes A/B/C) ==="
            << std::endl;
  std::cerr << "Config: input=" << INPUT_DIM << " hidden=" << HIDDEN_DIM
            << " output=" << OUTPUT_DIM << " batch=" << BATCH
            << " iters=" << NUM_ITERS << std::endl;

  // ── Create random weights and input ──
  Tensor W1({1, 1, INPUT_DIM, HIDDEN_DIM});
  Tensor W2({1, 1, HIDDEN_DIM, OUTPUT_DIM});
  Tensor b1({1, 1, 1, HIDDEN_DIM});
  Tensor b2({1, 1, 1, OUTPUT_DIM});
  Tensor input({BATCH, 1, 1, INPUT_DIM});

  fillRandom(W1, 0.0f, 0.05f, 42);
  fillRandom(W2, 0.0f, 0.05f, 43);
  fillRandom(b1, 0.0f, 0.01f, 44);
  fillRandom(b2, 0.0f, 0.01f, 45);
  fillRandom(input, 0.0f, 1.0f, 100);

  // ── Compute quantization parameters (simulated QAT stats) ──
  float w1_min, w1_max, w2_min, w2_max;
  getMinMax(W1, w1_min, w1_max);
  getMinMax(W2, w2_min, w2_max);
  QuantParams W1p = computeQuantParams(w1_min, w1_max);
  QuantParams W2p = computeQuantParams(w2_min, w2_max);

  // Activation stats (use input range for layer1, estimate for layer2)
  float in_min, in_max;
  getMinMax(input, in_min, in_max);
  QuantParams act1_p = computeQuantParams(in_min, in_max);
  // For layer 2 activations (post-ReLU): min=0
  QuantParams act2_p = computeQuantParams(0.0f, 5.0f);

  std::cerr << "\n--- Quantization Parameters ---" << std::endl;
  std::cerr << "  W1: scale=" << W1p.scale << " zp=" << W1p.zero_point
            << " range=[" << w1_min << ", " << w1_max << "]" << std::endl;
  std::cerr << "  W2: scale=" << W2p.scale << " zp=" << W2p.zero_point
            << " range=[" << w2_min << ", " << w2_max << "]" << std::endl;

  // ── Quantize weights to INT8 ──
  std::vector<int8_t> W1q(INPUT_DIM * HIDDEN_DIM);
  std::vector<int8_t> W2q(HIDDEN_DIM * OUTPUT_DIM);
  {
    const float *w1f = W1.getData<float>();
    for (size_t i = 0; i < W1q.size(); ++i)
      W1q[i] = quantize_val(w1f[i], W1p);
    const float *w2f = W2.getData<float>();
    for (size_t i = 0; i < W2q.size(); ++i)
      W2q[i] = quantize_val(w2f[i], W2p);
  }

  // ── Memory comparison ──
  size_t fp32_bytes = (INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM) * 4;
  size_t int8_bytes = (INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM) * 1
                      + 2 * sizeof(QuantParams);
  std::cerr << "\n--- Memory (weights only) ---" << std::endl;
  std::cerr << "  Mode A (FP32):  " << fp32_bytes << " bytes" << std::endl;
  std::cerr << "  Mode B (INT8):  " << int8_bytes << " bytes — "
            << std::fixed << std::setprecision(2)
            << (float)fp32_bytes / int8_bytes << "x reduction" << std::endl;
  std::cerr << "  Mode C (INT8):  " << int8_bytes << " bytes — "
            << (float)fp32_bytes / int8_bytes << "x reduction" << std::endl;

  // ── Compute baseline output (Mode A, single pass) for accuracy ──
  Tensor out_A({BATCH, 1, 1, OUTPUT_DIM});
  Tensor out_B({BATCH, 1, 1, OUTPUT_DIM});
  Tensor out_C({BATCH, 1, 1, OUTPUT_DIM});
  modeA_fp32(input, W1, b1, W2, b2, out_A);

  // ── Benchmark Mode A ──
  std::cerr << "\n--- Latency (" << NUM_ITERS << " iterations) ---"
            << std::endl;
  {
    auto start = Clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
      Tensor tmp({BATCH, 1, 1, OUTPUT_DIM});
      modeA_fp32(input, W1, b1, W2, b2, tmp);
    }
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cerr << "  Mode A (FP32xFP32):   " << std::fixed
              << std::setprecision(2) << ms / NUM_ITERS << " ms/iter  (total "
              << ms << " ms)" << std::endl;
  }

  // ── Benchmark Mode B ──
  {
    auto start = Clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
      Tensor tmp({BATCH, 1, 1, OUTPUT_DIM});
      modeB_dequant_fp32(input, W1q, W1p, b1, W2q, W2p, b2, tmp);
    }
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cerr << "  Mode B (FP32xINT8):   " << std::fixed
              << std::setprecision(2) << ms / NUM_ITERS << " ms/iter  (total "
              << ms << " ms)" << std::endl;
    modeB_dequant_fp32(input, W1q, W1p, b1, W2q, W2p, b2, out_B);
  }

  // ── Benchmark Mode C ──
  {
    auto start = Clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
      Tensor tmp({BATCH, 1, 1, OUTPUT_DIM});
      modeC_int8(input, act1_p, W1q, W1p, b1, act2_p, W2q, W2p, b2, tmp);
    }
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cerr << "  Mode C (INT8xINT8):   " << std::fixed
              << std::setprecision(2) << ms / NUM_ITERS << " ms/iter  (total "
              << ms << " ms)" << std::endl;
    modeC_int8(input, act1_p, W1q, W1p, b1, act2_p, W2q, W2p, b2, out_C);
  }

  // ── Accuracy comparison ──
  std::cerr << "\n--- Accuracy vs FP32 Baseline ---" << std::endl;
  compareAccuracy(out_A, out_B, "Mode B (FP32xINT8)");
  compareAccuracy(out_A, out_C, "Mode C (INT8xINT8)");

  std::cerr << "\n=== Benchmark Complete ===" << std::endl;
  return 0;
}
