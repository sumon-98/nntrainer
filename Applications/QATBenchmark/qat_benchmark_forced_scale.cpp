// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_benchmark_forced_scale.cpp
 * @brief  Forced-scale GGML benchmark: our per-tensor scale in Q6_K blocks
 *
 * Constructs Q6_K blocks with a forced global scale (simulating our QAT
 * per-tensor scale stuffed into GGML format), then runs SIMD matmul.
 * Compares latency and accuracy vs normal GGML Q6_K (per-block scales).
 *
 * Q6_K is chosen because it is NOT repacked (unlike Q4_0 which gets
 * repacked to q4_0x4), so we can safely overwrite block contents.
 *
 * Only benchmarks Layer 1 (768->128) since 768%256=0 is Q6_K compatible.
 */

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <tensor.h>
#include <tensor_dim.h>
#include <quantizer.h>

using namespace nntrainer;
using Clock = std::chrono::high_resolution_clock;

// ─── Local block_q6_K definition (matches nntr_ggml_impl_common.h) ─────────
#define QK_K_LOCAL 256
#pragma pack(push, 1)
struct block_q6_K_local {
  uint8_t ql[QK_K_LOCAL / 2];     // lower 4 bits of 256 quants (128 bytes)
  uint8_t qh[QK_K_LOCAL / 4];     // upper 2 bits of 256 quants (64 bytes)
  int8_t scales[QK_K_LOCAL / 16]; // 16 sub-block scales (16 bytes)
  uint16_t d;                     // super-block scale as FP16 (2 bytes)
};                                // Total: 210 bytes
#pragma pack(pop)
static_assert(sizeof(block_q6_K_local) == 210, "block_q6_K_local must be 210 bytes");

// ─── FP16 conversion (IEEE 754 half-precision) ─────────────────────────────
static uint16_t fp32_to_fp16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000;
  int exp = ((u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = u & 0x7FFFFF;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7C00;
  return sign | (exp << 10) | (mant >> 13);
}

static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  int exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0) {
    if (mant == 0) { float r; uint32_t z = sign; memcpy(&r, &z, 4); return r; }
    exp = 1;
    while (!(mant & 0x400)) { mant <<= 1; exp--; }
    mant &= 0x3FF;
  } else if (exp == 31) {
    uint32_t r = sign | 0x7F800000 | (mant << 13);
    float f;
    memcpy(&f, &r, 4);
    return f;
  }
  uint32_t result = sign | ((exp + 112) << 23) | (mant << 13);
  float f;
  memcpy(&f, &result, 4);
  return f;
}

// ─── Configuration ──────────────────────────────────────────────────────────
static constexpr unsigned int INPUT_DIM = 768;
static constexpr unsigned int HIDDEN_DIM = 256;
static constexpr unsigned int BATCH = 64;
static constexpr int NUM_ITERS = 1000;

static void fillRandom(Tensor &t, float mean, float stddev, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(mean, stddev);
  float *data = t.getData<float>();
  for (unsigned i = 0; i < t.size(); ++i) data[i] = dist(gen);
}

// ─── Pack 256 quantized values into Q6_K block ql/qh fields ────────────────
static void pack_q6k_block(block_q6_K_local *block, const uint8_t *L) {
  uint8_t *ql = block->ql;
  uint8_t *qh = block->qh;
  for (int j = 0; j < QK_K_LOCAL; j += 128) {
    for (int l = 0; l < 32; ++l) {
      const uint8_t q1 = L[j + l + 0] & 0xF;
      const uint8_t q2 = L[j + l + 32] & 0xF;
      const uint8_t q3 = L[j + l + 64] & 0xF;
      const uint8_t q4 = L[j + l + 96] & 0xF;
      ql[l + 0] = q1 | (q3 << 4);
      ql[l + 32] = q2 | (q4 << 4);
      qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
              ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
    }
    ql += 64;
    qh += 32;
  }
}

// ─── Build forced-scale Q6_K tensor ─────────────────────────────────────────
static Tensor buildForcedQ6K(const Tensor &W_fp32) {
  unsigned int K = W_fp32.getDim().height();  // INPUT_DIM = 768
  unsigned int N = W_fp32.getDim().width();   // HIDDEN_DIM = 128

  // Transpose (same as GgmlQuantizer does)
  Tensor W_t = W_fp32.transpose("0:2:1"); // [N x K] = [128 x 768]
  const float *src = W_t.getData<float>();
  size_t total = K * N;

  // Compute global symmetric scale
  float amax = 0.0f;
  for (size_t i = 0; i < total; ++i) {
    float ax = std::abs(src[i]);
    if (ax > amax) amax = ax;
  }
  // Q6_K: d * scales[k] * quant. For forced: scales[k]=127, quant range [-32,31]
  float forced_d = amax / (31.0f * 127.0f);
  if (forced_d < 1e-10f) forced_d = 1e-10f;
  uint16_t forced_d_fp16 = fp32_to_fp16(forced_d);
  float forced_d_actual = fp16_to_fp32(forced_d_fp16);

  std::cerr << "  Forced global scale: d=" << forced_d
            << " (fp16 roundtrip=" << forced_d_actual
            << "), amax=" << amax << std::endl;

  // Create normal Q6_K tensor via Quantizer (to get correct size/layout)
  auto quantizer = Quantization::createQuantizer(QScheme::Q6_K);
  Tensor q6k_tensor = quantizer->quantize(W_fp32, Tdatatype::Q6_K);

  // Overwrite block contents with forced scale
  uint8_t *raw = q6k_tensor.getData<uint8_t>();
  size_t block_size = sizeof(block_q6_K_local); // 210
  size_t blocks_per_row = K / QK_K_LOCAL; // 768 / 256 = 3

  for (size_t row = 0; row < N; ++row) {
    for (size_t b = 0; b < blocks_per_row; ++b) {
      block_q6_K_local *block = reinterpret_cast<block_q6_K_local *>(
        raw + (row * blocks_per_row + b) * block_size);

      block->d = forced_d_fp16;
      for (int s = 0; s < 16; ++s) block->scales[s] = 127;

      const float *x = src + row * K + b * QK_K_LOCAL;
      uint8_t L[QK_K_LOCAL];
      float effective_scale = forced_d_actual * 127.0f;
      for (int i = 0; i < QK_K_LOCAL; ++i) {
        if (effective_scale < 1e-10f) { L[i] = 32; continue; }
        int q = static_cast<int>(std::round(x[i] / effective_scale));
        q = std::max(-32, std::min(31, q));
        L[i] = static_cast<uint8_t>(q + 32);
      }
      pack_q6k_block(block, L);
    }
  }

  return q6k_tensor;
}

// ─── Accuracy comparison helper ─────────────────────────────────────────────
static void compareOutputs(const Tensor &ref, const Tensor &test,
                           const char *name) {
  const float *r = ref.getData<float>();
  const float *t = test.getData<float>();
  double mse = 0, max_err = 0;
  for (unsigned i = 0; i < ref.size(); ++i) {
    double d = (double)r[i] - (double)t[i];
    mse += d * d;
    if (std::abs(d) > max_err) max_err = std::abs(d);
  }
  mse /= ref.size();
  std::cerr << "  " << std::setw(28) << std::left << name
            << " MSE=" << std::scientific << std::setprecision(6) << mse
            << " MaxErr=" << std::fixed << std::setprecision(6) << max_err
            << std::endl;
}

int main() {
  std::cerr << "=== Forced-Scale GGML Benchmark (Layer 1 only) ===" << std::endl;
  std::cerr << "Comparing: our per-tensor scale in Q6_K blocks vs GGML Q6_K\n"
            << std::endl;

  Tensor W1(TensorDim(1, 1, INPUT_DIM, HIDDEN_DIM));
  Tensor input(TensorDim(BATCH, 1, 1, INPUT_DIM));
  fillRandom(W1, 0.0f, 0.05f, 42);
  fillRandom(input, 0.0f, 1.0f, 100);

  // ── Normal GGML Q6_K ──
  std::cerr << "--- Normal GGML Q6_K ---" << std::endl;
  auto quantizer = Quantization::createQuantizer(QScheme::Q6_K);
  Tensor W1_q6k_normal = quantizer->quantize(W1, Tdatatype::Q6_K);
  std::cerr << "  Size: " << W1_q6k_normal.bytes() << " bytes" << std::endl;

  // ── Forced-scale Q6_K ──
  std::cerr << "\n--- Forced-Scale Q6_K (our per-tensor scale) ---" << std::endl;
  Tensor W1_q6k_forced = buildForcedQ6K(W1);
  std::cerr << "  Size: " << W1_q6k_forced.bytes() << " bytes (same layout)"
            << std::endl;

  // ── FP32 baseline ──
  Tensor out_fp32(TensorDim(BATCH, 1, 1, HIDDEN_DIM));
  input.dot(W1, out_fp32, false, false);

  // ── GGML normal ──
  Tensor out_normal(TensorDim(BATCH, 1, 1, HIDDEN_DIM));
  input.dot(W1_q6k_normal, out_normal, false, false);

  // ── GGML forced ──
  Tensor out_forced(TensorDim(BATCH, 1, 1, HIDDEN_DIM));
  input.dot(W1_q6k_forced, out_forced, false, false);

  // ── Accuracy ──
  std::cerr << "\n--- Accuracy vs FP32 Baseline ---" << std::endl;
  compareOutputs(out_fp32, out_normal, "GGML Q6_K (per-block)");
  compareOutputs(out_fp32, out_forced, "Forced Q6_K (per-tensor)");

  // ── Latency ──
  std::cerr << "\n--- Latency (" << NUM_ITERS << " iters, layer 1 only) ---"
            << std::endl;

  {
    auto s = Clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
      Tensor tmp(TensorDim(BATCH, 1, 1, HIDDEN_DIM));
      input.dot(W1, tmp, false, false);
    }
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - s).count();
    std::cerr << "  FP32 baseline:       " << std::fixed << std::setprecision(2)
              << ms / NUM_ITERS << " ms/iter" << std::endl;
  }
  {
    auto s = Clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
      Tensor tmp(TensorDim(BATCH, 1, 1, HIDDEN_DIM));
      input.dot(W1_q6k_normal, tmp, false, false);
    }
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - s).count();
    std::cerr << "  GGML Q6_K normal:    " << std::fixed << std::setprecision(2)
              << ms / NUM_ITERS << " ms/iter" << std::endl;
  }
  {
    auto s = Clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
      Tensor tmp(TensorDim(BATCH, 1, 1, HIDDEN_DIM));
      input.dot(W1_q6k_forced, tmp, false, false);
    }
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - s).count();
    std::cerr << "  GGML Q6_K forced:    " << std::fixed << std::setprecision(2)
              << ms / NUM_ITERS << " ms/iter" << std::endl;
  }

  std::cerr << "\n--- Key Insight ---" << std::endl;
  std::cerr << "  Forced and normal Q6_K should have SAME latency (same SIMD path)."
            << std::endl;
  std::cerr << "  But forced should have WORSE accuracy (per-tensor < per-block)."
            << std::endl;
  std::cerr << "\n=== Done ===" << std::endl;
  return 0;
}
