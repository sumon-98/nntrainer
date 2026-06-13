// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_lora_q6k_inference_poc.cpp
 * @brief  POC: Fully quantized inference — PTQ Q6_K base + Forced Q6_K LoRA
 *
 * Workflow:
 *   Phase 1 — Train FP32 baseline on digits 0-7, extract weights, evaluate
 *   Phase 2 — PTQ: Quantize base weights to Q6_K (standard GGML quantization)
 *              Build LoRA model on frozen Q6_K base, train FP32 LoRA adapters
 *              on ALL digits (0-9) so LoRA must learn digits 8-9
 *   Phase 3 — Quantize LoRA contribution (A×B×scaling) to forced Q6_K
 *   Phase 4 — Benchmark 5 inference modes:
 *              Mode 1: FP32 baseline (trained on 0-7)
 *              Mode 2: PTQ Q6_K base only (no LoRA)
 *              Mode 3: PTQ Q6_K base + FP32 LoRA
 *              Mode 4: PTQ Q6_K base + Forced Q6_K LoRA
 *              Mode 5: FP32 base + FP32 LoRA (upper bound for LoRA recovery)
 *
 * Architecture: 768→256→256→256→10  (hidden layers have LoRA, output is trainable)
 *
 * Key idea:
 *   The LoRA contribution (A × B × scaling) has the same shape as the base
 *   weight matrix, so it can be quantized to Q6_K independently. At inference:
 *     output = input · Q6_K_base + input · Q6_K_lora + bias
 *   Both dot products use GGML's dotQnK kernels automatically.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <app_context.h>
#include <dataset.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <quantizer.h>
#include <tensor.h>
#include <tensor_dim.h>

#include "qat_fc_layer.h"

using namespace ml::train;
using namespace nntrainer;
using Clock = std::chrono::high_resolution_clock;

// ─── Network & Training Config ──────────────────────────────────────────────

static constexpr unsigned int FEATURE_SIZE_ORIG = 784;
static constexpr unsigned int FEATURE_SIZE = 768;  // truncated for Q6_K compat
static constexpr unsigned int HIDDEN_DIM = 256;
static constexpr unsigned int NUM_CLASSES = 10;

static constexpr unsigned int BATCH_SIZE = 1;
static constexpr unsigned int EPOCHS_FP32 = 5;
static constexpr unsigned int EPOCHS_LORA = 10;
static constexpr unsigned int LORA_RANK = 4;
static constexpr int BENCH_ITERS = 500;
constexpr unsigned int SEED = 42;

// ─── MNIST Data Loading ─────────────────────────────────────────────────────

class DataInformation {
public:
  DataInformation(unsigned int num_samples, const std::string &filename,
                  unsigned int offset = 0);
  unsigned int count;
  unsigned int num_samples;
  unsigned int sample_offset;
  std::ifstream file;
  std::vector<unsigned int> idxes;
  std::mt19937 rng;
};

DataInformation::DataInformation(unsigned int num_samples,
                                 const std::string &filename,
                                 unsigned int offset) :
  count(0), num_samples(num_samples), sample_offset(offset),
  file(filename, std::ios::in | std::ios::binary),
  idxes(num_samples) {
  std::iota(idxes.begin(), idxes.end(), offset);
  rng.seed(SEED);
  std::shuffle(idxes.begin(), idxes.end(), rng);
  if (!file.good())
    throw std::invalid_argument("Cannot open: " + filename);
}

static bool getData(std::ifstream &F, float *input, float *label,
                    unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position = (uint64_t)((FEATURE_SIZE_ORIG + NUM_CLASSES) *
                                 (uint64_t)id * sizeof(float));
  if (position > file_length) return false;
  F.seekg(position, std::ios::beg);

  // Read full 784 features, normalize by /255, copy first 768
  std::vector<float> tmp(FEATURE_SIZE_ORIG);
  F.read((char *)tmp.data(), sizeof(float) * FEATURE_SIZE_ORIG);
  for (unsigned int i = 0; i < FEATURE_SIZE; i++) {
    input[i] = tmp[i] / 255.0f;
  }

  F.read((char *)label, sizeof(float) * NUM_CLASSES);
  return true;
}

static int getSample_train(float **outVec, float **outLabel, bool *last,
                           void *user_data) {
  auto data = reinterpret_cast<DataInformation *>(user_data);
  getData(data->file, *outVec, *outLabel, data->idxes.at(data->count));
  data->count++;
  if (data->count < data->num_samples) {
    *last = false;
  } else {
    *last = true;
    data->count = 0;
    std::shuffle(data->idxes.begin(), data->idxes.end(), data->rng);
  }
  return 0;
}

// ─── Weight Extraction ──────────────────────────────────────────────────────

struct LayerWeights {
  std::string name;
  std::vector<float> weight;
  std::vector<float> bias;
  unsigned int in_dim;
  unsigned int out_dim;
};

static LayerWeights extractLayer(std::unique_ptr<Model> &model,
                                 const char *name,
                                 unsigned int in_d, unsigned int out_d) {
  LayerWeights lw;
  lw.name = name;
  lw.in_dim = in_d;
  lw.out_dim = out_d;

  std::shared_ptr<ml::train::Layer> layer;
  model->getLayer(name, &layer);
  std::vector<float *> wptrs;
  std::vector<TensorDim> wdims;
  layer->getWeights(wptrs, wdims);

  if (wptrs.size() >= 1 && wdims.size() >= 1) {
    size_t wsz = 1;
    for (unsigned d = 0; d < wdims[0].getNumDim(); ++d)
      wsz *= wdims[0].getDim()[d];
    lw.weight.assign(wptrs[0], wptrs[0] + wsz);
  }
  if (wptrs.size() >= 2 && wdims.size() >= 2) {
    size_t bsz = 1;
    for (unsigned d = 0; d < wdims[1].getNumDim(); ++d)
      bsz *= wdims[1].getDim()[d];
    lw.bias.assign(wptrs[1], wptrs[1] + bsz);
  }

  std::cerr << "  Extracted " << name << ": W["
            << lw.weight.size() << "] b[" << lw.bias.size() << "]"
            << std::endl;
  return lw;
}

// ─── Tensor helpers ─────────────────────────────────────────────────────────

static Tensor makeWeightTensor(const LayerWeights &lw) {
  Tensor t(TensorDim(1, 1, lw.in_dim, lw.out_dim));
  std::memcpy(t.getData<float>(), lw.weight.data(), lw.weight.size() * sizeof(float));
  return t;
}

static Tensor makeBiasTensor(const LayerWeights &lw) {
  Tensor t(TensorDim(1, 1, 1, lw.out_dim));
  std::memcpy(t.getData<float>(), lw.bias.data(), lw.bias.size() * sizeof(float));
  return t;
}

// ─── FP16 conversion for forced-scale Q6_K ──────────────────────────────────

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
    if (mant == 0) {
      float r;
      uint32_t z = sign;
      memcpy(&r, &z, 4);
      return r;
    }
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

// ─── Q6_K block struct ──────────────────────────────────────────────────────

#define QK_K_LOCAL 256
#pragma pack(push, 1)
struct block_q6_K_local {
  uint8_t ql[QK_K_LOCAL / 2];
  uint8_t qh[QK_K_LOCAL / 4];
  int8_t scales[QK_K_LOCAL / 16];
  uint16_t d;
};
#pragma pack(pop)

static void pack_q6k_block(block_q6_K_local *block, const uint8_t *L) {
  uint8_t *ql = block->ql;
  uint8_t *qh = block->qh;
  for (int j = 0; j < QK_K_LOCAL; j += 128) {
    for (int l = 0; l < 32; ++l) {
      ql[l + 0] = (L[j + l] & 0xF) | ((L[j + l + 64] & 0xF) << 4);
      ql[l + 32] = (L[j + l + 32] & 0xF) | ((L[j + l + 96] & 0xF) << 4);
      qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
              ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
    }
    ql += 64;
    qh += 32;
  }
}

// ─── Build forced-scale Q6_K from an FP32 weight tensor ────────────────────

static Tensor buildForcedQ6K(const Tensor &W_fp32, float forced_min,
                              float forced_max) {
  unsigned int K = W_fp32.getDim().height();
  unsigned int N = W_fp32.getDim().width();
  Tensor W_t = W_fp32.transpose("0:2:1");
  const float *src = W_t.getData<float>();

  float amax = std::max(std::abs(forced_min), std::abs(forced_max));
  float forced_d = amax / 31.0f;
  if (forced_d < 1e-10f) forced_d = 1e-10f;
  uint16_t forced_d_fp16 = fp32_to_fp16(forced_d);
  float forced_d_actual = fp16_to_fp32(forced_d_fp16);

  std::cerr << "  buildForcedQ6K: amax=" << amax
            << " d=" << forced_d << " d_fp16=" << forced_d_actual
            << " (scales[k]=1)" << std::endl;

  auto quantizer = Quantization::createQuantizer(QScheme::Q6_K);
  Tensor q6k = quantizer->quantize(W_fp32, Tdatatype::Q6_K);

  uint8_t *raw = q6k.getData<uint8_t>();
  size_t block_size = sizeof(block_q6_K_local);
  size_t blocks_per_row = K / QK_K_LOCAL;

  for (size_t row = 0; row < N; ++row) {
    for (size_t b = 0; b < blocks_per_row; ++b) {
      auto *block = reinterpret_cast<block_q6_K_local *>(
        raw + (row * blocks_per_row + b) * block_size);
      block->d = forced_d_fp16;
      for (int s = 0; s < 16; ++s)
        block->scales[s] = 1;

      const float *x = src + row * K + b * QK_K_LOCAL;
      uint8_t L[QK_K_LOCAL];
      float eff = forced_d_actual;
      for (int i = 0; i < QK_K_LOCAL; ++i) {
        if (eff < 1e-10f) { L[i] = 32; continue; }
        int q = static_cast<int>(std::round(x[i] / eff));
        q = std::max(-32, std::min(31, q));
        L[i] = static_cast<uint8_t>(q + 32);
      }
      pack_q6k_block(block, L);
    }
  }

  return q6k;
}

// ─── Forward pass with optional LoRA contribution ──────────────────────────

static void forward_pass(const Tensor &input,
                          const Tensor &W1, const Tensor &b1,
                          const Tensor &W2, const Tensor &b2,
                          const Tensor &W3, const Tensor &b3,
                          const Tensor &Wout, const Tensor &bout,
                          Tensor &logits,
                          const Tensor *W1_lora = nullptr,
                          const Tensor *W2_lora = nullptr,
                          const Tensor *W3_lora = nullptr) {
  unsigned int batch = input.getDim().batch();

  // Layer 1
  Tensor h1(TensorDim(batch, 1, 1, HIDDEN_DIM));
  input.dot(W1, h1, false, false);
  if (W1_lora) {
    Tensor h1_lora(TensorDim(batch, 1, 1, HIDDEN_DIM));
    input.dot(*W1_lora, h1_lora, false, false);
    h1.add_i(h1_lora);
  }
  h1.add_i(b1);
  h1.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, h1);

  // Layer 2
  Tensor h2(TensorDim(batch, 1, 1, HIDDEN_DIM));
  h1.dot(W2, h2, false, false);
  if (W2_lora) {
    Tensor h2_lora(TensorDim(batch, 1, 1, HIDDEN_DIM));
    h1.dot(*W2_lora, h2_lora, false, false);
    h2.add_i(h2_lora);
  }
  h2.add_i(b2);
  h2.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, h2);

  // Layer 3
  Tensor h3(TensorDim(batch, 1, 1, HIDDEN_DIM));
  h2.dot(W3, h3, false, false);
  if (W3_lora) {
    Tensor h3_lora(TensorDim(batch, 1, 1, HIDDEN_DIM));
    h2.dot(*W3_lora, h3_lora, false, false);
    h3.add_i(h3_lora);
  }
  h3.add_i(b3);
  h3.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, h3);

  // Output layer
  h3.dot(Wout, logits, false, false);
  logits.add_i(bout);
}

// ─── Accuracy evaluation ────────────────────────────────────────────────────

static float evaluateAccuracy(const std::string &data_file,
                               const Tensor &W1, const Tensor &b1,
                               const Tensor &W2, const Tensor &b2,
                               const Tensor &W3, const Tensor &b3,
                               const Tensor &Wout, const Tensor &bout,
                               unsigned int test_offset,
                               unsigned int num_test,
                               const Tensor *W1_lora = nullptr,
                               const Tensor *W2_lora = nullptr,
                               const Tensor *W3_lora = nullptr) {
  unsigned int correct = 0;
  unsigned int total = 0;

  std::ifstream file(data_file, std::ios::in | std::ios::binary);
  std::vector<float> input_buf(FEATURE_SIZE);
  std::vector<float> label_buf(NUM_CLASSES);

  for (unsigned int i = 0; i < num_test; ++i) {
    if (!getData(file, input_buf.data(), label_buf.data(), test_offset + i))
      break;

    Tensor input(TensorDim(1, 1, 1, FEATURE_SIZE));
    std::memcpy(input.getData<float>(), input_buf.data(),
                FEATURE_SIZE * sizeof(float));

    Tensor logits(TensorDim(1, 1, 1, NUM_CLASSES));
    forward_pass(input, W1, b1, W2, b2, W3, b3, Wout, bout, logits,
                 W1_lora, W2_lora, W3_lora);

    const float *lp = logits.getData<float>();
    int pred = 0;
    for (unsigned j = 1; j < NUM_CLASSES; ++j)
      if (lp[j] > lp[pred]) pred = j;

    int truth = 0;
    for (unsigned j = 1; j < NUM_CLASSES; ++j)
      if (label_buf[j] > label_buf[truth]) truth = j;

    if (pred == truth) correct++;
    total++;
  }

  return total > 0 ? (float)correct / total * 100.0f : 0.0f;
}

// ─── Latency benchmark ──────────────────────────────────────────────────────

static double benchmarkLatency(const Tensor &W1, const Tensor &b1,
                                const Tensor &W2, const Tensor &b2,
                                const Tensor &W3, const Tensor &b3,
                                const Tensor &Wout, const Tensor &bout,
                                const Tensor *W1_lora = nullptr,
                                const Tensor *W2_lora = nullptr,
                                const Tensor *W3_lora = nullptr) {
  Tensor dummy_in(TensorDim(32, 1, 1, FEATURE_SIZE));
  float *din = dummy_in.getData<float>();
  std::mt19937 gen(123);
  std::normal_distribution<float> dist(0.f, 0.5f);
  for (unsigned i = 0; i < dummy_in.size(); ++i)
    din[i] = dist(gen);

  Tensor logits(TensorDim(32, 1, 1, NUM_CLASSES));

  for (int i = 0; i < 10; ++i)
    forward_pass(dummy_in, W1, b1, W2, b2, W3, b3, Wout, bout, logits,
                 W1_lora, W2_lora, W3_lora);

  auto start = Clock::now();
  for (int i = 0; i < BENCH_ITERS; ++i) {
    forward_pass(dummy_in, W1, b1, W2, b2, W3, b3, Wout, bout, logits,
                 W1_lora, W2_lora, W3_lora);
  }
  auto end = Clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() /
         BENCH_ITERS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase 1: Train FP32 baseline on digits 0-7 ONLY
// ═════════════════════════════════════════════════════════════════════════════

static std::vector<LayerWeights> trainFP32_digits07(
    const std::string &data_file_07,
    unsigned int num_train_07, unsigned int num_val_07) {
  std::cerr << "\n╔══════════════════════════════════════════════╗"
            << "\n║  Phase 1: FP32 Training on DIGITS 0-7 ONLY  ║"
            << "\n╚══════════════════════════════════════════════╝"
            << std::endl;

  auto train_data = std::make_unique<DataInformation>(num_train_07, data_file_07, 0);
  auto val_data = std::make_unique<DataInformation>(num_val_07, data_file_07, num_train_07);

  std::shared_ptr<ml::train::Dataset> dataset_train =
    createDataset(DatasetType::GENERATOR, getSample_train, train_data.get());
  std::shared_ptr<ml::train::Dataset> dataset_val =
    createDataset(DatasetType::GENERATOR, getSample_train, val_data.get());

  auto model = createModel(ModelType::NEURAL_NET,
                           {"batch_size=" + std::to_string(BATCH_SIZE)});

  model->addLayer(createLayer("input", {"name=input0",
                  "input_shape=1:1:" + std::to_string(FEATURE_SIZE)}));
  model->addLayer(createLayer("fully_connected", {"name=fc1",
                  "unit=" + std::to_string(HIDDEN_DIM)}));
  model->addLayer(createLayer("activation", {"name=relu1", "activation=relu"}));
  model->addLayer(createLayer("fully_connected", {"name=fc2",
                  "unit=" + std::to_string(HIDDEN_DIM)}));
  model->addLayer(createLayer("activation", {"name=relu2", "activation=relu"}));
  model->addLayer(createLayer("fully_connected", {"name=fc3",
                  "unit=" + std::to_string(HIDDEN_DIM)}));
  model->addLayer(createLayer("activation", {"name=relu3", "activation=relu"}));
  model->addLayer(createLayer("fully_connected",
                  {"name=output", "unit=" + std::to_string(NUM_CLASSES), "activation=softmax"}));

  auto optimizer = createOptimizer("sgd", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));
  model->setProperty({"epochs=" + std::to_string(EPOCHS_FP32), "loss=cross"});

  model->compile();
  model->initialize();
  model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
  model->setDataset(DatasetModeType::MODE_VALID, dataset_val);

  model->train();
  std::cerr << "FP32 training loss: " << model->getTrainingLoss() << std::endl;

  std::cerr << "Extracting FP32 weights..." << std::endl;
  std::vector<LayerWeights> mw;
  mw.push_back(extractLayer(model, "fc1", FEATURE_SIZE, HIDDEN_DIM));
  mw.push_back(extractLayer(model, "fc2", HIDDEN_DIM, HIDDEN_DIM));
  mw.push_back(extractLayer(model, "fc3", HIDDEN_DIM, HIDDEN_DIM));
  mw.push_back(extractLayer(model, "output", HIDDEN_DIM, NUM_CLASSES));
  return mw;
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase 2: PTQ Q6_K Base + LoRA Training on ALL digits (0-9)
// ═════════════════════════════════════════════════════════════════════════════

struct LoRAWeights {
  std::vector<float> loraA;
  std::vector<float> loraB;
  float scaling;
};

struct LoRATrainingResult {
  std::vector<LoRAWeights> lora_weights;
  LayerWeights trained_output;     // output layer after LoRA training (trainable)
  std::vector<std::vector<float>> trained_biases;  // hidden layer biases after LoRA training
};

static LoRATrainingResult trainLoRA(
  const std::string &data_file_all,
  const std::vector<LayerWeights> &fp32_weights,
  unsigned int finetune_offset,
  unsigned int finetune_train,
  unsigned int finetune_val) {

  std::cerr << "\n╔══════════════════════════════════════════════╗"
            << "\n║  Phase 2: Q6_K Base (0-7) + LoRA on ALL     ║"
            << "\n╚══════════════════════════════════════════════╝"
            << std::endl;

  // ── Step 2a: Quantize base weights to Q6_K (PTQ) ──
  std::cerr << "\n--- Step 2a: PTQ — Quantizing base weights to GGML Q6_K ---" << std::endl;

  auto q6k_quantizer = Quantization::createQuantizer(QScheme::Q6_K);

  Tensor W1_fp32 = makeWeightTensor(fp32_weights[0]);
  Tensor W2_fp32 = makeWeightTensor(fp32_weights[1]);
  Tensor W3_fp32 = makeWeightTensor(fp32_weights[2]);

  Tensor W1_q6k = q6k_quantizer->quantize(W1_fp32, Tdatatype::Q6_K);
  Tensor W2_q6k = q6k_quantizer->quantize(W2_fp32, Tdatatype::Q6_K);
  Tensor W3_q6k = q6k_quantizer->quantize(W3_fp32, Tdatatype::Q6_K);

  std::cerr << "  W1 FP32: " << W1_fp32.bytes() << " bytes → Q6_K: "
            << W1_q6k.bytes() << " bytes ("
            << std::fixed << std::setprecision(1)
            << (1.0f - (float)W1_q6k.bytes() / W1_fp32.bytes()) * 100
            << "% reduction)" << std::endl;

  // ── Step 2b: Build LoRA model ──
  std::cerr << "\n--- Step 2b: Building LoRA model on frozen Q6_K base ---" << std::endl;

  auto train_data = std::make_unique<DataInformation>(finetune_train, data_file_all, finetune_offset);
  auto val_data = std::make_unique<DataInformation>(finetune_val, data_file_all, finetune_offset + finetune_train);
  std::shared_ptr<ml::train::Dataset> dataset_train =
    createDataset(DatasetType::GENERATOR, getSample_train, train_data.get());
  std::shared_ptr<ml::train::Dataset> dataset_val =
    createDataset(DatasetType::GENERATOR, getSample_train, val_data.get());

  auto lora_model = createModel(ModelType::NEURAL_NET,
                           {"batch_size=" + std::to_string(BATCH_SIZE)});

  lora_model->addLayer(createLayer("input", {"name=input0",
                  "input_shape=1:1:" + std::to_string(FEATURE_SIZE)}));

  lora_model->addLayer(createLayer("qat_fully_connected", {"name=lfc1",
                  "unit=" + std::to_string(HIDDEN_DIM),
                  "lora_rank=" + std::to_string(LORA_RANK)}));
  lora_model->addLayer(createLayer("activation", {"name=relu1", "activation=relu"}));

  lora_model->addLayer(createLayer("qat_fully_connected", {"name=lfc2",
                  "unit=" + std::to_string(HIDDEN_DIM),
                  "lora_rank=" + std::to_string(LORA_RANK)}));
  lora_model->addLayer(createLayer("activation", {"name=relu2", "activation=relu"}));

  lora_model->addLayer(createLayer("qat_fully_connected", {"name=lfc3",
                  "unit=" + std::to_string(HIDDEN_DIM),
                  "lora_rank=" + std::to_string(LORA_RANK)}));
  lora_model->addLayer(createLayer("activation", {"name=relu3", "activation=relu"}));

  // Output layer: TRAINABLE — LoRA needs to learn new output logits for digits 8-9
  lora_model->addLayer(createLayer("fully_connected",
                  {"name=output", "unit=" + std::to_string(NUM_CLASSES),
                   "activation=softmax", "trainable=true"}));

  auto lora_optimizer = createOptimizer("adam", {"learning_rate=0.001"});
  lora_model->setOptimizer(std::move(lora_optimizer));
  lora_model->setProperty({"epochs=" + std::to_string(EPOCHS_LORA), "loss=cross"});

  lora_model->compile();
  lora_model->initialize();

  // ── Step 2c: Inject Q6_K weights ──
  std::cerr << "\n--- Step 2c: Injecting Q6_K weights into frozen base ---" << std::endl;

  auto injectQ6K = [&](const char *name, Tensor &q6k_tensor) {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    std::memcpy(wptrs[0], q6k_tensor.getData<uint8_t>(), q6k_tensor.bytes());
    std::cerr << "  Injected Q6_K weights into " << name
              << " (" << q6k_tensor.bytes() << " bytes)" << std::endl;
  };

  injectQ6K("lfc1", W1_q6k);
  injectQ6K("lfc2", W2_q6k);
  injectQ6K("lfc3", W3_q6k);

  // Inject FP32 weights + bias into output layer
  {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer("output", &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    if (wptrs.size() >= 1)
      std::memcpy(wptrs[0], fp32_weights[3].weight.data(),
                  fp32_weights[3].weight.size() * sizeof(float));
    if (wptrs.size() >= 2)
      std::memcpy(wptrs[1], fp32_weights[3].bias.data(),
                  fp32_weights[3].bias.size() * sizeof(float));
    std::cerr << "  Injected FP32 weights into output" << std::endl;
  }

  // Inject biases into LoRA layers
  auto injectBias = [&](const char *name, const LayerWeights &lw) {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    if (wptrs.size() >= 2 && !lw.bias.empty())
      std::memcpy(wptrs[1], lw.bias.data(), lw.bias.size() * sizeof(float));
    std::cerr << "  Injected bias into " << name << std::endl;
  };

  injectBias("lfc1", fp32_weights[0]);
  injectBias("lfc2", fp32_weights[1]);
  injectBias("lfc3", fp32_weights[2]);

  // ── Step 2d: Train LoRA on ALL digits ──
  lora_model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
  lora_model->setDataset(DatasetModeType::MODE_VALID, dataset_val);

  std::cerr << "\n--- Step 2d: Training LoRA on ALL digits (0-9) ---" << std::endl;
  std::cerr << "  LoRA rank: " << LORA_RANK << ", Epochs: " << EPOCHS_LORA << std::endl;
  std::cerr << "  Base knows: digits 0-7 | LoRA must learn: digits 8-9" << std::endl;

  lora_model->train();
  std::cerr << "\nLoRA training loss: " << lora_model->getTrainingLoss() << std::endl;

  // ── Step 2e: Extract trained weights ──
  std::cerr << "\n--- Step 2e: Extracting trained LoRA weights ---" << std::endl;

  LoRATrainingResult result;
  result.lora_weights.resize(3);
  result.trained_biases.resize(3);

  auto extractLoRA = [&](const char *name, unsigned int in_d,
                         unsigned int out_d, unsigned int idx) {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);

    result.lora_weights[idx].scaling = 1.0f;

    if (wptrs.size() >= 4) {
      size_t a_sz = in_d * LORA_RANK;
      result.lora_weights[idx].loraA.assign(wptrs[2], wptrs[2] + a_sz);
      size_t b_sz = LORA_RANK * out_d;
      result.lora_weights[idx].loraB.assign(wptrs[3], wptrs[3] + b_sz);
    }

    // Extract trained bias
    if (wptrs.size() >= 2) {
      size_t bsz = out_d;
      result.trained_biases[idx].assign(wptrs[1], wptrs[1] + bsz);
    }

    std::cerr << "  Extracted LoRA from " << name
              << ": A[" << result.lora_weights[idx].loraA.size()
              << "] B[" << result.lora_weights[idx].loraB.size() << "]" << std::endl;
  };

  extractLoRA("lfc1", FEATURE_SIZE, HIDDEN_DIM, 0);
  extractLoRA("lfc2", HIDDEN_DIM, HIDDEN_DIM, 1);
  extractLoRA("lfc3", HIDDEN_DIM, HIDDEN_DIM, 2);

  // Extract trained output layer
  {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer("output", &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);

    result.trained_output.name = "output";
    result.trained_output.in_dim = HIDDEN_DIM;
    result.trained_output.out_dim = NUM_CLASSES;

    if (wptrs.size() >= 1 && wdims.size() >= 1) {
      size_t wsz = 1;
      for (unsigned d = 0; d < wdims[0].getNumDim(); ++d)
        wsz *= wdims[0].getDim()[d];
      result.trained_output.weight.assign(wptrs[0], wptrs[0] + wsz);
    }
    if (wptrs.size() >= 2 && wdims.size() >= 2) {
      size_t bsz = 1;
      for (unsigned d = 0; d < wdims[1].getNumDim(); ++d)
        bsz *= wdims[1].getDim()[d];
      result.trained_output.bias.assign(wptrs[1], wptrs[1] + bsz);
    }
    std::cerr << "  Extracted trained output: W["
              << result.trained_output.weight.size() << "] b["
              << result.trained_output.bias.size() << "]" << std::endl;
  }

  return result;
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase 3: Compute LoRA contribution and quantize to forced Q6_K
// ═════════════════════════════════════════════════════════════════════════════

struct LoRAQ6KWeights {
  Tensor W_lora_q6k;       // forced Q6_K quantized LoRA contribution
  Tensor W_lora_fp32;      // FP32 LoRA contribution (for comparison)
  float lora_min;
  float lora_max;
};

static std::vector<LoRAQ6KWeights> quantizeLoRA(
  const std::vector<LoRAWeights> &lora_weights) {

  std::cerr << "\n╔══════════════════════════════════════════════╗"
            << "\n║  Phase 3: Quantize LoRA to Forced Q6_K       ║"
            << "\n╚══════════════════════════════════════════════╝"
            << std::endl;

  std::vector<LoRAQ6KWeights> result;

  unsigned int in_dims[] = {FEATURE_SIZE, HIDDEN_DIM, HIDDEN_DIM};
  unsigned int out_dims[] = {HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM};

  for (size_t i = 0; i < lora_weights.size(); ++i) {
    const auto &lw = lora_weights[i];
    unsigned int in_d = in_dims[i];
    unsigned int out_d = out_dims[i];

    std::cerr << "\n--- LoRA layer " << i << " (in=" << in_d
              << ", out=" << out_d << ") ---" << std::endl;

    // Compute FP32 LoRA contribution: A × B × scaling
    Tensor A(TensorDim(1, 1, in_d, LORA_RANK));
    std::memcpy(A.getData<float>(), lw.loraA.data(),
                lw.loraA.size() * sizeof(float));

    Tensor B(TensorDim(1, 1, LORA_RANK, out_d));
    std::memcpy(B.getData<float>(), lw.loraB.data(),
                lw.loraB.size() * sizeof(float));

    Tensor W_lora_fp32(TensorDim(1, 1, in_d, out_d));
    A.dot(B, W_lora_fp32, false, false);
    W_lora_fp32.multiply_i(lw.scaling);

    float lmin = W_lora_fp32.minValue();
    float lmax = W_lora_fp32.maxValue();

    std::cerr << "  LoRA contribution range: [" << lmin << ", " << lmax << "]"
              << std::endl;

    Tensor W_lora_q6k = buildForcedQ6K(W_lora_fp32, lmin, lmax);

    LoRAQ6KWeights lq;
    lq.W_lora_q6k = std::move(W_lora_q6k);
    lq.W_lora_fp32 = std::move(W_lora_fp32);
    lq.lora_min = lmin;
    lq.lora_max = lmax;
    result.push_back(std::move(lq));

    std::cerr << "  LoRA FP32: " << result.back().W_lora_fp32.bytes()
              << " bytes → Forced Q6_K: " << result.back().W_lora_q6k.bytes()
              << " bytes" << std::endl;
  }

  return result;
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <mnist_digits_0to7.dat> <mnist_all_digits.dat>"
              << " [mnist_digits_8to9.dat]"
              << std::endl;
    std::cerr << "\nGenerate these files using create_digit_split_mnist.py" << std::endl;
    return 1;
  }

  std::string data_07 = argv[1];
  std::string data_all = argv[2];
  std::string data_89 = (argc > 3) ? argv[3] : "";

  std::cerr << "\n══════════════════════════════════════════════════"
            << "\n  Quantized LoRA Inference POC"
            << "\n  PTQ Q6_K Base + Forced Q6_K LoRA"
            << "\n  Base: trained on digits 0-7 only"
            << "\n  LoRA: fine-tuned to learn ALL digits (0-9)"
            << "\n══════════════════════════════════════════════════"
            << std::endl;

  // Register custom QAT layer BEFORE any model creation
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<nntrainer::QATFullyConnectedLayer>);
  } catch (std::invalid_argument &) {
    // Already registered — ignore
  }

  // ── Count samples in data files ──

  size_t sample_size = (FEATURE_SIZE_ORIG + NUM_CLASSES) * sizeof(float);

  unsigned int total_07 = 0;
  {
    std::ifstream f(data_07, std::ios::ate | std::ios::binary);
    if (!f.is_open()) {
      std::cerr << "Error: Cannot open file " << data_07 << std::endl;
      return 1;
    }
    total_07 = f.tellg() / sample_size;
    std::cerr << "  Digits 0-7 dataset: " << total_07 << " samples" << std::endl;
  }

  unsigned int total_all = 0;
  {
    std::ifstream f(data_all, std::ios::ate | std::ios::binary);
    total_all = f.tellg() / sample_size;
    std::cerr << "  All-digits dataset: " << total_all << " samples" << std::endl;
  }

  unsigned int total_89 = 0;
  if (!data_89.empty()) {
    std::ifstream f89(data_89, std::ios::ate | std::ios::binary);
    total_89 = f89.tellg() / sample_size;
    std::cerr << "  Digits 8-9 dataset: " << total_89 << " samples" << std::endl;
  }

  // ── Phase 1: Train FP32 baseline on digits 0-7 ──

  unsigned int train_07 = (unsigned int)(total_07 * 0.8);
  unsigned int val_07 = total_07 - train_07;

  auto fp32_weights = trainFP32_digits07(data_07, train_07, val_07);

  // Build FP32 weight tensors
  Tensor W1_fp32 = makeWeightTensor(fp32_weights[0]);
  Tensor b1_fp32 = makeBiasTensor(fp32_weights[0]);
  Tensor W2_fp32 = makeWeightTensor(fp32_weights[1]);
  Tensor b2_fp32 = makeBiasTensor(fp32_weights[1]);
  Tensor W3_fp32 = makeWeightTensor(fp32_weights[2]);
  Tensor b3_fp32 = makeBiasTensor(fp32_weights[2]);
  Tensor Wout_fp32 = makeWeightTensor(fp32_weights[3]);
  Tensor bout_fp32 = makeBiasTensor(fp32_weights[3]);

  // ── Compute data splits for all-digits dataset ──

  unsigned int test_count = 5000;
  unsigned int finetune_val_count = 3000;
  unsigned int finetune_train_count = 12000;
  unsigned int finetune_start = total_all - test_count - finetune_val_count - finetune_train_count;
  unsigned int test_start = total_all - test_count;

  // ── Phase 2: Train LoRA on frozen PTQ Q6_K base (on ALL digits) ──

  auto lora_result = trainLoRA(data_all, fp32_weights,
                               finetune_start, finetune_train_count, finetune_val_count);

  // ── Phase 3: Quantize LoRA contribution to forced Q6_K ──

  auto lora_q6k = quantizeLoRA(lora_result.lora_weights);

  // ── Build PTQ Q6_K base weight tensors ──

  std::cerr << "\n--- Building PTQ Q6_K base weights ---" << std::endl;
  auto q6k_quantizer = Quantization::createQuantizer(QScheme::Q6_K);
  Tensor W1_q6k = q6k_quantizer->quantize(W1_fp32, Tdatatype::Q6_K);
  Tensor W2_q6k = q6k_quantizer->quantize(W2_fp32, Tdatatype::Q6_K);
  Tensor W3_q6k = q6k_quantizer->quantize(W3_fp32, Tdatatype::Q6_K);

  // ── Build trained output weight/bias tensors (from LoRA training) ──

  Tensor Wout_trained = makeWeightTensor(lora_result.trained_output);
  Tensor bout_trained = makeBiasTensor(lora_result.trained_output);

  // ── Build trained bias tensors for hidden layers ──

  Tensor b1_trained(TensorDim(1, 1, 1, HIDDEN_DIM));
  std::memcpy(b1_trained.getData<float>(), lora_result.trained_biases[0].data(),
              HIDDEN_DIM * sizeof(float));
  Tensor b2_trained(TensorDim(1, 1, 1, HIDDEN_DIM));
  std::memcpy(b2_trained.getData<float>(), lora_result.trained_biases[1].data(),
              HIDDEN_DIM * sizeof(float));
  Tensor b3_trained(TensorDim(1, 1, 1, HIDDEN_DIM));
  std::memcpy(b3_trained.getData<float>(), lora_result.trained_biases[2].data(),
              HIDDEN_DIM * sizeof(float));

  // ── Memory stats ──

  size_t fp32_base_mem = W1_fp32.bytes() + W2_fp32.bytes() + W3_fp32.bytes();
  size_t q6k_base_mem = W1_q6k.bytes() + W2_q6k.bytes() + W3_q6k.bytes();
  size_t lora_fp32_mem = lora_q6k[0].W_lora_fp32.bytes() +
                         lora_q6k[1].W_lora_fp32.bytes() +
                         lora_q6k[2].W_lora_fp32.bytes();
  size_t lora_q6k_mem = lora_q6k[0].W_lora_q6k.bytes() +
                        lora_q6k[1].W_lora_q6k.bytes() +
                        lora_q6k[2].W_lora_q6k.bytes();

  // ── Phase 4: Benchmark all inference modes ──

  std::cerr << "\n╔══════════════════════════════════════════════╗"
            << "\n║  Phase 4: Inference Benchmark                ║"
            << "\n╚══════════════════════════════════════════════╝"
            << std::endl;

  struct Result {
    const char *name;
    float accuracy_all;
    float accuracy_89;
    double latency_ms;
    size_t total_weight_bytes;
  };

  std::vector<Result> results;

  // Mode 1: FP32 baseline (trained on 0-7 only)
  std::cerr << "Evaluating Mode 1: FP32 baseline (trained 0-7)..." << std::endl;
  float acc1_all = evaluateAccuracy(data_all,
    W1_fp32, b1_fp32, W2_fp32, b2_fp32, W3_fp32, b3_fp32,
    Wout_fp32, bout_fp32, test_start, test_count);
  float acc1_89 = 0;
  if (!data_89.empty()) {
    acc1_89 = evaluateAccuracy(data_89,
      W1_fp32, b1_fp32, W2_fp32, b2_fp32, W3_fp32, b3_fp32,
      Wout_fp32, bout_fp32, 0, std::min(total_89, (unsigned int)2000));
  }
  double lat1 = benchmarkLatency(
    W1_fp32, b1_fp32, W2_fp32, b2_fp32, W3_fp32, b3_fp32,
    Wout_fp32, bout_fp32);
  results.push_back({"1. FP32 Baseline", acc1_all, acc1_89, lat1, fp32_base_mem});

  // Mode 2: PTQ Q6_K base only (no LoRA)
  std::cerr << "Evaluating Mode 2: PTQ Q6_K base (no LoRA)..." << std::endl;
  float acc2_all = evaluateAccuracy(data_all,
    W1_q6k, b1_fp32, W2_q6k, b2_fp32, W3_q6k, b3_fp32,
    Wout_fp32, bout_fp32, test_start, test_count);
  float acc2_89 = 0;
  if (!data_89.empty()) {
    acc2_89 = evaluateAccuracy(data_89,
      W1_q6k, b1_fp32, W2_q6k, b2_fp32, W3_q6k, b3_fp32,
      Wout_fp32, bout_fp32, 0, std::min(total_89, (unsigned int)2000));
  }
  double lat2 = benchmarkLatency(
    W1_q6k, b1_fp32, W2_q6k, b2_fp32, W3_q6k, b3_fp32,
    Wout_fp32, bout_fp32);
  results.push_back({"2. Q6_K (no LoRA)", acc2_all, acc2_89, lat2, q6k_base_mem});

  // Mode 3: PTQ Q6_K base + FP32 LoRA
  std::cerr << "Evaluating Mode 3: PTQ Q6_K base + FP32 LoRA..." << std::endl;
  float acc3_all = evaluateAccuracy(data_all,
    W1_q6k, b1_trained, W2_q6k, b2_trained, W3_q6k, b3_trained,
    Wout_trained, bout_trained, test_start, test_count,
    &lora_q6k[0].W_lora_fp32, &lora_q6k[1].W_lora_fp32,
    &lora_q6k[2].W_lora_fp32);
  float acc3_89 = 0;
  if (!data_89.empty()) {
    acc3_89 = evaluateAccuracy(data_89,
      W1_q6k, b1_trained, W2_q6k, b2_trained, W3_q6k, b3_trained,
      Wout_trained, bout_trained, 0, std::min(total_89, (unsigned int)2000),
      &lora_q6k[0].W_lora_fp32, &lora_q6k[1].W_lora_fp32,
      &lora_q6k[2].W_lora_fp32);
  }
  double lat3 = benchmarkLatency(
    W1_q6k, b1_trained, W2_q6k, b2_trained, W3_q6k, b3_trained,
    Wout_trained, bout_trained,
    &lora_q6k[0].W_lora_fp32, &lora_q6k[1].W_lora_fp32,
    &lora_q6k[2].W_lora_fp32);
  results.push_back({"3. Q6_K+FP32 LoRA", acc3_all, acc3_89, lat3,
                      q6k_base_mem + lora_fp32_mem});

  // Mode 4: PTQ Q6_K base + Forced Q6_K LoRA (the key mode!)
  std::cerr << "Evaluating Mode 4: PTQ Q6_K base + Forced Q6_K LoRA..." << std::endl;
  float acc4_all = evaluateAccuracy(data_all,
    W1_q6k, b1_trained, W2_q6k, b2_trained, W3_q6k, b3_trained,
    Wout_trained, bout_trained, test_start, test_count,
    &lora_q6k[0].W_lora_q6k, &lora_q6k[1].W_lora_q6k,
    &lora_q6k[2].W_lora_q6k);
  float acc4_89 = 0;
  if (!data_89.empty()) {
    acc4_89 = evaluateAccuracy(data_89,
      W1_q6k, b1_trained, W2_q6k, b2_trained, W3_q6k, b3_trained,
      Wout_trained, bout_trained, 0, std::min(total_89, (unsigned int)2000),
      &lora_q6k[0].W_lora_q6k, &lora_q6k[1].W_lora_q6k,
      &lora_q6k[2].W_lora_q6k);
  }
  double lat4 = benchmarkLatency(
    W1_q6k, b1_trained, W2_q6k, b2_trained, W3_q6k, b3_trained,
    Wout_trained, bout_trained,
    &lora_q6k[0].W_lora_q6k, &lora_q6k[1].W_lora_q6k,
    &lora_q6k[2].W_lora_q6k);
  results.push_back({"4. Q6_K+Q6_K LoRA", acc4_all, acc4_89, lat4,
                      q6k_base_mem + lora_q6k_mem});

  // Mode 5: FP32 base + FP32 LoRA (upper bound for LoRA recovery)
  std::cerr << "Evaluating Mode 5: FP32 base + FP32 LoRA (upper bound)..." << std::endl;
  float acc5_all = evaluateAccuracy(data_all,
    W1_fp32, b1_trained, W2_fp32, b2_trained, W3_fp32, b3_trained,
    Wout_trained, bout_trained, test_start, test_count,
    &lora_q6k[0].W_lora_fp32, &lora_q6k[1].W_lora_fp32,
    &lora_q6k[2].W_lora_fp32);
  float acc5_89 = 0;
  if (!data_89.empty()) {
    acc5_89 = evaluateAccuracy(data_89,
      W1_fp32, b1_trained, W2_fp32, b2_trained, W3_fp32, b3_trained,
      Wout_trained, bout_trained, 0, std::min(total_89, (unsigned int)2000),
      &lora_q6k[0].W_lora_fp32, &lora_q6k[1].W_lora_fp32,
      &lora_q6k[2].W_lora_fp32);
  }
  double lat5 = benchmarkLatency(
    W1_fp32, b1_trained, W2_fp32, b2_trained, W3_fp32, b3_trained,
    Wout_trained, bout_trained,
    &lora_q6k[0].W_lora_fp32, &lora_q6k[1].W_lora_fp32,
    &lora_q6k[2].W_lora_fp32);
  results.push_back({"5. FP32+FP32 LoRA", acc5_all, acc5_89, lat5,
                      fp32_base_mem + lora_fp32_mem});

  // ── Print results table ──

  std::cerr << "\n╔══════════════════════════════════════════════════════════════════════════════╗"
            << std::endl;
  std::cerr << "║              Quantized LoRA Inference POC — Results                          ║"
            << std::endl;
  std::cerr << "╠══════════════════════╦═══════════╦═══════════╦════════════╦════════════════════╣"
            << std::endl;
  std::cerr << "║ Mode                 ║ Acc (all) ║ Acc (8-9) ║ Latency    ║ Weight Mem         ║"
            << std::endl;
  std::cerr << "╠══════════════════════╬═══════════╬═══════════╬════════════╬════════════════════╣"
            << std::endl;

  for (auto &r : results) {
    std::cerr << "║ " << std::setw(20) << std::left << r.name << " ║ "
              << std::setw(7) << std::fixed << std::setprecision(1)
              << r.accuracy_all << "% ║ "
              << std::setw(7) << r.accuracy_89 << "% ║ "
              << std::setw(8) << std::setprecision(2) << r.latency_ms << " ms ║ "
              << std::setw(10) << r.total_weight_bytes / 1024 << " KB"
              << std::setw(6) << " ║" << std::endl;
  }

  std::cerr << "╠══════════════════════╩═══════════╩═══════════╩════════════╩════════════════════╣"
            << std::endl;
  std::cerr << "║ Memory breakdown:                                                             ║"
            << std::endl;
  std::cerr << "║   FP32 base:  " << std::setw(6) << fp32_base_mem / 1024 << " KB"
            << "   Q6_K base:  " << std::setw(6) << q6k_base_mem / 1024 << " KB"
            << "   (" << std::setprecision(1) << (float)fp32_base_mem / q6k_base_mem
            << "x reduction)                    ║" << std::endl;
  std::cerr << "║   FP32 LoRA:  " << std::setw(6) << lora_fp32_mem / 1024 << " KB"
            << "   Q6_K LoRA:  " << std::setw(6) << lora_q6k_mem / 1024 << " KB"
            << "   (" << std::setprecision(1) << (float)lora_fp32_mem / lora_q6k_mem
            << "x reduction)                    ║" << std::endl;
  std::cerr << "║   Fully quantized (Mode 4): "
            << (q6k_base_mem + lora_q6k_mem) / 1024 << " KB total"
            << " vs FP32: " << (fp32_base_mem + lora_fp32_mem) / 1024 << " KB"
            << "   ║" << std::endl;
  std::cerr << "╚══════════════════════════════════════════════════════════════════════════════╝"
            << std::endl;

  // ── Accuracy delta summary ──

  std::cerr << "\n--- Accuracy Delta Summary (ALL digits) ---" << std::endl;
  std::cerr << "  FP32 baseline (0-7):     " << acc1_all << "%" << std::endl;
  std::cerr << "  PTQ Q6_K damage:        " << (acc2_all - acc1_all) << " pp" << std::endl;
  std::cerr << "  LoRA recovery (FP32):    " << (acc3_all - acc2_all) << " pp" << std::endl;
  std::cerr << "  LoRA recovery (Q6_K):    " << (acc4_all - acc2_all) << " pp" << std::endl;
  std::cerr << "  Q6_K LoRA vs FP32 LoRA: " << (acc4_all - acc3_all) << " pp" << std::endl;

  if (!data_89.empty()) {
    std::cerr << "\n--- Accuracy Delta Summary (digits 8-9 ONLY) ---" << std::endl;
    std::cerr << "  FP32 baseline (0-7):     " << acc1_89 << "% (never saw 8,9)" << std::endl;
    std::cerr << "  Q6_K base:               " << acc2_89 << "%" << std::endl;
    std::cerr << "  Q6_K + FP32 LoRA:        " << acc3_89 << "%" << std::endl;
    std::cerr << "  Q6_K + Q6_K LoRA:        " << acc4_89 << "%" << std::endl;
    std::cerr << "  FP32 + FP32 LoRA:         " << acc5_89 << "%" << std::endl;

    if (acc4_89 > acc2_89 + 5.0f) {
      std::cerr << "\n  ✓ SUCCESS: LoRA learned digits 8-9 that base model never saw!" << std::endl;
    }
  }

  std::cerr << "\n=== Done! ===" << std::endl;
  return 0;
}
