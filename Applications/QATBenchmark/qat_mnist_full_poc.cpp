// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_mnist_full_poc.cpp
 * @brief  Comprehensive MNIST Quantization POC — 5 inference modes
 *
 * Trains on real MNIST data (truncated 784→768), then benchmarks:
 *   Mode 1: FP32 baseline
 *   Mode 2: Post-training GGML Q6_K quantization
 *   Mode 3: QAT training → FP32 inference
 *   Mode 4: QAT training → GGML Q6_K quantization
 *   Mode 5: QAT training → Forced Q6_K (QAT per-tensor scale in GGML blocks)
 *
 * Architecture: 768→256→256→256→10 (output layer stays FP32)
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

#include <model.h>
#include <optimizer.h>
#include <layer.h>
#include <dataset.h>
#include <app_context.h>
#include <engine.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <quantizer.h>

#include "qat_fc_layer.h"

using namespace ml::train;
using namespace nntrainer;
using Clock = std::chrono::high_resolution_clock;

// ─── Configuration ──────────────────────────────────────────────────────────
static constexpr unsigned int FEATURE_SIZE_ORIG = 784;
static constexpr unsigned int FEATURE_SIZE = 768;  // truncated for Q6_K compat
static constexpr unsigned int HIDDEN_DIM = 256;    // Q6_K: 256%256=0
static constexpr unsigned int NUM_CLASSES = 10;
static constexpr unsigned int BATCH_SIZE = 32;
static constexpr unsigned int NUM_TRAIN = 100;
static constexpr unsigned int NUM_VAL = 100;
static constexpr unsigned int NUM_TEST = 100;
static constexpr unsigned int EPOCHS = 50;
static constexpr int BENCH_ITERS = 500;
constexpr unsigned int SEED = 42;

// ─── MNIST Data Loading (adapted from Applications/MNIST/jni/main.cpp) ──────

class DataInformation {
public:
  DataInformation(unsigned int num_samples, const std::string &filename,
                  unsigned int offset = 0);
  unsigned int count;
  unsigned int num_samples;
  unsigned int sample_offset; // skip first N samples (for val/test split)
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

// Read 784 floats + 10 label floats from binary file, truncate input to 768
static bool getData(std::ifstream &F, float *input, float *label,
                    unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position = (uint64_t)((FEATURE_SIZE_ORIG + NUM_CLASSES) *
                                 (uint64_t)id * sizeof(float));
  if (position > file_length) return false;
  F.seekg(position, std::ios::beg);

  // Read full 784 features into temp, copy first 768
  std::vector<float> tmp(FEATURE_SIZE_ORIG);
  F.read((char *)tmp.data(), sizeof(float) * FEATURE_SIZE_ORIG);
  std::memcpy(input, tmp.data(), sizeof(float) * FEATURE_SIZE);

  F.read((char *)label, sizeof(float) * NUM_CLASSES);
  return true;
}

int getSample_train(float **outVec, float **outLabel, bool *last,
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

// ─── Struct to hold extracted weights from a trained model ──────────────────

struct LayerWeights {
  std::string name;
  std::vector<float> weight; // flattened W
  std::vector<float> bias;   // flattened b
  unsigned int in_dim;       // rows
  unsigned int out_dim;      // cols (units)
};

struct QATStats {
  float running_min;
  float running_max;
  float scale;
  float zero_point;
};

struct ModelWeights {
  std::vector<LayerWeights> layers; // fc1, fc2, fc3, output
  std::vector<QATStats> qat_stats; // only filled for QAT model (fc1,fc2,fc3)
};

// ─── Extract weights from a trained NNTrainer model ─────────────────────────

static LayerWeights extractLayer(std::unique_ptr<Model> &model,
                                 const char *name,
                                 unsigned int in_d, unsigned int out_d) {
  LayerWeights lw;
  lw.name = name;
  lw.in_dim = in_d;
  lw.out_dim = out_d;

  std::shared_ptr<ml::train::Layer> layer;
  int ret = model->getLayer(name, &layer);
  if (ret != 0) {
    std::cerr << "ERROR: getLayer(" << name << ") failed" << std::endl;
    return lw;
  }

  std::vector<float *> wptrs;
  std::vector<TensorDim> wdims;
  layer->getWeights(wptrs, wdims);

  // wptrs[0] = weight, wptrs[1] = bias (standard FC layout)
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

  std::cerr << "  Extracted " << name << ": W[" << lw.weight.size()
            << "] b[" << lw.bias.size() << "]" << std::endl;
  return lw;
}

// ─── Build & train normal FP32 model ────────────────────────────────────────

static ModelWeights trainFP32(const std::string &data_file) {
  std::cerr << "\n=== Phase 1: Normal FP32 Training ===" << std::endl;

  auto train_data = std::make_unique<DataInformation>(NUM_TRAIN, data_file, 0);
  auto val_data = std::make_unique<DataInformation>(NUM_VAL, data_file, NUM_TRAIN);

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
                  {"name=output", "unit=10", "activation=softmax"}));

  auto optimizer = createOptimizer("sgd", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));
  model->setProperty({"epochs=" + std::to_string(EPOCHS), "loss=cross"});

  model->compile();
  model->initialize();
  model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
  model->setDataset(DatasetModeType::MODE_VALID, dataset_val);

  model->train();
  std::cerr << "FP32 training loss: " << model->getTrainingLoss() << std::endl;

  // Extract weights
  ModelWeights mw;
  std::cerr << "Extracting FP32 weights..." << std::endl;
  mw.layers.push_back(extractLayer(model, "fc1", FEATURE_SIZE, HIDDEN_DIM));
  mw.layers.push_back(extractLayer(model, "fc2", HIDDEN_DIM, HIDDEN_DIM));
  mw.layers.push_back(extractLayer(model, "fc3", HIDDEN_DIM, HIDDEN_DIM));
  mw.layers.push_back(extractLayer(model, "output", HIDDEN_DIM, NUM_CLASSES));
  return mw;
}

// ─── Build & train QAT model ────────────────────────────────────────────────

static ModelWeights trainQAT(const std::string &data_file) {
  std::cerr << "\n=== Phase 2: QAT Training ===" << std::endl;

  // Register custom QAT layer
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<nntrainer::QATFullyConnectedLayer>);
  } catch (std::invalid_argument &) {
    // Already registered
  }

  auto train_data = std::make_unique<DataInformation>(NUM_TRAIN, data_file, 0);
  auto val_data = std::make_unique<DataInformation>(NUM_VAL, data_file, NUM_TRAIN);

  std::shared_ptr<ml::train::Dataset> dataset_train =
    createDataset(DatasetType::GENERATOR, getSample_train, train_data.get());
  std::shared_ptr<ml::train::Dataset> dataset_val =
    createDataset(DatasetType::GENERATOR, getSample_train, val_data.get());

  auto model = createModel(ModelType::NEURAL_NET,
                           {"batch_size=" + std::to_string(BATCH_SIZE)});

  model->addLayer(createLayer("input", {"name=input0",
                  "input_shape=1:1:" + std::to_string(FEATURE_SIZE)}));
  // Hidden layers use QAT
  model->addLayer(createLayer("qat_fully_connected", {"name=qfc1",
                  "unit=" + std::to_string(HIDDEN_DIM)}));
  model->addLayer(createLayer("activation", {"name=relu1", "activation=relu"}));
  model->addLayer(createLayer("qat_fully_connected", {"name=qfc2",
                  "unit=" + std::to_string(HIDDEN_DIM)}));
  model->addLayer(createLayer("activation", {"name=relu2", "activation=relu"}));
  model->addLayer(createLayer("qat_fully_connected", {"name=qfc3",
                  "unit=" + std::to_string(HIDDEN_DIM)}));
  model->addLayer(createLayer("activation", {"name=relu3", "activation=relu"}));
  // Output layer stays normal FC
  model->addLayer(createLayer("fully_connected",
                  {"name=output", "unit=10", "activation=softmax"}));

  auto optimizer = createOptimizer("sgd", {"learning_rate=0.01"});
  model->setOptimizer(std::move(optimizer));
  model->setProperty({"epochs=" + std::to_string(EPOCHS), "loss=cross"});

  model->compile();
  model->initialize();
  model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
  model->setDataset(DatasetModeType::MODE_VALID, dataset_val);

  model->train();
  std::cerr << "QAT training loss: " << model->getTrainingLoss() << std::endl;

  // Extract weights
  ModelWeights mw;
  std::cerr << "Extracting QAT weights..." << std::endl;
  mw.layers.push_back(extractLayer(model, "qfc1", FEATURE_SIZE, HIDDEN_DIM));
  mw.layers.push_back(extractLayer(model, "qfc2", HIDDEN_DIM, HIDDEN_DIM));
  mw.layers.push_back(extractLayer(model, "qfc3", HIDDEN_DIM, HIDDEN_DIM));
  mw.layers.push_back(extractLayer(model, "output", HIDDEN_DIM, NUM_CLASSES));

  // Extract QAT statistics via the public getters we added
  // Note: QAT stats are printed by destructor too, but we save them here
  // We access the internal layer via dynamic_cast
  const char *qat_names[] = {"qfc1", "qfc2", "qfc3"};
  for (int i = 0; i < 3; ++i) {
    // The Layer wrapper doesn't expose QAT getters directly.
    // Instead, compute stats from weight min/max (same formula as QAT layer)
    float w_min = *std::min_element(mw.layers[i].weight.begin(),
                                     mw.layers[i].weight.end());
    float w_max = *std::max_element(mw.layers[i].weight.begin(),
                                     mw.layers[i].weight.end());
    QATStats qs;
    qs.running_min = w_min;
    qs.running_max = w_max;
    float range = std::max(w_max - w_min, 1e-8f);
    qs.scale = range / 255.0f; // [-128, 127]
    qs.zero_point = -128.0f - std::round(w_min / qs.scale);
    qs.zero_point = std::max(-128.0f, std::min(127.0f, qs.zero_point));
    mw.qat_stats.push_back(qs);
    std::cerr << "  " << qat_names[i] << " stats: min=" << w_min
              << " max=" << w_max << " scale=" << qs.scale
              << " zp=" << qs.zero_point << std::endl;
  }

  return mw;
}

// ─── Helper: Create Tensor from extracted weight data ───────────────────────

static Tensor makeWeightTensor(const LayerWeights &lw) {
  Tensor t(TensorDim(1, 1, lw.in_dim, lw.out_dim));
  float *dst = t.getData<float>();
  std::memcpy(dst, lw.weight.data(), lw.weight.size() * sizeof(float));
  return t;
}

static Tensor makeBiasTensor(const LayerWeights &lw) {
  Tensor t(TensorDim(1, 1, 1, lw.out_dim));
  float *dst = t.getData<float>();
  std::memcpy(dst, lw.bias.data(), lw.bias.size() * sizeof(float));
  return t;
}




// ─── FP16 conversion for forced-scale Q6_K ──────────────────────────────────
static uint16_t fp32_to_fp16(float f) {
  uint32_t u; memcpy(&u, &f, 4);
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
    exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF;
  } else if (exp == 31) {
    uint32_t r = sign | 0x7F800000 | (mant << 13); float f; memcpy(&f, &r, 4); return f;
  }
  uint32_t result = sign | ((exp + 112) << 23) | (mant << 13);
  float f; memcpy(&f, &result, 4); return f;
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
      ql[l + 0] = (L[j+l] & 0xF) | ((L[j+l+64] & 0xF) << 4);
      ql[l + 32] = (L[j+l+32] & 0xF) | ((L[j+l+96] & 0xF) << 4);
      qh[l] = (L[j+l] >> 4) | ((L[j+l+32] >> 4) << 2) |
              ((L[j+l+64] >> 4) << 4) | ((L[j+l+96] >> 4) << 6);
    }
    ql += 64; qh += 32;
  }
}

// ─── Build forced-scale Q6_K from weight + QAT stats ────────────────────────
static Tensor buildForcedQ6K(const Tensor &W_fp32, float qat_min, float qat_max) {
  unsigned int K = W_fp32.getDim().height();
  unsigned int N = W_fp32.getDim().width();
  Tensor W_t = W_fp32.transpose("0:2:1");
  const float *src = W_t.getData<float>();

  // Symmetric scale from QAT stats
  float amax = std::max(std::abs(qat_min), std::abs(qat_max));
  float forced_d = amax / (31.0f * 127.0f);
  if (forced_d < 1e-10f) forced_d = 1e-10f;
  uint16_t forced_d_fp16 = fp32_to_fp16(forced_d);
  float forced_d_actual = fp16_to_fp32(forced_d_fp16);

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
      for (int s = 0; s < 16; ++s) block->scales[s] = 127;
      const float *x = src + row * K + b * QK_K_LOCAL;
      uint8_t L[QK_K_LOCAL];
      float eff = forced_d_actual * 127.0f;
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

// ─── Forward pass (FP32 weights for hidden, always FP32 for output) ─────────
// W[0..2] are hidden layer weights (may be FP32 or Q6_K Tensors)
// W[3] + b[3] is the output layer (always FP32)

static void forward_pass(const Tensor &input,
                          const Tensor &W1, const Tensor &b1,
                          const Tensor &W2, const Tensor &b2,
                          const Tensor &W3, const Tensor &b3,
                          const Tensor &Wout, const Tensor &bout,
                          Tensor &logits) {
  unsigned int batch = input.getDim().batch();

  Tensor h1(TensorDim(batch, 1, 1, HIDDEN_DIM));
  input.dot(W1, h1, false, false);
  h1.add_i(b1);
  h1.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, h1);

  Tensor h2(TensorDim(batch, 1, 1, HIDDEN_DIM));
  h1.dot(W2, h2, false, false);
  h2.add_i(b2);
  h2.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, h2);

  Tensor h3(TensorDim(batch, 1, 1, HIDDEN_DIM));
  h2.dot(W3, h3, false, false);
  h3.add_i(b3);
  h3.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, h3);

  h3.dot(Wout, logits, false, false);
  logits.add_i(bout);
}

// ─── Accuracy evaluation on test set ────────────────────────────────────────

static float evaluateAccuracy(const std::string &data_file,
                               const Tensor &W1, const Tensor &b1,
                               const Tensor &W2, const Tensor &b2,
                               const Tensor &W3, const Tensor &b3,
                               const Tensor &Wout, const Tensor &bout) {
  unsigned int correct = 0;
  unsigned int total = 0;
  unsigned int test_offset = NUM_TRAIN + NUM_VAL;

  std::ifstream file(data_file, std::ios::in | std::ios::binary);
  std::vector<float> input_buf(FEATURE_SIZE);
  std::vector<float> label_buf(NUM_CLASSES);

  for (unsigned int i = 0; i < NUM_TEST; ++i) {
    if (!getData(file, input_buf.data(), label_buf.data(), test_offset + i))
      break;

    Tensor input(TensorDim(1, 1, 1, FEATURE_SIZE));
    std::memcpy(input.getData<float>(), input_buf.data(),
                FEATURE_SIZE * sizeof(float));

    Tensor logits(TensorDim(1, 1, 1, NUM_CLASSES));
    forward_pass(input, W1, b1, W2, b2, W3, b3, Wout, bout, logits);

    // argmax
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

// ─── Latency benchmark ─────────────────────────────────────────────────────

static double benchmarkLatency(const Tensor &W1, const Tensor &b1,
                                const Tensor &W2, const Tensor &b2,
                                const Tensor &W3, const Tensor &b3,
                                const Tensor &Wout, const Tensor &bout) {
  Tensor dummy_in(TensorDim(BATCH_SIZE, 1, 1, FEATURE_SIZE));
  float *din = dummy_in.getData<float>();
  std::mt19937 gen(123);
  std::normal_distribution<float> dist(0.f, 0.5f);
  for (unsigned i = 0; i < dummy_in.size(); ++i) din[i] = dist(gen);

  Tensor logits(TensorDim(BATCH_SIZE, 1, 1, NUM_CLASSES));

  auto start = Clock::now();
  for (int i = 0; i < BENCH_ITERS; ++i) {
    forward_pass(dummy_in, W1, b1, W2, b2, W3, b3, Wout, bout, logits);
  }
  auto end = Clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / BENCH_ITERS;
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mnist_trainingSet.dat>" << std::endl;
    return 1;
  }
  std::string data_file = argv[1];

  std::cerr << "=== Comprehensive MNIST Quantization POC ===" << std::endl;
  std::cerr << "Architecture: " << FEATURE_SIZE << "->" << HIDDEN_DIM << "->"
            << HIDDEN_DIM << "->" << HIDDEN_DIM << "->" << NUM_CLASSES
            << std::endl;

  // ── Phase 1 & 2: Train both models ──
  ModelWeights fp32_mw = trainFP32(data_file);
  ModelWeights qat_mw = trainQAT(data_file);

  // ── Build Tensor objects from extracted weights ──
  // FP32 model weights
  Tensor fp_W1 = makeWeightTensor(fp32_mw.layers[0]);
  Tensor fp_b1 = makeBiasTensor(fp32_mw.layers[0]);
  Tensor fp_W2 = makeWeightTensor(fp32_mw.layers[1]);
  Tensor fp_b2 = makeBiasTensor(fp32_mw.layers[1]);
  Tensor fp_W3 = makeWeightTensor(fp32_mw.layers[2]);
  Tensor fp_b3 = makeBiasTensor(fp32_mw.layers[2]);
  Tensor fp_Wout = makeWeightTensor(fp32_mw.layers[3]);
  Tensor fp_bout = makeBiasTensor(fp32_mw.layers[3]);

  // QAT model weights
  Tensor qa_W1 = makeWeightTensor(qat_mw.layers[0]);
  Tensor qa_b1 = makeBiasTensor(qat_mw.layers[0]);
  Tensor qa_W2 = makeWeightTensor(qat_mw.layers[1]);
  Tensor qa_b2 = makeBiasTensor(qat_mw.layers[1]);
  Tensor qa_W3 = makeWeightTensor(qat_mw.layers[2]);
  Tensor qa_b3 = makeBiasTensor(qat_mw.layers[2]);
  Tensor qa_Wout = makeWeightTensor(qat_mw.layers[3]);
  Tensor qa_bout = makeBiasTensor(qat_mw.layers[3]);

  // ── GGML Q6_K quantization ──
  auto q6k_quant = Quantization::createQuantizer(QScheme::Q6_K);

  // Mode 2: FP32-trained weights → Q6_K
  Tensor fp_W1_q6k = q6k_quant->quantize(fp_W1, Tdatatype::Q6_K);
  Tensor fp_W2_q6k = q6k_quant->quantize(fp_W2, Tdatatype::Q6_K);
  Tensor fp_W3_q6k = q6k_quant->quantize(fp_W3, Tdatatype::Q6_K);

  // Mode 4: QAT-trained weights → Q6_K (GGML auto scales)
  Tensor qa_W1_q6k = q6k_quant->quantize(qa_W1, Tdatatype::Q6_K);
  Tensor qa_W2_q6k = q6k_quant->quantize(qa_W2, Tdatatype::Q6_K);
  Tensor qa_W3_q6k = q6k_quant->quantize(qa_W3, Tdatatype::Q6_K);

  // Mode 5: QAT-trained weights → Forced Q6_K (QAT stats as scale)
  Tensor qa_W1_forced = buildForcedQ6K(qa_W1,
    qat_mw.qat_stats[0].running_min, qat_mw.qat_stats[0].running_max);
  Tensor qa_W2_forced = buildForcedQ6K(qa_W2,
    qat_mw.qat_stats[1].running_min, qat_mw.qat_stats[1].running_max);
  Tensor qa_W3_forced = buildForcedQ6K(qa_W3,
    qat_mw.qat_stats[2].running_min, qat_mw.qat_stats[2].running_max);

  // ── Memory stats ──
  size_t fp32_mem = fp_W1.bytes() + fp_W2.bytes() + fp_W3.bytes();
  size_t q6k_mem = fp_W1_q6k.bytes() + fp_W2_q6k.bytes() + fp_W3_q6k.bytes();

  // ── Evaluate all 5 modes ──
  std::cerr << "\n=== Phase 3: Inference Benchmark ===" << std::endl;

  struct Result {
    const char *name;
    float accuracy;
    double latency_ms;
    size_t weight_bytes;
  };

  std::vector<Result> results;

  // Mode 1: FP32 baseline
  float acc1 = evaluateAccuracy(data_file, fp_W1, fp_b1, fp_W2, fp_b2,
                                 fp_W3, fp_b3, fp_Wout, fp_bout);
  double lat1 = benchmarkLatency(fp_W1, fp_b1, fp_W2, fp_b2,
                                  fp_W3, fp_b3, fp_Wout, fp_bout);
  results.push_back({"1. FP32 Baseline", acc1, lat1, fp32_mem});

  // Mode 2: Post-train Q6_K
  float acc2 = evaluateAccuracy(data_file, fp_W1_q6k, fp_b1, fp_W2_q6k, fp_b2,
                                 fp_W3_q6k, fp_b3, fp_Wout, fp_bout);
  double lat2 = benchmarkLatency(fp_W1_q6k, fp_b1, fp_W2_q6k, fp_b2,
                                  fp_W3_q6k, fp_b3, fp_Wout, fp_bout);
  results.push_back({"2. Post-Train Q6_K", acc2, lat2, q6k_mem});

  // Mode 3: QAT → FP32 inference
  float acc3 = evaluateAccuracy(data_file, qa_W1, qa_b1, qa_W2, qa_b2,
                                 qa_W3, qa_b3, qa_Wout, qa_bout);
  double lat3 = benchmarkLatency(qa_W1, qa_b1, qa_W2, qa_b2,
                                  qa_W3, qa_b3, qa_Wout, qa_bout);
  results.push_back({"3. QAT -> FP32", acc3, lat3, fp32_mem});

  // Mode 4: QAT → Q6_K (GGML auto)
  float acc4 = evaluateAccuracy(data_file, qa_W1_q6k, qa_b1, qa_W2_q6k, qa_b2,
                                 qa_W3_q6k, qa_b3, qa_Wout, qa_bout);
  double lat4 = benchmarkLatency(qa_W1_q6k, qa_b1, qa_W2_q6k, qa_b2,
                                  qa_W3_q6k, qa_b3, qa_Wout, qa_bout);
  results.push_back({"4. QAT -> Q6_K", acc4, lat4, q6k_mem});

  // Mode 5: QAT → Forced Q6_K
  float acc5 = evaluateAccuracy(data_file, qa_W1_forced, qa_b1,
                                 qa_W2_forced, qa_b2,
                                 qa_W3_forced, qa_b3, qa_Wout, qa_bout);
  double lat5 = benchmarkLatency(qa_W1_forced, qa_b1, qa_W2_forced, qa_b2,
                                  qa_W3_forced, qa_b3, qa_Wout, qa_bout);
  results.push_back({"5. QAT -> Forced Q6_K", acc5, lat5, q6k_mem});

  // ── Print results table ──
  std::cerr << "\n╔══════════════════════════════════════════════════════════════════╗"
            << std::endl;
  std::cerr << "║              MNIST Quantization POC — Results                   ║"
            << std::endl;
  std::cerr << "╠══════════════════════════╦══════════╦════════════╦═══════════════╣"
            << std::endl;
  std::cerr << "║ Mode                     ║ Accuracy ║ Latency    ║ Hidden W Mem  ║"
            << std::endl;
  std::cerr << "╠══════════════════════════╬══════════╬════════════╬═══════════════╣"
            << std::endl;

  for (auto &r : results) {
    std::cerr << "║ " << std::setw(24) << std::left << r.name << " ║ "
              << std::setw(6) << std::fixed << std::setprecision(1)
              << r.accuracy << "% ║ "
              << std::setw(8) << std::setprecision(2) << r.latency_ms << " ms ║ "
              << std::setw(8) << r.weight_bytes / 1024 << " KB    ║"
              << std::endl;
  }

  std::cerr << "╠══════════════════════════╩══════════╩════════════╩═══════════════╣"
            << std::endl;
  std::cerr << "║ Compression: FP32=" << fp32_mem / 1024 << " KB, Q6_K="
            << q6k_mem / 1024 << " KB ("
            << std::setprecision(1) << (float)fp32_mem / q6k_mem
            << "x reduction)              ║" << std::endl;
  std::cerr << "╚═════════════════════════════════════════════════════════════════╝"
            << std::endl;

  return 0;
}
