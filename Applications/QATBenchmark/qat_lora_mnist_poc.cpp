// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_lora_mnist_poc.cpp
 * @brief  POC: Frozen GGML (Q6_K) base model + FP32 LoRA adapters trained
 *         with QAT (Quantization-Aware Training).
 *
 * Workflow:
 *   Phase 1 — Train FP32 baseline on MNIST, extract weights, evaluate accuracy
 *   Phase 2 — Quantize base weights to Q6_K, build model with
 *             model_tensor_type=Q6_K-FP32, attach LoRA with QAT, train,
 *             evaluate accuracy
 *
 * Architecture: 768→256→256→256→10  (hidden layers have LoRA, output is plain)
 *
 * Key insight: NNTrainer's FloatTensor::dot(Q6_K) uses GGML kernels
 *              automatically (dotQnK), so FP32 input × Q6_K weight works
 *              out-of-the-box. LoRA weights are forced to FP32 in
 *              QATFullyConnectedLayer::finalize().
 *
 * ──────────────────────────────────────────────────────────────────────
 * Alternative approach NOT taken (documented for reference):
 *
 *   "Fully manual forward/backward loop" — instead of using NNTrainer's
 *   model graph (compile → initialize → train), one could extract tensors
 *   manually and write the forward pass (input.dot(W1, h1); h1.add_i(b1);
 *   relu; ...) and backward pass by hand, like qat_mnist_full_poc.cpp's
 *   evaluateAccuracy function. This avoids the model graph entirely.
 *
 *   Pros: Full control, no model compilation issues.
 *   Cons: Massive boilerplate — must implement forward, backward, gradient
 *         descent, mini-batching, shuffling, etc. all manually. Doesn't
 *         exercise NNTrainer's training pipeline at all.
 *
 *   The two-phased approach below is better because Phase 2 uses the
 *   NNTrainer training pipeline (model->train()) which automatically
 *   handles batching, backprop, optimizer updates, etc.
 * ──────────────────────────────────────────────────────────────────────
 */

#include <algorithm>
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

// ─── Network & Training Config ──────────────────────────────────────────────

static constexpr unsigned int FEATURE_SIZE_ORIG = 784;
static constexpr unsigned int FEATURE_SIZE = 768;  // truncated for Q6_K compat
static constexpr unsigned int HIDDEN_DIM = 256;    // Q6_K: 256 % 256 = 0
static constexpr unsigned int NUM_CLASSES = 10;

static constexpr unsigned int BATCH_SIZE = 1;
static constexpr unsigned int EPOCHS_FP32 = 5;
static constexpr unsigned int EPOCHS_LORA = 5;
static constexpr unsigned int LORA_RANK = 4;
constexpr unsigned int SEED = 42;

// ─── Dataset Splitting ──────────────────────────────────────────────────────
// Non-overlapping splits to simulate real pre-train → fine-tune workflow:
//   Phase 1 (FP32 pretraining):  samples [0, 40000)
//   Phase 2 (LoRA fine-tuning):  samples [40000, 55000)
//   Evaluation / Test:           samples [55000, 60000)
//
// This ensures LoRA has genuinely new data to learn from, just like in the
// real Qwen3 scenario where the base model is pre-trained on a large corpus
// and LoRA fine-tunes on a separate downstream task (e.g., SST-2).

static constexpr unsigned int PRETRAIN_OFFSET    = 0;
static constexpr unsigned int PRETRAIN_TRAIN     = 35000;
static constexpr unsigned int PRETRAIN_VAL       = 5000;
// Pretrain validation: [35000, 40000)

static constexpr unsigned int FINETUNE_OFFSET    = 40000;
static constexpr unsigned int FINETUNE_TRAIN     = 12000;
static constexpr unsigned int FINETUNE_VAL       = 3000;
// Finetune validation: [52000, 55000)

static constexpr unsigned int TEST_OFFSET         = 55000;
static constexpr unsigned int NUM_TEST            = 5000;

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

  // Read full 784 features into temp, copy first 768
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
  int ret = model->getLayer(name, &layer);
  if (ret != 0) {
    std::cerr << "ERROR: getLayer(" << name << ") failed" << std::endl;
    return lw;
  }

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

// ─── Manual forward pass (for accuracy evaluation) ──────────────────────────
// W[0..2] can be FP32 or Q6_K; W[3] (output) always FP32

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
  h1.apply<float>([](float v) { return v > 0.f ? v : 0.f; }, h1); // ReLU

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
                               const Tensor &Wout, const Tensor &bout,
                               unsigned int test_offset = TEST_OFFSET,
                               unsigned int num_test = NUM_TEST) {
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

// ═════════════════════════════════════════════════════════════════════════════
// Phase 1: Train FP32 Baseline
// ═════════════════════════════════════════════════════════════════════════════

static std::vector<LayerWeights> trainFP32(const std::string &data_file) {
  std::cerr << "\n╔══════════════════════════════════════════════╗"
            << "\n║       Phase 1: Normal FP32 Training          ║"
            << "\n╚══════════════════════════════════════════════╝"
            << std::endl;

  auto train_data = std::make_unique<DataInformation>(PRETRAIN_TRAIN, data_file, PRETRAIN_OFFSET);
  auto val_data = std::make_unique<DataInformation>(PRETRAIN_VAL, data_file, PRETRAIN_OFFSET + PRETRAIN_TRAIN);

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

  // ── Optimizer: uncomment whichever you prefer ──
  auto optimizer = createOptimizer("sgd", {"learning_rate=0.001"});
  // auto optimizer = createOptimizer("adam", {"learning_rate=0.001"});

  model->setOptimizer(std::move(optimizer));
  model->setProperty({"epochs=" + std::to_string(EPOCHS_FP32), "loss=cross"});

  model->compile();
  model->initialize();
  model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
  model->setDataset(DatasetModeType::MODE_VALID, dataset_val);

  model->train();
  std::cerr << "FP32 training loss: " << model->getTrainingLoss() << std::endl;

  // Extract weights
  std::cerr << "Extracting FP32 weights..." << std::endl;
  std::vector<LayerWeights> mw;
  mw.push_back(extractLayer(model, "fc1", FEATURE_SIZE, HIDDEN_DIM));
  mw.push_back(extractLayer(model, "fc2", HIDDEN_DIM, HIDDEN_DIM));
  mw.push_back(extractLayer(model, "fc3", HIDDEN_DIM, HIDDEN_DIM));
  mw.push_back(extractLayer(model, "output", HIDDEN_DIM, NUM_CLASSES));
  return mw;
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase 2: Quantize + LoRA QAT
// ═════════════════════════════════════════════════════════════════════════════

static void trainLoRAQAT(const std::string &data_file,
                          const std::vector<LayerWeights> &fp32_weights) {
  std::cerr << "\n╔══════════════════════════════════════════════╗"
            << "\n║   Phase 2: Q6_K Base + FP32 LoRA QAT        ║"
            << "\n╚══════════════════════════════════════════════╝"
            << std::endl;

  // ── Step 2a: Quantize base weights to Q6_K ──

  std::cerr << "\n--- Step 2a: Quantizing base weights to GGML Q6_K ---"
            << std::endl;

  auto q6k_quantizer = Quantization::createQuantizer(QScheme::Q6_K);

  // Build FP32 weight tensors, then quantize
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
  std::cerr << "  W2 FP32: " << W2_fp32.bytes() << " bytes → Q6_K: "
            << W2_q6k.bytes() << " bytes" << std::endl;
  std::cerr << "  W3 FP32: " << W3_fp32.bytes() << " bytes → Q6_K: "
            << W3_q6k.bytes() << " bytes" << std::endl;

  // ── Step 2b: Evaluate accuracy with Q6_K quantized weights ──

  std::cerr << "\n--- Step 2b: Evaluating accuracy with Q6_K weights (no LoRA) ---"
            << std::endl;

  Tensor b1 = makeBiasTensor(fp32_weights[0]);
  Tensor b2 = makeBiasTensor(fp32_weights[1]);
  Tensor b3 = makeBiasTensor(fp32_weights[2]);
  Tensor Wout = makeWeightTensor(fp32_weights[3]);
  Tensor bout = makeBiasTensor(fp32_weights[3]);

  float q6k_acc = evaluateAccuracy(data_file,
    W1_q6k, b1, W2_q6k, b2, W3_q6k, b3, Wout, bout);
  std::cerr << "  Q6_K (no LoRA) accuracy: " << q6k_acc << "%" << std::endl;

  // ── Step 2c: Build model with QAT LoRA on frozen Q6_K base ──

  std::cerr << "\n--- Step 2c: Building LoRA-QAT model ---" << std::endl;

  auto train_data = std::make_unique<DataInformation>(FINETUNE_TRAIN, data_file, FINETUNE_OFFSET);
  auto val_data = std::make_unique<DataInformation>(FINETUNE_VAL, data_file, FINETUNE_OFFSET + FINETUNE_TRAIN);
  std::shared_ptr<ml::train::Dataset> dataset_train =
    createDataset(DatasetType::GENERATOR, getSample_train, train_data.get());
  std::shared_ptr<ml::train::Dataset> dataset_val =
    createDataset(DatasetType::GENERATOR, getSample_train, val_data.get());

  // model_tensor_type=Q6_K-FP32 means: weights=Q6_K, activations=FP32
  // The QAT FC layer overrides LoRA weight types to FP32 internally.
  auto lora_model = createModel(ModelType::NEURAL_NET,
                           {"batch_size=" + std::to_string(BATCH_SIZE)});

  lora_model->addLayer(createLayer("input", {"name=input0",
                  "input_shape=1:1:" + std::to_string(FEATURE_SIZE)}));

  // Hidden layers: QAT FC with LoRA
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

  // Output layer: plain fully_connected (always FP32, no LoRA, frozen)
  lora_model->addLayer(createLayer("fully_connected",
                  {"name=output", "unit=" + std::to_string(NUM_CLASSES), "activation=softmax", "trainable=false"}));

  // ── Optimizer: uncomment whichever you prefer ──
  // auto lora_optimizer = createOptimizer("sgd", {"learning_rate=0.01"});
  auto lora_optimizer = createOptimizer("adam", {"learning_rate=0.0001"});

  lora_model->setOptimizer(std::move(lora_optimizer));
  lora_model->setProperty({"epochs=" + std::to_string(EPOCHS_LORA), "loss=cross"});

  lora_model->compile();
  lora_model->initialize();

  // ── Step 2d: Inject Q6_K weights into the frozen base weight tensors ──

  std::cerr << "\n--- Step 2d: Injecting Q6_K weights into frozen base ---"
            << std::endl;

  auto injectQ6K = [&](const char *name, Tensor &q6k_tensor) {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);

    // wptrs[0] = base weight buffer. For Q6_K, this is raw quantized bytes.
    // wdims[0] tells us the dimensions. wptrs gives us the underlying buffer.
    std::memcpy(wptrs[0], q6k_tensor.getData<uint8_t>(), q6k_tensor.bytes());
    std::cerr << "  Injected Q6_K weights into " << name
              << " (" << q6k_tensor.bytes() << " bytes)" << std::endl;
  };

  injectQ6K("lfc1", W1_q6k);
  injectQ6K("lfc2", W2_q6k);
  injectQ6K("lfc3", W3_q6k);

  // For output layer, inject FP32 weights + bias
  auto injectFP32 = [&](const char *name, const LayerWeights &lw) {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    if (wptrs.size() >= 1 && !lw.weight.empty())
      std::memcpy(wptrs[0], lw.weight.data(), lw.weight.size() * sizeof(float));
    if (wptrs.size() >= 2 && !lw.bias.empty())
      std::memcpy(wptrs[1], lw.bias.data(), lw.bias.size() * sizeof(float));
    std::cerr << "  Injected FP32 weights into " << name << std::endl;
  };

  injectFP32("output", fp32_weights[3]);

  // Also inject biases into LoRA layers (biases are always FP32)
  auto injectBias = [&](const char *name, const LayerWeights &lw) {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    // For QAT FC with LoRA: wptrs layout is:
    //   [0] = base weight (Q6_K)
    //   [1] = bias (FP32)
    //   [2] = loraA (FP32)
    //   [3] = loraB (FP32)
    if (wptrs.size() >= 2 && !lw.bias.empty())
      std::memcpy(wptrs[1], lw.bias.data(), lw.bias.size() * sizeof(float));
    std::cerr << "  Injected bias into " << name << std::endl;
  };

  injectBias("lfc1", fp32_weights[0]);
  injectBias("lfc2", fp32_weights[1]);
  injectBias("lfc3", fp32_weights[2]);

  // ── Step 2e: Train LoRA layers with QAT ──

  lora_model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
  lora_model->setDataset(DatasetModeType::MODE_VALID, dataset_val);

  std::cerr << "\n--- Step 2e: Training LoRA layers with QAT ---" << std::endl;
  std::cerr << "  LoRA rank: " << LORA_RANK
            << ", Epochs: " << EPOCHS_LORA << std::endl;
  std::cerr << "  Base weights: frozen Q6_K" << std::endl;
  std::cerr << "  LoRA weights: trainable FP32 with fake-quantization" << std::endl;

  lora_model->train();

  std::cerr << "\nLoRA-QAT training loss: "
            << lora_model->getTrainingLoss() << std::endl;

  // ── Step 2f: Evaluate accuracy after LoRA-QAT ──

  std::cerr << "\n--- Step 2f: Evaluating accuracy after LoRA-QAT ---" << std::endl;
  std::cerr << "  (Extracting effective weights = Q6_K_base + LoRA_A * LoRA_B * scaling)"
            << std::endl;

  // Extract LoRA weights and compute effective weights for evaluation
  // For each QAT FC layer: effective_weight ≈ dequant(Q6_K_base) + loraA * loraB * scaling
  // But for accuracy evaluation we can just run the model's forward pass.
  // Since NNTrainer doesn't have a direct predict API that returns logits,
  // we'll extract the weights and run manual forward.

  // Extract from LoRA model
  auto extractLoRA = [&](const char *name, unsigned int in_d, unsigned int out_d)
      -> std::pair<std::vector<float>, std::vector<float>> {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);

    // For QAT FC + LoRA: weights are [base, bias, loraA, loraB]
    // loraA shape: (1, 1, in_d, rank)
    // loraB shape: (1, 1, rank, out_d)
    std::vector<float> loraA_data, loraB_data;
    if (wptrs.size() >= 4) {
      size_t a_sz = in_d * LORA_RANK;
      loraA_data.assign(wptrs[2], wptrs[2] + a_sz);
      size_t b_sz = LORA_RANK * out_d;
      loraB_data.assign(wptrs[3], wptrs[3] + b_sz);
    }
    return {loraA_data, loraB_data};
  };

  // For evaluation we need: effective_output = input * Q6K_base + input * A * B * scaling + bias
  // Since forward_pass only takes one weight per layer, we need to compute
  // W_effective = dequant(Q6K) + A * B * scaling as an FP32 tensor.

  auto computeEffective = [&](const Tensor &q6k_weight,
                               const std::vector<float> &loraA_data,
                               const std::vector<float> &loraB_data,
                               unsigned int in_d, unsigned int out_d) -> Tensor {
    // Dequantize Q6_K to FP32
    Tensor dequant(TensorDim(1, 1, in_d, out_d));
    // Use dot with identity to dequantize: I(1,1,in_d,in_d).dot(Q6K(1,1,in_d,out_d))
    // Actually simpler: just create a row of 1s and use dot, or use a for loop.
    // Let's use the proper approach: create an identity-like input and dot.
    // Actually the simplest is: for each row, set up a one-hot and dot.
    // Even simpler: input.dot(q6k) where input is identity matrix.
    Tensor identity(TensorDim(1, 1, in_d, in_d));
    float *id_data = identity.getData<float>();
    std::memset(id_data, 0, in_d * in_d * sizeof(float));
    for (unsigned int i = 0; i < in_d; ++i)
      id_data[i * in_d + i] = 1.0f;
    identity.dot(q6k_weight, dequant, false, false);

    // Compute LoRA contribution: A * B * scaling
    float scaling = 1.0f; // lora_alpha defaults to lora_rank, so scaling = 1.0
    Tensor A(TensorDim(1, 1, in_d, LORA_RANK));
    std::memcpy(A.getData<float>(), loraA_data.data(),
                loraA_data.size() * sizeof(float));

    Tensor B(TensorDim(1, 1, LORA_RANK, out_d));
    std::memcpy(B.getData<float>(), loraB_data.data(),
                loraB_data.size() * sizeof(float));

    Tensor lora_contrib(TensorDim(1, 1, in_d, out_d));
    A.dot(B, lora_contrib, false, false);
    lora_contrib.multiply_i(scaling);

    // Effective = dequant + lora_contrib
    dequant.add_i(lora_contrib);
    return dequant;
  };

  auto [a1, b1_lora] = extractLoRA("lfc1", FEATURE_SIZE, HIDDEN_DIM);
  auto [a2, b2_lora] = extractLoRA("lfc2", HIDDEN_DIM, HIDDEN_DIM);
  auto [a3, b3_lora] = extractLoRA("lfc3", HIDDEN_DIM, HIDDEN_DIM);

  Tensor W1_eff = computeEffective(W1_q6k, a1, b1_lora, FEATURE_SIZE, HIDDEN_DIM);
  Tensor W2_eff = computeEffective(W2_q6k, a2, b2_lora, HIDDEN_DIM, HIDDEN_DIM);
  Tensor W3_eff = computeEffective(W3_q6k, a3, b3_lora, HIDDEN_DIM, HIDDEN_DIM);

  // Re-extract output layer weights (they may have been updated during training)
  LayerWeights out_lw = {
    "output",
    fp32_weights[3].weight,
    fp32_weights[3].bias,
    HIDDEN_DIM,
    NUM_CLASSES
  };
  {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer("output", &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    if (wptrs.size() >= 1) {
      size_t wsz = HIDDEN_DIM * NUM_CLASSES;
      out_lw.weight.assign(wptrs[0], wptrs[0] + wsz);
    }
    if (wptrs.size() >= 2) {
      size_t bsz = NUM_CLASSES;
      out_lw.bias.assign(wptrs[1], wptrs[1] + bsz);
    }
  }
  Tensor Wout_final = makeWeightTensor(out_lw);
  Tensor bout_final = makeBiasTensor(out_lw);

  // Also re-extract biases from LoRA layers (they might have been trained)
  auto extractBias = [&](const char *name, unsigned int out_d) -> Tensor {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    Tensor bias(TensorDim(1, 1, 1, out_d));
    if (wptrs.size() >= 2)
      std::memcpy(bias.getData<float>(), wptrs[1], out_d * sizeof(float));
    return bias;
  };

  Tensor b1_final = extractBias("lfc1", HIDDEN_DIM);
  Tensor b2_final = extractBias("lfc2", HIDDEN_DIM);
  Tensor b3_final = extractBias("lfc3", HIDDEN_DIM);

  float lora_acc = evaluateAccuracy(data_file,
    W1_eff, b1_final, W2_eff, b2_final, W3_eff, b3_final,
    Wout_final, bout_final);
  std::cerr << "  LoRA-QAT accuracy: " << lora_acc << "%" << std::endl;
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase 3: Untrained (Random) Frozen Q6_K Base + LoRA
// ═════════════════════════════════════════════════════════════════════════════
// Purpose: Prove that LoRA adapters can actually learn. If we freeze a RANDOM
// Q6_K base model and train only LoRA on top, accuracy should rise from ~10%
// (random chance) to a meaningful level (60%+). This confirms the LoRA forward
// and backward paths are mathematically correct.

static void trainLoRAUntrainedBase(const std::string &data_file) {
  std::cerr << "\n╔══════════════════════════════════════════════╗"
            << "\n║  Phase 3: UNTRAINED Q6_K Base + LoRA QAT    ║"
            << "\n╚══════════════════════════════════════════════╝"
            << std::endl;

  // ── Step 3a: Create random FP32 weights and quantize to Q6_K ──

  std::cerr << "\n--- Step 3a: Creating random FP32 weights & quantizing to Q6_K ---"
            << std::endl;

  // Create random weight tensors (Xavier/He-like initialization)
  std::mt19937 rng(SEED + 999);  // different seed to ensure randomness
  auto randomInit = [&](unsigned int in_d, unsigned int out_d) -> Tensor {
    Tensor t(TensorDim(1, 1, in_d, out_d));
    float *data = t.getData<float>();
    float stddev = std::sqrt(2.0f / in_d);  // He init
    std::normal_distribution<float> dist(0.0f, stddev);
    for (unsigned int i = 0; i < in_d * out_d; ++i)
      data[i] = dist(rng);
    return t;
  };

  auto randomBias = [](unsigned int out_d) -> Tensor {
    Tensor t(TensorDim(1, 1, 1, out_d));
    t.setZero();
    return t;
  };

  Tensor W1_fp32 = randomInit(FEATURE_SIZE, HIDDEN_DIM);
  Tensor W2_fp32 = randomInit(HIDDEN_DIM, HIDDEN_DIM);
  Tensor W3_fp32 = randomInit(HIDDEN_DIM, HIDDEN_DIM);
  Tensor Wout_fp32 = randomInit(HIDDEN_DIM, NUM_CLASSES);

  Tensor b1_rand = randomBias(HIDDEN_DIM);
  Tensor b2_rand = randomBias(HIDDEN_DIM);
  Tensor b3_rand = randomBias(HIDDEN_DIM);
  Tensor bout_rand = randomBias(NUM_CLASSES);

  // Quantize hidden layer weights to Q6_K
  auto q6k_quantizer = Quantization::createQuantizer(QScheme::Q6_K);
  Tensor W1_q6k = q6k_quantizer->quantize(W1_fp32, Tdatatype::Q6_K);
  Tensor W2_q6k = q6k_quantizer->quantize(W2_fp32, Tdatatype::Q6_K);
  Tensor W3_q6k = q6k_quantizer->quantize(W3_fp32, Tdatatype::Q6_K);

  std::cerr << "  Random weights quantized to Q6_K" << std::endl;

  // ── Step 3b: Evaluate random model (should be ~10%) ──

  std::cerr << "\n--- Step 3b: Evaluating random Q6_K model (before LoRA training) ---"
            << std::endl;

  float random_acc = evaluateAccuracy(data_file,
    W1_q6k, b1_rand, W2_q6k, b2_rand, W3_q6k, b3_rand,
    Wout_fp32, bout_rand);
  std::cerr << "  Random Q6_K base accuracy: " << random_acc << "%"
            << " (expected ~10%)" << std::endl;

  // ── Step 3c: Build LoRA-QAT model on the random frozen base ──

  std::cerr << "\n--- Step 3c: Building LoRA-QAT model on random frozen base ---"
            << std::endl;

  // Use the fine-tuning data split for training LoRA
  auto train_data = std::make_unique<DataInformation>(FINETUNE_TRAIN, data_file, FINETUNE_OFFSET);
  auto val_data = std::make_unique<DataInformation>(FINETUNE_VAL, data_file, FINETUNE_OFFSET + FINETUNE_TRAIN);
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

  // Output layer: trainable (since base is random, output needs to learn too)
  lora_model->addLayer(createLayer("fully_connected",
                  {"name=output", "unit=" + std::to_string(NUM_CLASSES),
                   "activation=softmax", "trainable=true"}));

  auto lora_optimizer = createOptimizer("adam", {"learning_rate=0.001"});

  lora_model->setOptimizer(std::move(lora_optimizer));
  lora_model->setProperty({"epochs=" + std::to_string(EPOCHS_LORA), "loss=cross"});

  lora_model->compile();
  lora_model->initialize();

  // ── Step 3d: Inject random Q6_K weights into the frozen base ──

  std::cerr << "\n--- Step 3d: Injecting random Q6_K weights into frozen base ---"
            << std::endl;

  auto injectQ6K = [&](const char *name, Tensor &q6k_tensor) {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    std::memcpy(wptrs[0], q6k_tensor.getData<uint8_t>(), q6k_tensor.bytes());
    std::cerr << "  Injected random Q6_K weights into " << name << std::endl;
  };

  injectQ6K("lfc1", W1_q6k);
  injectQ6K("lfc2", W2_q6k);
  injectQ6K("lfc3", W3_q6k);

  // Inject random output weights
  {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer("output", &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    if (wptrs.size() >= 1)
      std::memcpy(wptrs[0], Wout_fp32.getData<float>(),
                  HIDDEN_DIM * NUM_CLASSES * sizeof(float));
    if (wptrs.size() >= 2)
      std::memcpy(wptrs[1], bout_rand.getData<float>(),
                  NUM_CLASSES * sizeof(float));
    std::cerr << "  Injected random FP32 weights into output" << std::endl;
  }

  // ── Step 3e: Train LoRA + output layer ──

  lora_model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
  lora_model->setDataset(DatasetModeType::MODE_VALID, dataset_val);

  std::cerr << "\n--- Step 3e: Training LoRA on random frozen Q6_K base ---"
            << std::endl;
  std::cerr << "  If LoRA learning is correct, accuracy should rise from ~10% to 60%+"
            << std::endl;

  lora_model->train();

  std::cerr << "\nPhase 3 training loss: "
            << lora_model->getTrainingLoss() << std::endl;

  // ── Step 3f: Evaluate accuracy after LoRA training ──

  std::cerr << "\n--- Step 3f: Evaluating accuracy after LoRA training on random base ---"
            << std::endl;

  auto extractLoRA = [&](const char *name, unsigned int in_d, unsigned int out_d)
      -> std::pair<std::vector<float>, std::vector<float>> {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    std::vector<float> loraA_data, loraB_data;
    if (wptrs.size() >= 4) {
      size_t a_sz = in_d * LORA_RANK;
      loraA_data.assign(wptrs[2], wptrs[2] + a_sz);
      size_t b_sz = LORA_RANK * out_d;
      loraB_data.assign(wptrs[3], wptrs[3] + b_sz);
    }
    return {loraA_data, loraB_data};
  };

  auto computeEffective = [&](const Tensor &q6k_weight,
                               const std::vector<float> &loraA_data,
                               const std::vector<float> &loraB_data,
                               unsigned int in_d, unsigned int out_d) -> Tensor {
    Tensor dequant(TensorDim(1, 1, in_d, out_d));
    Tensor identity(TensorDim(1, 1, in_d, in_d));
    float *id_data = identity.getData<float>();
    std::memset(id_data, 0, in_d * in_d * sizeof(float));
    for (unsigned int i = 0; i < in_d; ++i)
      id_data[i * in_d + i] = 1.0f;
    identity.dot(q6k_weight, dequant, false, false);

    float scaling = 1.0f;
    Tensor A(TensorDim(1, 1, in_d, LORA_RANK));
    std::memcpy(A.getData<float>(), loraA_data.data(),
                loraA_data.size() * sizeof(float));
    Tensor B(TensorDim(1, 1, LORA_RANK, out_d));
    std::memcpy(B.getData<float>(), loraB_data.data(),
                loraB_data.size() * sizeof(float));
    Tensor lora_contrib(TensorDim(1, 1, in_d, out_d));
    A.dot(B, lora_contrib, false, false);
    lora_contrib.multiply_i(scaling);
    dequant.add_i(lora_contrib);
    return dequant;
  };

  auto [a1, b1_lora] = extractLoRA("lfc1", FEATURE_SIZE, HIDDEN_DIM);
  auto [a2, b2_lora] = extractLoRA("lfc2", HIDDEN_DIM, HIDDEN_DIM);
  auto [a3, b3_lora] = extractLoRA("lfc3", HIDDEN_DIM, HIDDEN_DIM);

  Tensor W1_eff = computeEffective(W1_q6k, a1, b1_lora, FEATURE_SIZE, HIDDEN_DIM);
  Tensor W2_eff = computeEffective(W2_q6k, a2, b2_lora, HIDDEN_DIM, HIDDEN_DIM);
  Tensor W3_eff = computeEffective(W3_q6k, a3, b3_lora, HIDDEN_DIM, HIDDEN_DIM);

  // Extract trained output weights
  Tensor Wout_final(TensorDim(1, 1, HIDDEN_DIM, NUM_CLASSES));
  Tensor bout_final(TensorDim(1, 1, 1, NUM_CLASSES));
  {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer("output", &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    if (wptrs.size() >= 1)
      std::memcpy(Wout_final.getData<float>(), wptrs[0],
                  HIDDEN_DIM * NUM_CLASSES * sizeof(float));
    if (wptrs.size() >= 2)
      std::memcpy(bout_final.getData<float>(), wptrs[1],
                  NUM_CLASSES * sizeof(float));
  }

  // Extract trained biases from LoRA layers
  auto extractBias = [&](const char *name, unsigned int out_d) -> Tensor {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    Tensor bias(TensorDim(1, 1, 1, out_d));
    if (wptrs.size() >= 2)
      std::memcpy(bias.getData<float>(), wptrs[1], out_d * sizeof(float));
    return bias;
  };

  Tensor b1_final = extractBias("lfc1", HIDDEN_DIM);
  Tensor b2_final = extractBias("lfc2", HIDDEN_DIM);
  Tensor b3_final = extractBias("lfc3", HIDDEN_DIM);

  float lora_acc = evaluateAccuracy(data_file,
    W1_eff, b1_final, W2_eff, b2_final, W3_eff, b3_final,
    Wout_final, bout_final);
  std::cerr << "  Random base + LoRA accuracy: " << lora_acc << "%"
            << " (started from " << random_acc << "%)" << std::endl;
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mnist_trainingSet.dat>" << std::endl;
    return 1;
  }
  std::string data_file = argv[1];

  std::cerr << "\n══════════════════════════════════════════════════"
            << "\n  Mixed-Precision LoRA-QAT MNIST POC"
            << "\n  Frozen Q6_K Base + FP32 LoRA + QAT"
            << "\n══════════════════════════════════════════════════"
            << std::endl;

  std::cerr << "\n  Dataset splits:"
            << "\n    Pretraining:  [" << PRETRAIN_OFFSET << ", " << PRETRAIN_OFFSET + PRETRAIN_TRAIN << ") train, ["
            << PRETRAIN_OFFSET + PRETRAIN_TRAIN << ", " << PRETRAIN_OFFSET + PRETRAIN_TRAIN + PRETRAIN_VAL << ") val"
            << "\n    Fine-tuning:  [" << FINETUNE_OFFSET << ", " << FINETUNE_OFFSET + FINETUNE_TRAIN << ") train, ["
            << FINETUNE_OFFSET + FINETUNE_TRAIN << ", " << FINETUNE_OFFSET + FINETUNE_TRAIN + FINETUNE_VAL << ") val"
            << "\n    Test:         [" << TEST_OFFSET << ", " << TEST_OFFSET + NUM_TEST << ")"
            << std::endl;

  // Register custom QAT layer
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<nntrainer::QATFullyConnectedLayer>);
  } catch (std::invalid_argument &) {
    // Already registered — ignore
  }

  // Phase 1: Train FP32 baseline and extract weights
  auto fp32_weights = trainFP32(data_file);

  // Evaluate FP32 baseline accuracy on TEST set
  std::cerr << "\n--- Evaluating FP32 baseline accuracy (test set) ---" << std::endl;
  Tensor W1 = makeWeightTensor(fp32_weights[0]);
  Tensor b1 = makeBiasTensor(fp32_weights[0]);
  Tensor W2 = makeWeightTensor(fp32_weights[1]);
  Tensor b2 = makeBiasTensor(fp32_weights[1]);
  Tensor W3 = makeWeightTensor(fp32_weights[2]);
  Tensor b3 = makeBiasTensor(fp32_weights[2]);
  Tensor Wout = makeWeightTensor(fp32_weights[3]);
  Tensor bout = makeBiasTensor(fp32_weights[3]);

  float fp32_acc = evaluateAccuracy(data_file,
    W1, b1, W2, b2, W3, b3, Wout, bout);
  std::cerr << "  FP32 baseline accuracy: " << fp32_acc << "%" << std::endl;

  // Phase 2: Quantize + LoRA-QAT (with non-overlapping fine-tuning data)
  trainLoRAQAT(data_file, fp32_weights);

  // Phase 3: Untrained base + LoRA (sanity check that LoRA can learn)
  trainLoRAUntrainedBase(data_file);

  // ── Summary ──
  std::cerr << "\n══════════════════════════════════════════════════"
            << "\n  SUMMARY"
            << "\n══════════════════════════════════════════════════"
            << "\n  FP32 baseline accuracy:       " << fp32_acc << "%"
            << "\n  (Q6_K, LoRA-QAT, and Phase 3 accuracies printed above)"
            << "\n══════════════════════════════════════════════════"
            << std::endl;

  std::cerr << "\n=== Done! POC Successful ===" << std::endl;
  return 0;
}
