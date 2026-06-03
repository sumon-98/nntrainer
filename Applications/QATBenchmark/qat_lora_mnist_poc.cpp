// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_lora_mnist_poc.cpp
 * @brief  POC for frozen GGML (Q6_K) base model with QAT-trained LoRA layers
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <dataset.h>
#include <model.h>
// #include <nntrainer_engine.h>
#include <optimizer.h>
#include <quantizer.h>
#include <tensor.h>

#include "qat_fc_layer.h" // The layer with LoRA QAT support

using namespace ml::train;
using namespace nntrainer;

// ─── Network & Training Config ───────────────────────────────────────────────

static constexpr unsigned int FEATURE_SIZE_ORIG = 784;
static constexpr unsigned int FEATURE_SIZE = 768; // Truncated to multiple of 256 for GGML
static constexpr unsigned int HIDDEN_DIM = 256;
static constexpr unsigned int NUM_CLASSES = 10;

static constexpr unsigned int BATCH_SIZE = 32;
static constexpr unsigned int NUM_TRAIN = 100;
static constexpr unsigned int NUM_VAL = 100;
static constexpr unsigned int EPOCHS = 5;
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

  std::vector<float> tmp(FEATURE_SIZE_ORIG);
  F.read((char *)tmp.data(), sizeof(float) * FEATURE_SIZE_ORIG);
  std::memcpy(input, tmp.data(), sizeof(float) * FEATURE_SIZE);

  F.read((char *)label, sizeof(float) * NUM_CLASSES);
  return true;
}

static int getSample_train(float **input, float **label, bool *last,
                           void *user_data) {
  DataInformation *data = static_cast<DataInformation *>(user_data);
  unsigned int id = data->idxes[data->count++];
  getData(data->file, input[0], label[0], id);
  if (data->count >= data->num_samples) {
    data->count = 0;
    std::shuffle(data->idxes.begin(), data->idxes.end(), data->rng);
    *last = true;
  } else {
    *last = false;
  }
  return ML_ERROR_NONE;
}

// ─── Weight Extraction Helpers ──────────────────────────────────────────────

struct ExtractedWeights {
  std::vector<float> weight;
};

static ExtractedWeights extractWeights(std::shared_ptr<ml::train::Model> model,
                                       const std::string &name) {
  ExtractedWeights ew;
  std::shared_ptr<ml::train::Layer> layer;
  if (model->getLayer(name, &layer) != 0) {
    std::cerr << "ERROR: getLayer(" << name << ") failed" << std::endl;
    return ew;
  }

  std::vector<float *> wptrs;
  std::vector<TensorDim> wdims;
  layer->getWeights(wptrs, wdims);

  if (wptrs.size() >= 1 && wdims.size() >= 1) {
    size_t wsz = 1;
    for (unsigned d = 0; d < wdims[0].getNumDim(); ++d)
      wsz *= wdims[0].getDim()[d];
    ew.weight.assign(wptrs[0], wptrs[0] + wsz);
  }
  return ew;
}

// ─── Phase 1: Train FP32 Baseline ───────────────────────────────────────────

static std::vector<ExtractedWeights> trainFP32(const std::string &data_file) {
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

  auto optimizer = createOptimizer("sgd", {"learning_rate=0.01"});
  model->setOptimizer(std::move(optimizer));
  model->setProperty({"epochs=" + std::to_string(EPOCHS), "loss=cross"});

  model->compile();
  model->initialize();
  model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
  model->setDataset(DatasetModeType::MODE_VALID, dataset_val);

  model->train();
  std::cerr << "FP32 training loss: " << model->getTrainingLoss() << std::endl;

  std::vector<ExtractedWeights> mw;
  mw.push_back(extractWeights(model, "fc1"));
  mw.push_back(extractWeights(model, "fc2"));
  mw.push_back(extractWeights(model, "fc3"));
  mw.push_back(extractWeights(model, "output"));
  return mw;
}

// ─── Main POC Workflow ──────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mnist_trainingSet.dat>" << std::endl;
    return 1;
  }
  std::string data_file = argv[1];

  std::cerr << "=== LoRA + QAT MNIST POC ===" << std::endl;
  
  // Register custom QAT layer
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<nntrainer::QATFullyConnectedLayer>);
  } catch (...) {}

  // 1. Train FP32 Model
  auto fp32_weights = trainFP32(data_file);

  std::cerr << "\n=== Phase 2: Quantize Base Weights to GGML Q6_K ===" << std::endl;
  auto q6k_quantizer = Quantization::createQuantizer(QScheme::Q6_K);
  
  // Dimensions for hidden layers
  TensorDim dim_w1(1, 1, FEATURE_SIZE, HIDDEN_DIM);
  TensorDim dim_w2(1, 1, HIDDEN_DIM, HIDDEN_DIM);
  TensorDim dim_w3(1, 1, HIDDEN_DIM, HIDDEN_DIM);

  Tensor w1_fp32(dim_w1); w1_fp32.copy(fp32_weights[0].weight.data());
  Tensor w2_fp32(dim_w2); w2_fp32.copy(fp32_weights[1].weight.data());
  Tensor w3_fp32(dim_w3); w3_fp32.copy(fp32_weights[2].weight.data());

  Tensor w1_q6k = q6k_quantizer->quantize(w1_fp32, Tdatatype::Q6_K);
  Tensor w2_q6k = q6k_quantizer->quantize(w2_fp32, Tdatatype::Q6_K);
  Tensor w3_q6k = q6k_quantizer->quantize(w3_fp32, Tdatatype::Q6_K);

  std::cerr << "Quantized to Q6_K. Example memory sizes:" << std::endl;
  std::cerr << "  W1 FP32: " << w1_fp32.bytes() << " bytes" << std::endl;
  std::cerr << "  W1 Q6_K: " << w1_q6k.bytes() << " bytes" << std::endl;

  std::cerr << "\n=== Phase 3: Train LoRA with QAT on Frozen Q6_K Base ===" << std::endl;

  auto train_data = std::make_unique<DataInformation>(NUM_TRAIN, data_file, 0);
  auto val_data = std::make_unique<DataInformation>(NUM_VAL, data_file, NUM_TRAIN);
  std::shared_ptr<ml::train::Dataset> dataset_train =
    createDataset(DatasetType::GENERATOR, getSample_train, train_data.get());
  std::shared_ptr<ml::train::Dataset> dataset_val =
    createDataset(DatasetType::GENERATOR, getSample_train, val_data.get());

  auto lora_model = createModel(ModelType::NEURAL_NET,
                           {"batch_size=" + std::to_string(BATCH_SIZE),
                            "model_tensor_type=Q6_K-FP32"});

  lora_model->addLayer(createLayer("input", {"name=input0",
                  "input_shape=1:1:" + std::to_string(FEATURE_SIZE)}));
  
  // Use QAT fully connected layers with LoRA enabled
  lora_model->addLayer(createLayer("qat_fully_connected", {"name=lfc1",
                  "unit=" + std::to_string(HIDDEN_DIM), "lora_rank=4"}));
  lora_model->addLayer(createLayer("activation", {"name=relu1", "activation=relu"}));
  
  lora_model->addLayer(createLayer("qat_fully_connected", {"name=lfc2",
                  "unit=" + std::to_string(HIDDEN_DIM), "lora_rank=4"}));
  lora_model->addLayer(createLayer("activation", {"name=relu2", "activation=relu"}));
  
  lora_model->addLayer(createLayer("qat_fully_connected", {"name=lfc3",
                  "unit=" + std::to_string(HIDDEN_DIM), "lora_rank=4"}));
  lora_model->addLayer(createLayer("activation", {"name=relu3", "activation=relu"}));
  
  lora_model->addLayer(createLayer("fully_connected",
                  {"name=output", "unit=10", "activation=softmax"}));

  auto optimizer = createOptimizer("adam", {"learning_rate=0.01"});
  lora_model->setOptimizer(std::move(optimizer));
  lora_model->setProperty({"epochs=" + std::to_string(EPOCHS), "loss=cross"});

  lora_model->compile();
  lora_model->initialize();
  
  // INJECT the Q6_K weights into the frozen base weight tensors!
  auto injectWeights = [&](const std::string& name, Tensor& q6k_tensor) {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    
    // Copy bytes (wptrs[0] acts as the buffer pointer)
    std::memcpy(wptrs[0], q6k_tensor.getData<uint8_t>(), q6k_tensor.bytes());
    std::cerr << "Injected Q6_K weights into " << name << std::endl;
  };
  
  injectWeights("lfc1", w1_q6k);
  injectWeights("lfc2", w2_q6k);
  injectWeights("lfc3", w3_q6k);

  // For the output layer, copy the FP32 weights directly
  auto injectFP32 = [&](const std::string& name, const std::vector<float>& fp32_vec) {
    std::shared_ptr<ml::train::Layer> layer;
    lora_model->getLayer(name, &layer);
    std::vector<float *> wptrs;
    std::vector<TensorDim> wdims;
    layer->getWeights(wptrs, wdims);
    std::memcpy(wptrs[0], fp32_vec.data(), fp32_vec.size() * sizeof(float));
  };
  injectFP32("output", fp32_weights[3].weight);

  lora_model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
  lora_model->setDataset(DatasetModeType::MODE_VALID, dataset_val);

  std::cerr << "\nTraining LoRA layers with QAT..." << std::endl;
  lora_model->train();
  
  std::cerr << "LoRA + QAT training loss: " << lora_model->getTrainingLoss() << std::endl;

  std::cerr << "\n=== Done! POC Successful ===" << std::endl;
  return 0;
}
