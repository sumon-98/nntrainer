// SPDX-License-Identifier: Apache-2.0
/**
 * @file   main_qat_mnist.cpp
 * @brief  QAT Proof of Concept with Real MNIST Dataset
 *
 * Uses actual MNIST data for training with QAT layers.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <fstream>
#include <numeric>
#include <cstring>

#include <model.h>
#include <optimizer.h>
#include <layer.h>
#include <dataset.h>
#include <app_context.h>
#include <engine.h>

#include "qat_fc_layer.h"

using namespace ml::train;

// ============================================================================
// Constants
// ============================================================================
constexpr unsigned int SEED = 0;
constexpr unsigned int FEATURE_SIZE = 784;
constexpr unsigned int TOTAL_LABEL_SIZE = 10;
constexpr unsigned int BATCH_SIZE = 32;

// Adjust these based on your data file
const unsigned int total_train_data_size = 100;
const unsigned int total_val_data_size = 100;

// ============================================================================
// DataInformation Class (handles file I/O, shuffling)
// ============================================================================
class DataInformation {
public:
  DataInformation(unsigned int num_samples, const std::string &filename);
  unsigned int count;
  unsigned int num_samples;
  std::ifstream file;
  std::vector<unsigned int> idxes;
  std::mt19937 rng;
};

DataInformation::DataInformation(unsigned int num_samples,
                                 const std::string &filename) :
  count(0),
  num_samples(num_samples),
  file(filename, std::ios::in | std::ios::binary),
  idxes(num_samples) {
  std::iota(idxes.begin(), idxes.end(), 0);
  rng.seed(SEED);
  std::shuffle(idxes.begin(), idxes.end(), rng);
  if (!file.good()) {
    throw std::invalid_argument("given file is not good, filename: " +
                                filename);
  }
}

// ============================================================================
// getData Function (reads a specific sample from binary file)
// ============================================================================
bool getData(std::ifstream &F, float *input, float *label, unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position = (uint64_t)((FEATURE_SIZE + TOTAL_LABEL_SIZE) *
                                 (uint64_t)id * sizeof(float));

  if (position > file_length) {
    return false;
  }
  F.seekg(position, std::ios::beg);
  F.read((char *)input, sizeof(float) * FEATURE_SIZE);
  F.read((char *)label, sizeof(float) * TOTAL_LABEL_SIZE);

  return true;
}

// ============================================================================
// getSample Callback (feeds data to NNTrainer)
// ============================================================================
int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
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

// ============================================================================
// Test A: Built-in fully_connected layers (baseline)
// ============================================================================
int test_builtin_fc(const std::string &data_file) {
  std::cerr << "\n=== TEST A: Built-in FC layers only ===" << std::endl;

  std::unique_ptr<DataInformation> train_user_data;
  std::unique_ptr<DataInformation> val_user_data;
  
  try {
    train_user_data = std::make_unique<DataInformation>(total_train_data_size, data_file);
    val_user_data = std::make_unique<DataInformation>(total_val_data_size, data_file);
    std::cerr << "[A] Data loaded from: " << data_file << std::endl;
  } catch (std::invalid_argument &e) {
    std::cerr << "[A] Error loading data: " << e.what() << std::endl;
    return 1;
  }

  try {
    auto model = createModel(ModelType::NEURAL_NET,
                              {"batch_size=" + std::to_string(BATCH_SIZE)});
    std::cerr << "[A] Model created" << std::endl;

    model->addLayer(createLayer("input", {"name=input0",
                                           "input_shape=1:1:784"}));
    std::cerr << "[A] Input layer added" << std::endl;

    model->addLayer(createLayer("fully_connected",
                                {"name=fc1", "unit=128"}));
    std::cerr << "[A] FC1 layer added" << std::endl;

    model->addLayer(createLayer("activation",
                                {"name=relu1", "activation=relu"}));
    std::cerr << "[A] ReLU layer added" << std::endl;

    model->addLayer(createLayer("fully_connected",
                                {"name=fc2", "unit=10", "activation=softmax"}));
    std::cerr << "[A] FC2 layer added" << std::endl;

    auto optimizer = createOptimizer("adam", {"learning_rate=0.001"});
    model->setOptimizer(std::move(optimizer));
    std::cerr << "[A] Optimizer set" << std::endl;

    model->setProperty({"epochs=5", "loss=cross"});
    std::cerr << "[A] Properties set" << std::endl;

    auto dataset_train = std::shared_ptr<Dataset>(
      createDataset(DatasetType::GENERATOR, getSample, train_user_data.get()));
    auto dataset_val = std::shared_ptr<Dataset>(
      createDataset(DatasetType::GENERATOR, getSample, val_user_data.get()));
    
    model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
    model->setDataset(DatasetModeType::MODE_VALID, dataset_val);
    std::cerr << "[A] Dataset set" << std::endl;

    model->compile();
    std::cerr << "[A] Compiled OK" << std::endl;

    model->initialize();
    std::cerr << "[A] Initialized OK" << std::endl;

    std::cerr << "[A] Calling train()..." << std::endl;
    model->train();
    std::cerr << "[A] Training COMPLETED successfully!" << std::endl;
    std::cerr << "[A] Training Loss: " << model->getTrainingLoss() << std::endl;
    std::cerr << "[A] Validation Loss: " << model->getValidationLoss() << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "[A] ERROR: " << e.what() << std::endl;
    return 1;
  }
}

// ============================================================================
// Test B: Custom QAT fully_connected layers
// ============================================================================
int test_custom_qat_fc(const std::string &data_file) {
  std::cerr << "\n=== TEST B: Custom QAT FC layers ===" << std::endl;

  try {
    // Register custom layer
    auto &ct_engine = nntrainer::Engine::Global();
    auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));
    app_context->registerFactory(
      nntrainer::createLayer<nntrainer::QATFullyConnectedLayer>);
    std::cerr << "[B] Custom layer registered" << std::endl;
  } catch (std::invalid_argument &e) {
    std::cerr << "[B] Registration failed: " << e.what() << std::endl;
    return 1;
  }

  std::unique_ptr<DataInformation> train_user_data;
  std::unique_ptr<DataInformation> val_user_data;
  
  try {
    train_user_data = std::make_unique<DataInformation>(total_train_data_size, data_file);
    val_user_data = std::make_unique<DataInformation>(total_val_data_size, data_file);
    std::cerr << "[B] Data loaded from: " << data_file << std::endl;
  } catch (std::invalid_argument &e) {
    std::cerr << "[B] Error loading data: " << e.what() << std::endl;
    return 1;
  }

  try {
    auto model = createModel(ModelType::NEURAL_NET,
                              {"batch_size=" + std::to_string(BATCH_SIZE)});
    std::cerr << "[B] Model created" << std::endl;

    model->addLayer(createLayer("input", {"name=input0",
                                           "input_shape=1:1:784"}));
    std::cerr << "[B] Input layer added" << std::endl;

    model->addLayer(createLayer("qat_fully_connected",
                                {"name=qat_fc1", "unit=128"}));
    std::cerr << "[B] QAT_FC1 layer added" << std::endl;

    model->addLayer(createLayer("activation",
                                {"name=relu1", "activation=relu"}));
    std::cerr << "[B] ReLU layer added" << std::endl;

    model->addLayer(createLayer("qat_fully_connected",
                                {"name=qat_fc2", "unit=10", "activation=softmax"}));
    std::cerr << "[B] QAT_FC2 layer added" << std::endl;

    auto optimizer = createOptimizer("adam", {"learning_rate=0.001"});
    model->setOptimizer(std::move(optimizer));
    std::cerr << "[B] Optimizer set" << std::endl;

    model->setProperty({"epochs=5", "loss=cross"});
    std::cerr << "[B] Properties set" << std::endl;

    auto dataset_train = std::shared_ptr<Dataset>(
      createDataset(DatasetType::GENERATOR, getSample, train_user_data.get()));
    auto dataset_val = std::shared_ptr<Dataset>(
      createDataset(DatasetType::GENERATOR, getSample, val_user_data.get()));
    
    model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);
    model->setDataset(DatasetModeType::MODE_VALID, dataset_val);
    std::cerr << "[B] Dataset set" << std::endl;

    model->compile();
    std::cerr << "[B] Compiled OK" << std::endl;

    model->initialize();
    std::cerr << "[B] Initialized OK" << std::endl;

    std::cerr << "[B] Calling train()..." << std::endl;
    model->train();
    std::cerr << "[B] Training COMPLETED successfully!" << std::endl;
    std::cerr << "[B] Training Loss: " << model->getTrainingLoss() << std::endl;
    std::cerr << "[B] Validation Loss: " << model->getValidationLoss() << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "[B] ERROR: " << e.what() << std::endl;
    return 1;
  }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[]) {
  std::cerr << "=== QAT MNIST Training ===" << std::endl;
  std::cerr << "Batch size: " << BATCH_SIZE << std::endl;
  std::cerr << "Train samples: " << total_train_data_size << std::endl;
  std::cerr << "Val samples: " << total_val_data_size << std::endl;

  if (argc < 2) {
    std::cerr << "\nUsage: " << argv[0] << " <mnist_data_file.dat>" << std::endl;
    std::cerr << "\nExample:" << std::endl;
    std::cerr << "  " << argv[0] << " /path/to/mnist_trainingSet.dat" << std::endl;
    return 1;
  }

  std::string data_file = argv[1];

  // Run Test A first (built-in layers only)
  int result_a = test_builtin_fc(data_file);
  std::cerr << "\nTest A result: " << (result_a == 0 ? "PASS" : "FAIL")
            << std::endl;

  // Only run Test B if Test A passes
  if (result_a == 0) {
    int result_b = test_custom_qat_fc(data_file);
    std::cerr << "\nTest B result: " << (result_b == 0 ? "PASS" : "FAIL")
              << std::endl;
  } else {
    std::cerr << "\nSkipping Test B (Test A failed - issue is in model setup, "
              << "not custom layer)" << std::endl;
  }

  return 0;
}
