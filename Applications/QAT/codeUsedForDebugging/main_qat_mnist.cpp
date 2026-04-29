// SPDX-License-Identifier: Apache-2.0
/**
 * @file   main_qat_mnist.cpp
 * @brief  QAT Proof of Concept - DIAGNOSTIC VERSION
 *
 * Heavy debug printing to isolate where the segfault occurs.
 * Also provides a FALLBACK that uses only built-in layers.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include <model.h>
#include <optimizer.h>
#include <layer.h>
#include <dataset.h>
#include <app_context.h>
#include <engine.h>

#include "qat_fc_layer.h"

using namespace ml::train;

static const unsigned int NUM_SAMPLES = 100;  // Small for fast debugging
static const unsigned int BATCH_SIZE = 10;
static const unsigned int FEATURE_SIZE = 784;
static const unsigned int NUM_CLASSES = 10;

struct GeneratorContext {
  unsigned int count = 0;
};

/**
 * @brief Simple data generator using a context struct
 */
int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  auto ctx = static_cast<GeneratorContext *>(user_data);

  // Fill input with simple pattern
  for (unsigned int i = 0; i < FEATURE_SIZE; i++) {
    outVec[0][i] = static_cast<float>(i % 10) / 10.0f;
  }

  // One-hot label: always class 3
  for (unsigned int i = 0; i < NUM_CLASSES; i++) {
    outLabel[0][i] = 0.0f;
  }
  outLabel[0][3] = 1.0f;

  ctx->count++;
  if (ctx->count >= NUM_SAMPLES) {
    *last = true;
    ctx->count = 0;
  } else {
    *last = false;
  }

  return 0;
}

/**
 * Test A: Uses ONLY built-in fully_connected layers (no custom layer).
 * If this crashes, the problem is in the model setup/data generator.
 */
int test_builtin_fc() {
  std::cerr << "\n=== TEST A: Built-in FC layers only ===" << std::endl;

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

    auto optimizer = createOptimizer("sgd", {"learning_rate=0.01"});
    model->setOptimizer(std::move(optimizer));
    std::cerr << "[A] Optimizer set" << std::endl;

    model->setProperty({"epochs=1", "loss=cross"});
    std::cerr << "[A] Properties set" << std::endl;

    GeneratorContext ctx_a;
    auto dataset = std::shared_ptr<Dataset>(
      createDataset(DatasetType::GENERATOR, getSample, &ctx_a));
    model->setDataset(DatasetModeType::MODE_TRAIN, dataset);
    std::cerr << "[A] Dataset set" << std::endl;

    model->compile();
    std::cerr << "[A] Compiled OK" << std::endl;

    model->initialize();
    std::cerr << "[A] Initialized OK" << std::endl;

    std::cerr << "[A] Calling train()..." << std::endl;
    model->train();
    std::cerr << "[A] Training COMPLETED successfully!" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "[A] ERROR: " << e.what() << std::endl;
    return 1;
  }
}

/**
 * Test B: Uses our custom qat_fully_connected layer.
 * If Test A passes but this crashes, the problem is in the custom layer.
 */
int test_custom_qat_fc() {
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

    auto optimizer = createOptimizer("sgd", {"learning_rate=0.01"});
    model->setOptimizer(std::move(optimizer));
    std::cerr << "[B] Optimizer set" << std::endl;

    model->setProperty({"epochs=1", "loss=cross"});
    std::cerr << "[B] Properties set" << std::endl;

    GeneratorContext ctx_b;
    auto dataset = std::shared_ptr<Dataset>(
      createDataset(DatasetType::GENERATOR, getSample, &ctx_b));
    model->setDataset(DatasetModeType::MODE_TRAIN, dataset);
    std::cerr << "[B] Dataset set" << std::endl;

    model->compile();
    std::cerr << "[B] Compiled OK" << std::endl;

    model->initialize();
    std::cerr << "[B] Initialized OK" << std::endl;

    std::cerr << "[B] Calling train()..." << std::endl;
    model->train();
    std::cerr << "[B] Training COMPLETED successfully!" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "[B] ERROR: " << e.what() << std::endl;
    return 1;
  }
}

int main(int argc, char *argv[]) {
  std::cerr << "=== QAT Diagnostic Test ===" << std::endl;
  std::cerr << "Batch size: " << BATCH_SIZE
            << ", Samples: " << NUM_SAMPLES << std::endl;

  // Run Test A first (built-in layers only)
  int result_a = test_builtin_fc();
  std::cerr << "\nTest A result: " << (result_a == 0 ? "PASS" : "FAIL")
            << std::endl;

  // Only run Test B if Test A passes
  if (result_a == 0) {
    int result_b = test_custom_qat_fc();
    std::cerr << "\nTest B result: " << (result_b == 0 ? "PASS" : "FAIL")
              << std::endl;
  } else {
    std::cerr << "\nSkipping Test B (Test A failed - issue is in model setup, "
              << "not custom layer)" << std::endl;
  }

  return 0;
}
