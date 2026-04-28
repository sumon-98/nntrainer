#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include <model.h>
#include <optimizer.h>
#include <layer.h>
#include <dataset.h>
#include <app_context.h>

#include "qat_fc_layer.h"

using namespace ml::train;

const unsigned int num_samples = 1000;
const unsigned int batch_size = 32;
const unsigned int feature_size = 784;
const unsigned int num_classes = 10;

class RandomDataGenerator {
public:
  RandomDataGenerator() : count(0), rng(42) {}
  unsigned int count;
  std::mt19937 rng;
};

int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  auto data = reinterpret_cast<RandomDataGenerator *>(user_data);

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (unsigned int i = 0; i < feature_size; i++) {
    (*outVec)[i] = dist(data->rng);
  }

  float sum = 0.0f;
  for (unsigned int i = 0; i < feature_size; i++) {
    sum += (*outVec)[i];
  }
  int target_class = static_cast<int>(sum) % num_classes;

  for (unsigned int i = 0; i < num_classes; i++) {
    (*outLabel)[i] = (i == target_class) ? 1.0f : 0.0f;
  }

  data->count++;
  if (data->count < num_samples) {
    *last = false;
  } else {
    *last = true;
    data->count = 0;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  std::cout << "--- Phase 1: Quantization Aware Training (QAT) POC ---" << std::endl;

  // Register our custom QAT layer
  nntrainer::AppContext::Global().registerFactory(nntrainer::createLayer<nntrainer::QATFullyConnectedLayer>);

  RandomDataGenerator train_data;

  std::shared_ptr<Dataset> dataset_train;
  try {
    dataset_train = createDataset(DatasetType::GENERATOR, getSample, &train_data);
  } catch (const std::exception &e) {
    std::cerr << "Error creating dataset: " << e.what() << std::endl;
    return 1;
  }

  std::unique_ptr<Model> model = createModel(ModelType::NEURAL_NET);

  // Model Definition - use createLayer normally
  model->addLayer(createLayer("input", {"name=input0", "input_shape=1:1:784"}));
  model->addLayer(createLayer("qat_fully_connected", {"name=qat_fc1", "unit=128"}));
  model->addLayer(createLayer("activation", {"name=relu1", "activation=relu"}));
  model->addLayer(createLayer("qat_fully_connected", {"name=qat_fc2", "unit=10"}));
  // model->addLayer(createLayer("activation", {"name=softmax", "activation=softmax"}));
  
  auto optimizer = createOptimizer("adam");
  optimizer->setProperty({"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  model->setProperty({"epochs=1", "batch_size=" + std::to_string(batch_size), "loss=cross"});

  model->compile();
  model->initialize();
  model->setDataset(DatasetModeType::MODE_TRAIN, dataset_train);

  try {
    model->train();
    std::cout << "Training completed. Loss: " << model->getTrainingLoss() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error during train: " << e.what() << std::endl;
    return 1;
  }

  // Print QAT Statistics using the static registry
  std::cout << "\n--- QAT Fake Quantization Statistics ---" << std::endl;

  auto qat_fc1 = nntrainer::QATFullyConnectedLayer::getLayerByName("qat_fc1");
  auto qat_fc2 = nntrainer::QATFullyConnectedLayer::getLayerByName("qat_fc2");

  if (qat_fc1) {
    std::cout << "Layer: qat_fc1" << std::endl;
    std::cout << "  Activation Scale: " << qat_fc1->getActScale() << ", Zero Point: " << qat_fc1->getActZeroPoint() << std::endl;
    std::cout << "  Weight Scale: " << qat_fc1->getWeightScale() << ", Zero Point: " << qat_fc1->getWeightZeroPoint() << std::endl;
  }

  if (qat_fc2) {
    std::cout << "Layer: qat_fc2" << std::endl;
    std::cout << "  Activation Scale: " << qat_fc2->getActScale() << ", Zero Point: " << qat_fc2->getActZeroPoint() << std::endl;
    std::cout << "  Weight Scale: " << qat_fc2->getWeightScale() << ", Zero Point: " << qat_fc2->getWeightZeroPoint() << std::endl;
  }

  return 0;
}
