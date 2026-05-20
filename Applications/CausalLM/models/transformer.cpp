// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Transformer's basic actions
 */

#include <fstream>
#include <iostream>

#include <app_context.h>
#include <engine.h>
#include <model.h>

#include <llm_util.hpp>
#include <tokenizers_cpp.h>
#include <transformer.h>
#include <layer.h>

#include <embedding_layer.h>
#include <mha_core.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

namespace causallm {

std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

ModelType strToModelType(std::string model_type) {

  std::string model_type_lower = model_type;
  std::transform(model_type_lower.begin(), model_type_lower.end(),
                 model_type_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  static const std::unordered_map<std::string, ModelType> model_type_map = {
    {"model", ModelType::MODEL},
    {"causallm", ModelType::CAUSALLM},
    {"embedding", ModelType::EMBEDDING}};

  if (model_type_map.find(model_type_lower) == model_type_map.end()) {
    return ModelType::UNKNOWN;
  }

  return model_type_map.at(model_type_lower);
}

Transformer::Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
                         ModelType model_type) {

  std::string config_model_type_str = "Model";
  if (nntr_cfg.contains("model_type")) {
    config_model_type_str = nntr_cfg["model_type"].get<std::string>();
  }

  ModelType config_model_type = strToModelType(config_model_type_str);

  if (model_type != config_model_type) {
    throw std::runtime_error("model_type mismatch. Class Type: " +
                             std::to_string(static_cast<int>(model_type)) +
                             ", Config Type: " + config_model_type_str);
  }

  // Initialize the model with the provided configurations
  // This is where you would set up the model layers, parameters, etc.
  setupParameters(cfg, generation_cfg, nntr_cfg);

  // prep tokenizer
  tokenizer = tokenizers::Tokenizer::FromBlobJSON(
    LoadBytesFromFile(nntr_cfg["tokenizer_file"]));
};

void Transformer::setupParameters(json &cfg, json &generation_cfg,
                                  json &nntr_cfg) {

  /** Initialize nntr prameters */
  BATCH_SIZE = nntr_cfg["batch_size"].get<unsigned int>();
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"].get<std::string>();
  INIT_SEQ_LEN = nntr_cfg["init_seq_len"];
  MAX_SEQ_LEN = nntr_cfg["max_seq_len"];
  NUM_TO_GENERATE = nntr_cfg["num_to_generate"];
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"];
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  EMBEDDING_DTYPE = nntr_cfg["embedding_dtype"];
  FC_LAYER_DTYPE = nntr_cfg["fc_layer_dtype"];

  if (cfg.contains("is_causal")) {
    IS_CAUSAL = cfg["is_causal"].get<bool>();
  } else if (cfg.contains("use_bidirectional_attention")) {
    IS_CAUSAL = !cfg["use_bidirectional_attention"].get<bool>();
  }

  NUM_VOCAB = cfg["vocab_size"];
  DIM = cfg["hidden_size"];
  INTERMEDIATE_SIZE = cfg["intermediate_size"];
  NUM_LAYERS = cfg["num_hidden_layers"];
  NUM_HEADS = cfg["num_attention_heads"];
  HEAD_DIM = cfg.contains("head_dim")
               ? cfg["head_dim"].get<int>()
               : DIM / NUM_HEADS; // default value is hidden_size / num_heads
  NUM_KEY_VALUE_HEADS = cfg.contains("num_key_value_heads")
                          ? cfg["num_key_value_heads"].get<int>()
                          : NUM_HEADS;
  SLIDING_WINDOW =
    cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
      ? cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;
  SLIDING_WINDOW_PATTERN = cfg.contains("sliding_window_pattern")
                             ? cfg["sliding_window_pattern"].get<unsigned int>()
                             : 1;
  MAX_POSITION_EMBEDDINGS = cfg["max_position_embeddings"].get<unsigned int>();
  ROPE_THETA = cfg["rope_theta"].get<unsigned int>();
  TIE_WORD_EMBEDDINGS = cfg["tie_word_embeddings"].get<bool>();
  NORM_EPS = cfg["rms_norm_eps"];
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  /** LoRA parameters */
  if (nntr_cfg.contains("lora_rank")) {
    LORA_RANK = nntr_cfg["lora_rank"].get<unsigned int>();
  }
  if (nntr_cfg.contains("lora_alpha")) {
    LORA_ALPHA = nntr_cfg["lora_alpha"].get<unsigned int>();
  }
  if (nntr_cfg.contains("lora_target")) {
    for (const auto &target : nntr_cfg["lora_target"]) {
      LORA_TARGETS.push_back(target.get<std::string>());
    }
  }

  return;
};

void Transformer::initialize() {

  // RegisterCustomLayers
  registerCustomLayers();

  // construct causalLM model
  constructModel();

  // setup model property
  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }

  model->setProperty(model_props);

  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model initialization failed.");
  }

  is_initialized = true;

#ifdef DEBUG
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
#endif
}

void Transformer::initializeForTraining(float lr, unsigned int epochs) {
  registerCustomLayers();
  constructModel();

  // Append cross_softmax loss explicitly for training the network
  try {
    model->addLayer(ml::train::createLayer("cross_softmax", {"name=loss"}));
  } catch (const std::exception &e) {
    std::cerr << "Note: " << e.what() << std::endl;
  }

  // setup model property
  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE),
    withKey("epochs", epochs),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};

  model->setProperty(model_props);

  // set optimizer (Adam for LoRA fine-tuning)
  auto optimizer =
    ml::train::createOptimizer("adam", {"learning_rate=" + std::to_string(lr)});
  if (model->setOptimizer(std::move(optimizer))) {
    throw std::invalid_argument("Failed to set optimizer.");
  }

  if (model->compile(ml::train::ExecutionMode::TRAIN)) {
    throw std::invalid_argument("Model compilation for training failed.");
  }

  if (model->initialize(ml::train::ExecutionMode::TRAIN)) {
    throw std::invalid_argument("Model initialization for training failed.");
  }

  is_initialized = true;
}

void Transformer::constructModel() {

  // layers used in the model
  std::vector<LayerHandle> layers;

  // create model
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  // create input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // create embedding layer
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  std::vector<std::string> embed_params = {
    "name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
    "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
    "scale=" + std::to_string(EMBEDDING_SCALE)};
  embed_params.push_back(withKey("trainable", "false"));
  layers.push_back(createLayer(embedding_type, embed_params));

  // create transformer layers
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoderBlock(0, "embedding0");
    else
      transformer = createTransformerDecoderBlock(
        i, "layer" + std::to_string(i - 1) + "_decoder_output");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  // create rms_norm
  std::vector<std::string> out_norm_params = {
    withKey("name", "output_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("input_layers",
            "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
    withKey("packed", "false")};
  if (LORA_RANK > 0) {
    out_norm_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("rms_norm", out_norm_params));

  // create lm_head
  const std::string lm_head_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "fully_connected";
  std::vector<std::string> lm_head_params = {
    "name=lm_head", "unit=" + std::to_string(NUM_VOCAB),
    "weight_dtype=" + EMBEDDING_DTYPE,
    "bias_initializer=zeros", "input_layers=output_norm"};
  if (LORA_RANK > 0) {
    lm_head_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer(lm_head_type, lm_head_params));

  // add created layers into the model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
};

void Transformer::load_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before load_weight().");
  }

  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load model weights: " +
                             std::string(e.what()));
  }
};

void Transformer::load_weight_lora(const std::string &weight_path, const std::string &lora_path) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before load_weight().");
  }

  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN, lora_path);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load model weights: " +
                             std::string(e.what()));
  }
};


void Transformer::save_weight(const std::string &weight_path, ml::train::ModelFormat format) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, format);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights: " +
                             std::string(e.what()));
  }
};

void Transformer::run(const WSTR prompt, bool do_sample,
                      const WSTR system_prompt, const WSTR tail_prompt,
                      bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before run().");
  }
  ///@note This part should be filled in.
  /// The run action can be defined by the precedent classes.
}

std::vector<LayerHandle>
Transformer::createTransformerDecoderBlock(const int layer_id,
                                           std::string input_name) {

  std::vector<LayerHandle> layers;

  std::vector<std::string> attn_norm_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
    withKey("input_layers", input_name),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("packed", "false")};
  if (LORA_RANK > 0) {
    attn_norm_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("rms_norm", attn_norm_params));

  auto att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm");

  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  std::vector<std::string> dec_add_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
    withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")};
  if (LORA_RANK > 0) {
    dec_add_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("addition", dec_add_params));

  std::vector<std::string> ffn_norm_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_decoder_add"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("packed", "false")};
  if (LORA_RANK > 0) {
    ffn_norm_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("rms_norm", ffn_norm_params));

  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  std::vector<std::string> dec_out_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
    withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_down")};
  if (LORA_RANK > 0) {
    dec_out_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("addition", dec_out_params));

  return layers;
}

bool Transformer::isLoRATarget(const std::string &layer_suffix) const {
  if (LORA_RANK == 0)
    return false;
  for (const auto &target : LORA_TARGETS) {
    if (target == layer_suffix)
      return true;
  }
  return false;
}

std::vector<LayerHandle>
Transformer::createAttention(const int layer_id, int seq_len, int n_heads,
                             int head_dim, std::string query_name,
                             std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  if (isLoRATarget("wq")) {
    q_params.push_back(withKey("lora_rank", LORA_RANK));
    q_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  } else if (LORA_RANK > 0) {
    q_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("fully_connected", q_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  if (isLoRATarget("wk")) {
    k_params.push_back(withKey("lora_rank", LORA_RANK));
    k_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  } else if (LORA_RANK > 0) {
    k_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("fully_connected", k_params));

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  if (isLoRATarget("wv")) {
    v_params.push_back(withKey("lora_rank", LORA_RANK));
    v_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  } else if (LORA_RANK > 0) {
    v_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("fully_connected", v_params));

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", (layer_id + 1) % SLIDING_WINDOW_PATTERN
                                ? SLIDING_WINDOW
                                : UINT_MAX),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false"),
    withKey("input_layers", {Q, K, V})};
  if (LORA_RANK > 0) {
    a_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  if (isLoRATarget("wo")) {
    o_params.push_back(withKey("lora_rank", LORA_RANK));
    o_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  } else if (LORA_RANK > 0) {
    o_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle> Transformer::createMlp(const int layer_id, int dim,
                                                int hidden_dim,
                                                std::string input_name) {

  std::vector<LayerHandle> layers;

  std::vector<std::string> up_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("unit", hidden_dim), withKey("disable_bias", "true"),
    withKey("input_layers", input_name),
    withKey("weight_initializer", "ones")};
  if (isLoRATarget("ffn_up")) {
    up_params.push_back(withKey("lora_rank", LORA_RANK));
    up_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  } else if (LORA_RANK > 0) {
    up_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("fully_connected", up_params));

  std::vector<std::string> gate_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
    withKey("unit", hidden_dim), withKey("disable_bias", "true"),
    withKey("input_layers", input_name),
    withKey("weight_initializer", "ones")};
  if (isLoRATarget("ffn_gate")) {
    gate_params.push_back(withKey("lora_rank", LORA_RANK));
    gate_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  } else if (LORA_RANK > 0) {
    gate_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("fully_connected", gate_params));

  std::vector<std::string> swiglu_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
    withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_gate," +
                               "layer" + std::to_string(layer_id) +
                               "_ffn_up")};
  if (LORA_RANK > 0) {
    swiglu_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("swiglu", swiglu_params));

  std::vector<std::string> down_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
    withKey("unit", dim), withKey("disable_bias", "true"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
    withKey("weight_initializer", "ones")};
  if (isLoRATarget("ffn_down")) {
    down_params.push_back(withKey("lora_rank", LORA_RANK));
    down_params.push_back(withKey("lora_alpha", LORA_ALPHA));
  } else if (LORA_RANK > 0) {
    down_params.push_back(withKey("trainable", "false"));
  }
  layers.push_back(createLayer("fully_connected", down_params));

  return layers;
}

void Transformer::registerCustomLayers() {
  ///
  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::SwiGLULayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::RMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MHACoreLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::TieWordEmbedding>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingLayer>);

  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

// LoRA Debugging
void Transformer::exportWeightsToFile(const std::string& filename) const {
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }
  
  outfile << std::fixed << std::setprecision(6);
  outfile << "=== Detailed Model Weights Export ===" << std::endl;
  outfile << "Export Time: " << std::time(nullptr) << std::endl;
  outfile << "=====================================" << std::endl << std::endl;
  
  if (!model) {
    outfile << "Error: Model is not initialized" << std::endl;
    outfile.close();
    return;
  }
  
  // Use the forEachLayer method to iterate through all layers
  model->forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &rc, void *user_data) {
      std::ofstream &outfile = *static_cast<std::ofstream*>(user_data);
      
      outfile << "Layer: " << layer.getName() << " (Type: " << layer.getType() << ")\n";
      
      try {
        // Get weights using the layer interface
        std::vector<float*> weights;
        std::vector<ml::train::TensorDim> weight_dims;
        layer.getWeights(weights, weight_dims);
        
        for (size_t i = 0; i < weights.size(); ++i) {
          if (weights[i] && weight_dims[i].getFeatureLen() > 0) {
            outfile << "  Weight " << i << " (Name: " << layer.getWeightName(i) 
                    << ", Dim: " << weight_dims[i].getFeatureLen() << "): ";
            
            // Print first 10 and last 10 values for large weights
            size_t len = weight_dims[i].getFeatureLen();
            size_t print_count = std::min(len, (size_t)20);
            
            for (size_t j = 0; j < std::min(print_count/2, len); ++j) {
              outfile << weights[i][j] << " ";
            }
            
            if (len > print_count) {
              outfile << "... ";
              for (size_t j = len - print_count/2; j < len; ++j) {
                outfile << weights[i][j] << " ";
              }
            } else {
              for (size_t j = print_count/2; j < len; ++j) {
                outfile << weights[i][j] << " ";
              }
            }
            outfile << "\n";
          }
        }
      } catch (const std::exception& e) {
        outfile << "  Error accessing weights: " << e.what() << "\n";
      }
      outfile << "\n";
    },
    &outfile
  );
  
  outfile.close();
  std::cout << "Detailed model weights exported to: " << filename << std::endl;
}

} // namespace causallm
