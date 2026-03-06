// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Sumon Nath <sumon.nath@samsung.com>
 *
 * @file   eager_attention_layer.cpp
 * @date   14 January 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is EagerAttention Layer Class for Neural Network
 *
 */

#include <cmath>

#include <eager_attention_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

EagerAttentionLayer::EagerAttentionLayer() {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

EagerAttentionLayer::~EagerAttentionLayer() {}

void EagerAttentionLayer::finalizeCommon(InitLayerContext &context) {
}

void EagerAttentionLayer::finalize(InitLayerContext &context) {
  // Print input information
  // std::cout << "EagerAttentionLayer Input Information:" << std::endl;
  // std::cout << "  Number of inputs: " << context.getNumInputs() << std::endl;
  
  // For now, we'll set a default output dimension same as first input
  // In a real implementation, this would be based on the attention computation
  if (context.getNumInputs() > 0) {
    // std::vector<TensorDim> output_dims = {context.getInputDimensions()[0]};
    // context.setOutputDimensions(output_dims);
    // Get input dimensions
    auto const &all_dims = context.getInputDimensions();
    auto const &query_dim = all_dims[TensorParams::query];
    auto const &key_dim = all_dims[TensorParams::key];
    auto const &value_dim = all_dims[TensorParams::value];

    // Calculate context layer dimension (result of attention computation)
    auto context_layer_dim = query_dim;
    context_layer_dim.width(value_dim.width()); // Result of matmul

    // Set 3 outputs: attention output, keys, and values
    std::vector<TensorDim> output_dims = {context_layer_dim, key_dim, value_dim};
    context.setOutputDimensions(output_dims); 
  }
  
  // Pre-allocate temporary tensors
  // Get input dimensions
  auto const &all_dims = context.getInputDimensions();
  auto const &query_dim = all_dims[TensorParams::query];
  auto const &key_dim = all_dims[TensorParams::key];
  auto const &value_dim = all_dims[TensorParams::value];
  
  // 1. Temporary tensor for permuted key (key.transpose("0:2:1"))
  auto key_permuted_dim = key_dim.transpose("0:2:1");
  wt_idx[0] = context.requestTensor(key_permuted_dim, "key_permuted", 
                                    Initializer::NONE, false, 
                                    TensorLifespan::ITERATION_LIFESPAN);
  
  // 2. Temporary tensor for attention scores (query.dotBatched(key_permuted))
  auto attention_scores_dim = query_dim;
  attention_scores_dim.width(key_dim.height()); // Result of matmul
  wt_idx[1] = context.requestTensor(attention_scores_dim, "attention_scores", 
                                    Initializer::NONE, false, 
                                    TensorLifespan::ITERATION_LIFESPAN);
  
  // 3. Temporary tensor for attention weights (after softmax)
  wt_idx[2] = context.requestTensor(attention_scores_dim, "attention_weights", 
                                    Initializer::NONE, false, 
                                    TensorLifespan::ITERATION_LIFESPAN);
  
  // 4. Temporary tensor for context layer (attention_weights.dotBatched(value))
  auto context_layer_dim = query_dim;
  context_layer_dim.width(value_dim.width()); // Result of matmul
  wt_idx[3] = context.requestTensor(context_layer_dim, "context_layer", 
                                    Initializer::NONE, false, 
                                    TensorLifespan::ITERATION_LIFESPAN);
  
  // 5. Temporary tensor for attention mask
  // Create mask dimension to match attention_scores: (batch, channel, query_height, key_height)
  auto attention_mask_dim = TensorDim({1, 1, query_dim.height(), key_dim.height()});
  wt_idx[4] = context.requestTensor(attention_mask_dim, "attention_mask", 
                                    Initializer::NONE, false, 
                                    TensorLifespan::ITERATION_LIFESPAN);
  
  // Initialize softmax activation function
  auto data_type = context.getActivationDataType();
  if (data_type == ml::train::TensorDim::DataType::FP32) {
    sm.setActiFunc<float>(ActivationType::ACT_SOFTMAX);
  } else if (data_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    sm.setActiFunc<_FP16>(ActivationType::ACT_SOFTMAX);
#else
    throw std::runtime_error("enable-fp16 is not enabled");
#endif
  }
}

void EagerAttentionLayer::forwarding(RunLayerContext &context, bool training) {
  // std::cout << "------------------------------Attention called-------------------------------" << std::endl;
  // Get input tensors using proper indices from TensorParams
  Tensor &query = context.getInput(TensorParams::query);
  Tensor &key = context.getInput(TensorParams::key);
  Tensor &value = context.getInput(TensorParams::value);
  
  // std::cout << "------Query Key Value" << std::endl;
  // std::cout << query << key << value;
  // Get output tensor
  Tensor &output = context.getOutput(0);
  
  // Get pre-allocated temporary tensors
  Tensor &key_permuted = context.getTensor(wt_idx[0]);
  Tensor &attention_scores = context.getTensor(wt_idx[1]);
  Tensor &attention_weights = context.getTensor(wt_idx[2]);
  Tensor &context_layer = context.getTensor(wt_idx[3]);

  // 1. Permute key tensor with direction: 1,3,2 (0-indexed: 0,2,1)
  key.transpose("0:2:1", key_permuted);
  // std::cout << "------Key transpose" << std::endl;
  // std::cout << key_permuted;
  
  // 2. MatMul query and permuted key
  unsigned int batch = query.batch();
  unsigned int channel = query.channel();
  unsigned int height0 = query.height();
  unsigned int width0 = query.width();

  unsigned int height1 = key_permuted.height();
  unsigned int width1 = key_permuted.width();

  query.reshape(TensorDim({batch * channel, 1, height0, width0}));
  key_permuted.reshape(TensorDim({batch * channel, 1, height1, width1}));
  attention_scores.reshape(TensorDim({batch * channel, 1, height0, width1}));

  query.dotBatched(key_permuted, attention_scores);

  attention_scores.reshape(TensorDim({batch, channel, height0, width1}));
  // std::cout << "------Attention scores" << std::endl;
  // std::cout << attention_scores;
  
  // 3. Mul (scale by sqrt of head size)
  float scale = 1.0f / sqrt(static_cast<float>(key.getDim().width()));
  attention_scores.multiply_i(scale);
  // std::cout << "------Scaled Attention scores" << std::endl;
  // std::cout << attention_scores;

  // 4. Create and apply attention mask
  Tensor &attention_mask = context.getTensor(wt_idx[4]);
  
  // Initialize mask with large negative values
  attention_mask.setValue(-3.4028e+38f);
  
  // Apply causal mask: set diagonal and lower triangular part to 0
  // This is equivalent to torch.triu(causal_mask, diagonal=1) in PyTorch
  unsigned int mask_height = attention_scores.height();
  unsigned int mask_width = attention_scores.width();
  
  for (unsigned int i = 0; i < mask_height; ++i) {
    for (unsigned int j = 0; j <= i && j < mask_width; ++j) {
      attention_mask.setValue(0, 0, i, j, 0.0f);  // Set to 0 for non-masked positions
    }
  }

  // std::cout << "------Attention Mask" << std::endl;
  // std::cout << attention_mask;
  
  // Add mask to attention scores
  attention_scores.add_i(attention_mask);

  // std::cout << "------Masked Scaled Attention scores" << std::endl;
  // std::cout << attention_scores;
  
  // 5. Softmax activation
  sm.run_fn(attention_scores, attention_weights);
  // std::cout << "------Attention weights" << std::endl;
  // std::cout << attention_weights;

 
  // 6. MatMul softmax result with value
  batch = attention_weights.batch();
  channel = attention_weights.channel();
  height0 = attention_weights.height();
  width0 = attention_weights.width();

  height1 = value.height();
  width1 = value.width();

  attention_weights.reshape(TensorDim({batch * channel, 1, height0, width0}));
  value.reshape(TensorDim({batch * channel, 1, height1, width1}));
  context_layer.reshape(TensorDim({batch * channel, 1, height0, width1}));
  
  attention_weights.dotBatched(value, context_layer);
  context_layer.reshape(TensorDim({batch, channel, height0, width1}));
  // std::cout << "------Context layer" << std::endl;
  // std::cout << context_layer;

  
  // 7. Permute output with direction: 2,1,3 (0-indexed: 1,0,2)
  context_layer.transpose("1:0:2", output);
  
  // Copy keys and values to output tensors (for graph outputs)
  Tensor &output_keys = context.getOutput(1);
  Tensor &output_values = context.getOutput(2);

  // Copy the input keys and values to outputs
  output_keys.copy(key);
  output_values.copy(value);
  // std::cout << "------Attention Output" << std::endl;
  // std::cout << output;
  // std::cout << "--------------------------------------------------------------------------------" << std::endl;
}

void EagerAttentionLayer::incremental_forwarding(RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
}

void EagerAttentionLayer::calcDerivative(RunLayerContext &context) {
}

void EagerAttentionLayer::setProperty(const std::vector<std::string> &values) {
  // auto remain_props = loadProperties(values, attention_props);
  // if (!remain_props.empty()) {
  //   std::string msg = "[EagerAttentionLayer] Unknown Layer Properties count " +
  //                     std::to_string(values.size());
  //   throw exception::not_supported(msg);
  // }
}

void EagerAttentionLayer::setBatch(RunLayerContext &context, unsigned int batch) {
}

} /* namespace nntrainer */
