// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   weight_layer.cpp
 * @date   2 August 2024
 * @brief  This is a layer that simply stores a weight tensor without any
 * operation.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>
#include <weight_layer.h>

#include <iostream>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;
WeightLayer::WeightLayer() : LayerImpl(), weight_props({}, {}, {}) {}

void WeightLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  const auto &weight_dim = std::get<props::TensorDimension>(weight_props).get();
  const auto &weight_dtype = std::get<props::TensorDataType>(weight_props);
  const auto &weight_name = std::get<props::WeightName>(weight_props);
  std::string data_type_str;
  switch (weight_dim.getDataType()) {
    case TensorDim::DataType::FP32:
      data_type_str = "FP32";
      break;
    case TensorDim::DataType::FP16:
      data_type_str = "FP16";
      break;
    default:
      data_type_str = "Unknown";
      break;
  }

  std::cout << "[Finalize] Weight Layer" << std::endl;
  std::cout << "Name: " << context.getName() << std::endl;
  std::cout << "Dim: " << weight_dim << std::endl;
  std::cout << "Data Type: " << data_type_str << std::endl;
  

  std::vector<TensorDim> output_dims(1);

  output_dims[SINGLE_INOUT_IDX] = weight_dim;
  output_dims[SINGLE_INOUT_IDX].setTensorType(
    {context.getFormat(), weight_dtype});

  context.setOutputDimensions(output_dims);

  weight_idx = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, weight_name, true);
}

void WeightLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(weight_props, method, this);
}

void WeightLayer::setProperty(const std::vector<std::string> &values) {
  std::cout << "[Set Property] Weight Layer props: " << std::endl;
  auto remain_props = loadProperties(values, weight_props);
  if (!remain_props.empty()) {
    std::string msg = "[WeightLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
  std::string data_type_str;
  switch (std::get<props::TensorDataType>(weight_props)) {
    case TensorDim::DataType::FP32:
      data_type_str = "FP32";
      break;
    case TensorDim::DataType::FP16:
      data_type_str = "FP16";
      break;
    default:
      data_type_str = "Unknown";
      break;
  }
  std::cout << "[After LoadProps] DType: " << data_type_str << std::endl;
  
  // Check if TensorDimension property is set before accessing it
  const auto &tensor_dim_prop = std::get<props::TensorDimension>(weight_props);
  if (tensor_dim_prop.empty()) {
    std::cout << "Dim: Not set yet" << std::endl;
  } else {
    std::cout << "Dim: " << tensor_dim_prop.get() << std::endl;
  }
  
  std::cout << "[Remaining Props]" << std::endl;
  LayerImpl::setProperty(remain_props);
  std::cout << "[Exit setProp]" << std::endl;
}

void WeightLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx);
  // std::cout << context.getName() << std::endl;
  // std::cout << weight << std::endl;
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  output.copy(weight);
}

void WeightLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for weight layer is not supported");
}

void WeightLayer::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx);
  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  djdw.copy(derivative_);
}

} /* namespace nntrainer */
