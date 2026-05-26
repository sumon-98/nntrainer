// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_fc_layer.cpp
 * @brief  QAT Fully Connected Layer — Weight-Only Fake Quantization
 *
 * Implements weight-only QAT using the Straight-Through Estimator (STE).
 *
 * Forward pass:
 *   1. Fake-quantize weights to INT8 grid: w_fq = fakeQuantize(weight)
 *   2. Compute output = input * w_fq + bias
 *
 * Backward pass (STE):
 *   - calcDerivative: dL/dx = dL/dy * w_fq^T  (uses fake-quantized weights)
 *   - calcGradient:   dL/dW = x^T * dL/dy     (uses original FP32 inputs)
 *
 * The optimizer updates the FP32 master weights normally. The fake quantization
 * is only applied during the forward pass to simulate quantization effects.
 */

#include "qat_fc_layer.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <iostream>
#include <functional>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

QATFullyConnectedLayer::QATFullyConnectedLayer() :
  LayerImpl(),
  qat_fc_props(props::Unit()),
  q_min_weight(-128.0f),
  q_max_weight(127.0f),
  momentum(0.1f),
  initialized(false) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

QATFullyConnectedLayer::~QATFullyConnectedLayer() {
  if (initialized) {
    printQATStats();
  }
}

Tensor QATFullyConnectedLayer::fakeQuantize(
  Tensor &x, Tensor &running_min, Tensor &running_max,
  float q_min, float q_max, bool training) {

  if (training) {
    float current_min = x.minValue();
    float current_max = x.maxValue();

    if (std::isinf(running_min.getValue<float>(0))) {
      running_min.setValue(current_min);
      running_max.setValue(current_max);
    } else {
      float r_min = running_min.getValue<float>(0);
      float r_max = running_max.getValue<float>(0);
      running_min.setValue((1.0f - momentum) * r_min + momentum * current_min);
      running_max.setValue((1.0f - momentum) * r_max + momentum * current_max);
    }
  }

  float min_val = running_min.getValue<float>(0);
  float max_val = running_max.getValue<float>(0);

  float range = max_val - min_val;
  if (range < 1e-8f)
    range = 1e-8f;

  float scale = range / (q_max - q_min);
  float zero_point = q_min - std::round(min_val / scale);
  zero_point = std::max(q_min, std::min(q_max, zero_point)); // clamp

  Tensor x_fq = x.clone();
  std::function<float(float)> quantize_fn =
    [scale, zero_point, q_min, q_max](float v) -> float {
    float q = std::round((v / scale) + zero_point);
    q = std::max(q_min, std::min(q_max, q));
    return (q - zero_point) * scale;
  };
  x_fq.apply(quantize_fn, x_fq);

  return x_fq;
}

void QATFullyConnectedLayer::printQATStats() const {
  std::cerr << "--- " << getType() << " QAT Stats ---" << std::endl;

  float min_val = weight_running_min.getValue<float>(0);
  float max_val = weight_running_max.getValue<float>(0);
  float range = std::max(max_val - min_val, 1e-8f);
  float scale = range / (q_max_weight - q_min_weight);
  float zero_point = std::max(
    q_min_weight,
    std::min(q_max_weight, q_min_weight - std::round(min_val / scale)));

  std::cerr << "Weight running min: " << min_val
            << ", max: " << max_val << std::endl;
  std::cerr << "Weight Scale: " << scale
            << ", Weight ZP: " << zero_point << std::endl;
}

void QATFullyConnectedLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, qat_fc_props);
  LayerImpl::setProperty(remain_props);
}

void QATFullyConnectedLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(qat_fc_props, method, this);
}

void QATFullyConnectedLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  const auto &unit = std::get<props::Unit>(qat_fc_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "QATFullyConnectedLayer takes only one input";

  std::vector<TensorDim> output_dims(1);

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);

  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    TensorDim bias_dim(
      1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b0001 : 0b0100);

    weight_idx[FCParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
      "bias", true);
  }

  // Initialize weight quantization running stats
  weight_running_min = Tensor({1});
  weight_running_max = Tensor({1});
  weight_running_min.setValue(std::numeric_limits<float>::infinity());
  weight_running_max.setValue(-std::numeric_limits<float>::infinity());

  initialized = true;
}

void QATFullyConnectedLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  // Weight-only fake quantization: quantize weights, pass activations as-is
  w_fq = fakeQuantize(weight, weight_running_min, weight_running_max,
                      q_min_weight, q_max_weight, training);

  // output = input (FP32) * w_fq (fake-quantized)
  input_.dot(w_fq, hidden_, false, false);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

void QATFullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  // STE: Use fake-quantized weights for backprop through the layer
  ret_.dot_deriv_wrt_1(w_fq, derivative_, false, false);
}

void QATFullyConnectedLayer::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
  djdw.setZero();

  const Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);
    djdb.setZero();

    if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
      derivative_.sum({0, 1, 2}, djdb);
    } else {
      Tensor t = derivative_.sum({0, 1, 2});
      djdb.add_i(t);
    }
  }

  // Weight gradient uses original FP32 input (weight-only QAT: no
  // activation quantization, so input_ is used directly, not x_fq)
  input_.dot_deriv_wrt_2(
    djdw, derivative_, false, false,
    !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
}

} // namespace nntrainer
