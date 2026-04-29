// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_fc_layer.cpp
 * @brief  QAT Fully Connected Layer - DIAGNOSTIC VERSION
 *
 * This is a carbon copy of FullyConnectedLayer logic.
 * NO fake quantization, NO temp tensors.
 * Purpose: prove that a custom layer with weights can compile & train.
 */

#include "qat_fc_layer.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <iostream>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

QATFullyConnectedLayer::QATFullyConnectedLayer() :
  LayerImpl(),
  qat_fc_props(props::Unit()),
  q_min_act(0.0f),
  q_max_act(255.0f),
  q_min_weight(-128.0f),
  q_max_weight(127.0f),
  momentum(0.1f),
  initialized(false) {  // Add this
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

QATFullyConnectedLayer::~QATFullyConnectedLayer() {
  if (initialized) {
    printQATStats();
  }
}

Tensor QATFullyConnectedLayer::fakeQuantize(Tensor &x, Tensor &running_min, Tensor &running_max, float q_min, float q_max, bool training) {
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
  if (range < 1e-8f) range = 1e-8f;
  
  float scale = range / (q_max - q_min);
  float zero_point = q_min - std::round(min_val / scale);
  zero_point = std::max(q_min, std::min(q_max, zero_point)); // clamp

  // Tensor x_fq = x.clone();
  // x_fq.apply([scale, zero_point, q_min, q_max](float v) -> float {
  //   float q = std::round((v / scale) + zero_point);
  //   q = std::max(q_min, std::min(q_max, q));
  //   return (q - zero_point) * scale;
  // }, x_fq);
  Tensor x_fq = x.clone();
  std::function<float(float)> quantize_fn = [scale, zero_point, q_min, q_max](float v) -> float {
    float q = std::round((v / scale) + zero_point);
    q = std::max(q_min, std::min(q_max, q));
    return (q - zero_point) * scale;
  };
  x_fq.apply(quantize_fn, x_fq);

  return x_fq;
}

void QATFullyConnectedLayer::printQATStats() const {
  // std::cerr << "--- " << getName() << " QAT Stats ---" << std::endl;
  std::cerr << "--- " << getType() << " QAT Stats ---" << std::endl;

  
  auto get_scale_zp = [](const Tensor& r_min, const Tensor& r_max, float q_min, float q_max) {
    float min_val = r_min.getValue<float>(0);
    float max_val = r_max.getValue<float>(0);
    float range = std::max(max_val - min_val, 1e-8f);
    float scale = range / (q_max - q_min);
    float zero_point = std::max(q_min, std::min(q_max, q_min - std::round(min_val / scale)));
    return std::make_pair(scale, zero_point);
  };

  auto act_stats = get_scale_zp(act_running_min, act_running_max, q_min_act, q_max_act);
  std::cerr << "Act Scale: " << act_stats.first << ", Act ZP: " << act_stats.second << std::endl;
  
  auto weight_stats = get_scale_zp(weight_running_min, weight_running_max, q_min_weight, q_max_weight);
  std::cerr << "Weight Scale: " << weight_stats.first << ", Weight ZP: " << weight_stats.second << std::endl;
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
  std::cerr << "[QAT_DIAG] finalize() ENTER for layer" << std::endl;

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

  // Initialize QAT running stats
  act_running_min = Tensor({1});
  act_running_max = Tensor({1});
  weight_running_min = Tensor({1});
  weight_running_max = Tensor({1});
  
  act_running_min.setValue(std::numeric_limits<float>::infinity());
  act_running_max.setValue(-std::numeric_limits<float>::infinity());
  weight_running_min.setValue(std::numeric_limits<float>::infinity());
  weight_running_max.setValue(-std::numeric_limits<float>::infinity());

  initialized = true;  // Add this at the end

  std::cerr << "[QAT_DIAG] finalize() EXIT" << std::endl;
}

void QATFullyConnectedLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  std::cerr << "[QAT_DIAG] forwarding() ENTER" << std::endl;

  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  // Fake Quantization Pass
  x_fq = fakeQuantize(input_, act_running_min, act_running_max, q_min_act, q_max_act, training);
  w_fq = fakeQuantize(weight, weight_running_min, weight_running_max, q_min_weight, q_max_weight, training);

  // Perform standard FC on the fake-quantized tensors
  x_fq.dot(w_fq, hidden_, false, false);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }

  std::cerr << "[QAT_DIAG] forwarding() EXIT" << std::endl;
}

void QATFullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  std::cerr << "[QAT_DIAG] calcDerivative() ENTER" << std::endl;

  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  const Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  // STE: Use fake-quantized weights for backprop
  ret_.dot_deriv_wrt_1(w_fq, derivative_, false, false);

  std::cerr << "[QAT_DIAG] calcDerivative() EXIT" << std::endl;
}

void QATFullyConnectedLayer::calcGradient(RunLayerContext &context) {
  std::cerr << "[QAT_DIAG] calcGradient() ENTER" << std::endl;

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

  // STE: Use fake-quantized inputs for weight gradient
  x_fq.dot_deriv_wrt_2(
    djdw, derivative_, false, false,
    !context.isGradientFirstAccess(weight_idx[FCParams::weight]));

  std::cerr << "[QAT_DIAG] calcGradient() EXIT" << std::endl;
}

} // namespace nntrainer
