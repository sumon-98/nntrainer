// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_fc_layer.cpp
 * @brief  QAT Fully Connected Layer — supports both full QAT and LoRA QAT
 *
 * Two modes using the same layer code:
 *
 * MODE 1 — Full Model QAT (lora_rank not set):
 *   Forward:  w_fq = fakeQuantize(W)
 *             output = input * w_fq + bias
 *   Backward: STE on base weight gradients
 *   Use case: train_qwen3_qat.cpp (all weights trainable with QAT)
 *
 * MODE 2 — LoRA QAT (lora_rank > 0):
 *   Forward:  a_fq = fakeQuantize(loraA)
 *             b_fq = fakeQuantize(loraB)
 *             output = input * W_frozen
 *                    + input * a_fq * b_fq * scaling
 *                    + bias
 *   Backward: STE on LoRA weight gradients only (base W is frozen)
 *   Use case: train_qwen3_qat_lora.cpp (base frozen, LoRA adapters
 *             trained with QAT for INT8 deployment of adapters)
 *
 * Key difference from previous version:
 *   OLD: QAT was on the BASE weight, LoRA was added in FP32 on top
 *   NEW: Base weight is vanilla FP32 (frozen), QAT is on the LORA weights
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
  lora_scaling(1.0f),
  qat_fc_props(props::Unit(), props::LoraRank(), props::LoraAlpha()),
  q_min(-128.0f),
  q_max(127.0f),
  momentum(0.01f),
  initialized(false) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  lora_idx.fill(std::numeric_limits<unsigned>::max());
}

QATFullyConnectedLayer::~QATFullyConnectedLayer() {
  if (initialized) {
    printQATStats();
  }
}

Tensor QATFullyConnectedLayer::fakeQuantize(
  Tensor &x, Tensor &running_min, Tensor &running_max,
  float q_min_val, float q_max_val, bool training) {

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

  float scale = range / (q_max_val - q_min_val);
  float zero_point = q_min_val - std::round(min_val / scale);
  zero_point = std::max(q_min_val, std::min(q_max_val, zero_point));

  Tensor x_fq = x.clone();
  std::function<float(float)> quantize_fn =
    [scale, zero_point, q_min_val, q_max_val](float v) -> float {
    float q = std::round((v / scale) + zero_point);
    q = std::max(q_min_val, std::min(q_max_val, q));
    return (q - zero_point) * scale;
  };
  x_fq.apply(quantize_fn, x_fq);

  return x_fq;
}

void QATFullyConnectedLayer::printQATStats() const {
  const auto &lora_rank = std::get<props::LoraRank>(qat_fc_props);

  if (lora_rank.empty()) {
    // Mode 1: print base weight quantization stats
    std::cerr << "--- " << getType() << " QAT Stats (Base Weight) ---"
              << std::endl;

    float min_val = weight_running_min.getValue<float>(0);
    float max_val = weight_running_max.getValue<float>(0);
    float range = std::max(max_val - min_val, 1e-8f);
    float scale = range / (q_max - q_min);
    float zero_point = std::max(
      q_min, std::min(q_max, q_min - std::round(min_val / scale)));

    std::cerr << "Weight running min: " << min_val
              << ", max: " << max_val << std::endl;
    std::cerr << "Weight Scale: " << scale
              << ", Weight ZP: " << zero_point << std::endl;
  } else {
    // Mode 2: print LoRA weight quantization stats
    std::cerr << "--- " << getType() << " QAT Stats (LoRA Weights) ---"
              << std::endl;
    std::cerr << "LoRA rank: " << lora_rank.get()
              << ", scaling: " << lora_scaling << std::endl;

    float a_min = lora_a_running_min.getValue<float>(0);
    float a_max = lora_a_running_max.getValue<float>(0);
    float a_range = std::max(a_max - a_min, 1e-8f);
    float a_scale = a_range / (q_max - q_min);
    std::cerr << "LoraA running min: " << a_min << ", max: " << a_max
              << ", scale: " << a_scale << std::endl;

    float b_min = lora_b_running_min.getValue<float>(0);
    float b_max = lora_b_running_max.getValue<float>(0);
    float b_range = std::max(b_max - b_min, 1e-8f);
    float b_scale = b_range / (q_max - q_min);
    std::cerr << "LoraB running min: " << b_min << ", max: " << b_max
              << ", scale: " << b_scale << std::endl;
  }
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

void QATFullyConnectedLayer::setBatch(nntrainer::RunLayerContext &context,
                                       unsigned int batch) {
  if (!std::get<props::LoraRank>(qat_fc_props).empty()) {
    context.updateTensor(lora_idx[LORAParams::loraTmp], batch);
    context.updateTensor(lora_idx[LORAParams::loraOut], batch);
  }
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

  // LoRA setup
  const auto &lora_rank = (std::get<props::LoraRank>(qat_fc_props).empty())
                            ? 0
                            : std::get<props::LoraRank>(qat_fc_props).get();
  lora_scaling =
    (lora_rank && !std::get<props::LoraAlpha>(qat_fc_props).empty())
      ? (float)std::get<props::LoraAlpha>(qat_fc_props) / lora_rank
      : 1;

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

  // Force Q6_K for base weights in LoRA mode for this POC
  ml::train::TensorDim::DataType base_weight_type = context.getWeightDataType();
  if (lora_rank > 0) {
    base_weight_type = ml::train::TensorDim::DataType::Q6_K;
  }

  // Weight dimension
  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), base_weight_type),
    is_nchw ? 0b0011 : 0b0101);

  // When LoRA is active, base weight is NOT trainable (lora_rank == 0 → true)
  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", (lora_rank == 0));

  if (disable_bias.empty() || disable_bias.get() == false) {
    TensorDim bias_dim(
      1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b0001 : 0b0100);

    weight_idx[FCParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
      "bias", (lora_rank == 0));
  }

  // Create LoRA weights if lora_rank is specified
  if (lora_rank) {
    // ─── Mixed-precision support ───────────────────────────────────────
    // LoRA weights MUST always be FP32, even when the base model weight type
    // is a quantized format (Q6_K, Q4_0, etc.).
    //
    // The forward path does:
    //   base:  input(FP32) . weight(Q6_K)  → dotQnK kernel handles this
    //   lora:  input(FP32) . loraA(FP32) . loraB(FP32) * scaling
    //
    // If we inherited context.getWeightDataType() here, LoRA A and B would
    // be created as Q6_K tensors — which cannot be trained, cannot be
    // fake-quantized, and would crash the backward pass.
    //
    // This is backward-compatible: when model_tensor_type=FP32-FP32,
    // context.getWeightDataType() is already FP32, so nothing changes.
    // ───────────────────────────────────────────────────────────────────
    TensorDim loraA_dim(
      1, is_nchw ? 1 : lora_rank, is_nchw ? in_dim.width() : 1,
      is_nchw ? lora_rank : in_dim.channel(),
      TensorDim::TensorType(context.getFormat(), ml::train::TensorDim::DataType::FP32),
      is_nchw ? 0b0011 : 0b0101);

    TensorDim loraB_dim(
      1, is_nchw ? 1 : unit, is_nchw ? lora_rank : 1,
      is_nchw ? unit : lora_rank,
      TensorDim::TensorType(context.getFormat(), ml::train::TensorDim::DataType::FP32),
      is_nchw ? 0b0011 : 0b0101);

    TensorDim loraTmp_dim(
      in_dim.batch(), is_nchw ? 1 : lora_rank, is_nchw ? in_dim.height() : 1,
      is_nchw ? lora_rank : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    TensorDim loraOut_dim(
      in_dim.batch(), is_nchw ? 1 : unit, is_nchw ? in_dim.height() : 1,
      is_nchw ? unit : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    lora_idx[LORAParams::loraA] = context.requestWeight(
      loraA_dim, Initializer::ZEROS, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraA", true);

    lora_idx[LORAParams::loraB] = context.requestWeight(
      loraB_dim, Initializer::LECUN_NORMAL, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraB", true);

    lora_idx[LORAParams::loraTmp] =
      context.requestTensor(loraTmp_dim, "hidden_tmp_lora", Initializer::NONE,
                            true, TensorLifespan::FORWARD_GRAD_LIFESPAN);

    lora_idx[LORAParams::loraOut] =
      context.requestTensor(loraOut_dim, "hidden_lora", Initializer::NONE, true,
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);

    // Initialize running stats for LoRA weight quantization
    lora_a_running_min = Tensor({1});
    lora_a_running_max = Tensor({1});
    lora_a_running_min.setValue(std::numeric_limits<float>::infinity());
    lora_a_running_max.setValue(-std::numeric_limits<float>::infinity());

    lora_b_running_min = Tensor({1});
    lora_b_running_max = Tensor({1});
    lora_b_running_min.setValue(std::numeric_limits<float>::infinity());
    lora_b_running_max.setValue(-std::numeric_limits<float>::infinity());
  }

  // Initialize base weight quantization running stats (used in Mode 1 only)
  weight_running_min = Tensor({1});
  weight_running_max = Tensor({1});
  weight_running_min.setValue(std::numeric_limits<float>::infinity());
  weight_running_max.setValue(-std::numeric_limits<float>::infinity());

  initialized = true;
}

// =============================================================================
// FORWARDING
// =============================================================================
void QATFullyConnectedLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (std::get<props::LoraRank>(qat_fc_props).empty()) {
    // ===== MODE 1: Full QAT — fake-quantize base weight =====
    // w_fq = fakeQuantize(W); output = input * w_fq
    w_fq = fakeQuantize(weight, weight_running_min, weight_running_max,
                        q_min, q_max, training);
    input_.dot(w_fq, hidden_, false, false);
  } else {
    // ===== MODE 2: LoRA QAT — base weight vanilla, QAT on LoRA =====
    // Base path: output = input * W_frozen (no fake-quant on W)
    input_.dot(weight, hidden_, false, false);

    // LoRA path: fake-quantize A and B, then compute LoRA contribution
    Tensor &loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    Tensor &hidden_tmp_lora = context.getTensor(lora_idx[LORAParams::loraTmp]);
    Tensor &hidden_out_lora = context.getTensor(lora_idx[LORAParams::loraOut]);

    // Fake-quantize LoRA weights (simulate INT8 deployment of adapters)
    a_fq = fakeQuantize(loraA, lora_a_running_min, lora_a_running_max,
                        q_min, q_max, training);
    b_fq = fakeQuantize(loraB, lora_b_running_min, lora_b_running_max,
                        q_min, q_max, training);

    // LoRA contribution: input * fakeQuant(A) * fakeQuant(B) * scaling
    input_.dot(a_fq, hidden_tmp_lora, false, false);
    hidden_tmp_lora.dot(b_fq, hidden_out_lora, false, false);
    hidden_out_lora.multiply_i(lora_scaling);
    hidden_.add_i(hidden_out_lora);
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

// =============================================================================
// CALC DERIVATIVE (dL/dx — gradient flowing to previous layer)
// =============================================================================
void QATFullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  if (std::get<props::LoraRank>(qat_fc_props).empty()) {
    // MODE 1: dL/dx = dL/dy * w_fq^T (STE on base weight)
    ret_.dot_deriv_wrt_1(w_fq, derivative_, false, false);
  } else {
    // MODE 2: effective weight = W_frozen + fakeQuant(A) * fakeQuant(B) * scaling
    // dL/dx = dL/dy * [W + a_fq * b_fq * scaling]^T
    Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
    
    // For Q6_K weights, we need to dequantize first before adding LoRA contribution
    Tensor weight_fp32;
    if (weight.getDataType() == ml::train::TensorDim::DataType::Q6_K) {
      // Dequantize Q6_K to FP32 for the addition
      auto dequantizer = Quantization::createQuantizer(QScheme::Q6_K);
      weight_fp32 = dequantizer->dequantize(weight, ml::train::TensorDim::DataType::FP32);
    } else {
      weight_fp32 = weight;
    }
    
    // Compute LoRA contribution
    Tensor lora_contrib = a_fq.dot(b_fq).multiply(lora_scaling);
    
    // Effective weight = dequantized_base + lora_contribution
    Tensor effective_weight = weight_fp32.add(lora_contrib);
    
    ret_.dot_deriv_wrt_1(effective_weight, derivative_, false, false);
  }
}

// =============================================================================
// CALC GRADIENT (dL/dW or dL/dA, dL/dB)
// =============================================================================
void QATFullyConnectedLayer::calcGradient(RunLayerContext &context) {
  const Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (std::get<props::LoraRank>(qat_fc_props).empty()) {
    // ===== MODE 1: Full QAT — gradients for base weight via STE =====
    Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
    djdw.setZero();

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

    // STE: gradient flows through fakeQuantize as identity
    input_.dot_deriv_wrt_2(
      djdw, derivative_, false, false,
      !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
  } else {
    // ===== MODE 2: LoRA QAT — gradients for LoRA params only =====
    // Base weight is frozen → no djdw computed
    //
    // Forward was: loraTmp = input * a_fq;  loraOut = loraTmp * b_fq * scaling
    // STE: treat fakeQuant(A), fakeQuant(B) as identity for gradient purposes
    //
    // Chain rule:
    //   dL/dB = loraTmp^T * (dL/dy * scaling)     [STE on B]
    //   dL/d(loraTmp) = (dL/dy * scaling) * b_fq^T [use b_fq from forward]
    //   dL/dA = input^T * dL/d(loraTmp)            [STE on A]

    Tensor &djdla = context.getWeightGrad(lora_idx[LORAParams::loraA]);
    Tensor &djdlb = context.getWeightGrad(lora_idx[LORAParams::loraB]);
    Tensor &djdtmp = context.getTensorGrad(lora_idx[LORAParams::loraTmp]);

    Tensor &loraTmp = context.getTensor(lora_idx[LORAParams::loraTmp]);
    const auto &lora_derivative_ = derivative_.multiply(lora_scaling);

    // dL/dB: loraTmp already contains (input * a_fq) from forward pass
    loraTmp.dot_deriv_wrt_2(
      djdlb, lora_derivative_, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraB]));

    // dL/d(loraTmp): use b_fq (the fake-quantized B from forward)
    djdtmp.dot_deriv_wrt_1(
      b_fq, lora_derivative_, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp]));

    // dL/dA: STE — gradient flows through fakeQuant(A) as identity
    input_.dot_deriv_wrt_2(
      djdla, djdtmp, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraA]));
  }
}

} // namespace nntrainer
