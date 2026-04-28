#include <qat_fc_layer.h>
#include <common_properties.h>  // Add this
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <cmath>
#include <algorithm>

namespace nntrainer {

void QATFullyConnectedLayer::setProperty(const std::vector<std::string> &values) {
  // DO NOT reset unit and disable_bias here - they may have been set already
  // unit = 0;
  // disable_bias = false; 
  for (auto &v : values) {
    if (v.find("unit=") != std::string::npos) {
      unit = std::stoi(v.substr(5));
    } else if (v.find("disable_bias=") != std::string::npos) {
      disable_bias = (v.find("true") != std::string::npos);
    } else if (v.find("name=") != std::string::npos) {
      std::string name = v.substr(5);
      registerLayer(name);  // Register this layer in the static map
    }
  }
}

void QATFullyConnectedLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument) << "QATFullyConnectedLayer takes exactly 1 input";
  
  if (unit == 0) {
    NNTR_THROW_IF(true, std::invalid_argument) << "unit must be > 0";
  }

  // Set effective and dynamic dimension flags (required for proper tensor handling)
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);

  std::vector<TensorDim> output_dims(1);
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);
  
  // Set the tensor type for output
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  // Weight dimension with proper tensor type and eff_dim_flag
  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
      weight_dim, Initializer::XAVIER_UNIFORM, WeightRegularizer::NONE, 1.0f, 0.0f, "weight", true);

  if (!disable_bias) {
    // Bias dimension with proper tensor type
    TensorDim bias_dim(
      1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
      TensorDim::TensorType(context.getFormat(), context.getActivationDataType()),
      is_nchw ? 0b0001 : 0b0100);
    
    weight_idx[FCParams::bias] = context.requestWeight(
        bias_dim, Initializer::ZEROS, WeightRegularizer::NONE, 1.0f, 0.0f, "bias", true);
  }
}


Tensor QATFullyConnectedLayer::fakeQuantize(const Tensor &x, float &min_val, float &max_val, float q_min, float q_max, bool update_stats, float &out_scale, float &out_zp) {
  if (update_stats) {
    float current_min = x.minValue();
    float current_max = x.maxValue();
    
    if (!initialized_stats) {
      min_val = current_min;
      max_val = current_max;
      initialized_stats = true;
    } else {
      min_val = (1.0f - momentum) * min_val + momentum * current_min;
      max_val = (1.0f - momentum) * max_val + momentum * current_max;
    }
  }

  float range = std::max(max_val - min_val, 1e-8f);
  out_scale = range / (q_max - q_min);
  out_zp = std::max(q_min, std::min(q_min - std::round(min_val / out_scale), q_max));

  // Use apply() which returns a new tensor with the transformation applied
  Tensor x_fq = x.apply<float>([out_scale, out_zp, q_min, q_max](float v) {
    float q = std::round(v / out_scale + out_zp);
    q = std::max(q_min, std::min(q, q_max));
    return (q - out_zp) * out_scale;
  });
  
  return x_fq;
}

void QATFullyConnectedLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(0);
  Tensor &input_ = context.getInput(0);

  float a_min = running_min_val;
  float a_max = running_max_val;
  Tensor input_fq = fakeQuantize(input_, a_min, a_max, q_min_a, q_max_a, training, a_scale, a_zp);
  if (training) {
    running_min_val = a_min;
    running_max_val = a_max;
  }

  float w_min = 0.0f;
  float w_max = 0.0f;
  Tensor weight_fq = fakeQuantize(weight, w_min, w_max, q_min_w, q_max_w, true, w_scale, w_zp);

  input_fq.dot(weight_fq, hidden_, false, false);

  if (!disable_bias) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

void QATFullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  const Tensor &derivative_ = context.getIncomingDerivative(0);
  Tensor &ret_ = context.getOutgoingDerivative(0);

  ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
}

void QATFullyConnectedLayer::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
  const Tensor &derivative_ = context.getIncomingDerivative(0);
  Tensor &input_ = context.getInput(0);

  if (!disable_bias) {
    Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);
    if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
      derivative_.sum({0, 1, 2}, djdb);
    } else {
      Tensor t = derivative_.sum({0, 1, 2});
      djdb.add_i(t);
    }
  }

  input_.dot_deriv_wrt_2(djdw, derivative_, false, false,
                         !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
}

} // namespace nntrainer
