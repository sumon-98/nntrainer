// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_fc_layer.h
 * @brief  QAT Fully Connected Layer — Weight-Only Fake Quantization
 *
 * This layer implements weight-only Quantization Aware Training (QAT) using
 * the Straight-Through Estimator (STE). During the forward pass, the weights
 * are fake-quantized to simulate INT8 precision, while activations pass
 * through in full FP32 precision. During backpropagation, gradients flow
 * through the non-differentiable round/clamp operations via STE.
 *
 * This layer is a drop-in replacement for NNTrainer's built-in
 * fully_connected layer. Register it as "qat_fully_connected" and use it
 * anywhere you would use "fully_connected".
 */

#ifndef __QAT_FC_LAYER_H__
#define __QAT_FC_LAYER_H__

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

class QATFullyConnectedLayer : public LayerImpl {
public:
  QATFullyConnectedLayer();
  ~QATFullyConnectedLayer();

  QATFullyConnectedLayer(QATFullyConnectedLayer &&rhs) noexcept = default;
  QATFullyConnectedLayer &operator=(QATFullyConnectedLayer &&rhs) = default;

  void finalize(InitLayerContext &context) override;
  void forwarding(RunLayerContext &context, bool training) override;
  void calcDerivative(RunLayerContext &context) override;
  void calcGradient(RunLayerContext &context) override;

  bool supportBackwarding() const override { return true; }

  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  const std::string getType() const override {
    return QATFullyConnectedLayer::type;
  }

  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "qat_fully_connected";

  /**
   * @brief Print the learned quantization scale and zero-point
   */
  void printQATStats() const;

private:
  std::tuple<props::Unit> qat_fc_props;
  std::array<unsigned int, 2> weight_idx;
  enum FCParams { weight = 0, bias = 1 };

  // Weight quantization parameters (INT8: [-128, 127])
  float q_min_weight;
  float q_max_weight;
  float momentum;

  // Flag to track if layer was properly initialized
  bool initialized = false;

  // Weight quantization running stats (EMA-tracked)
  Tensor weight_running_min;
  Tensor weight_running_max;

  // Cached fake-quantized weight tensor (used by STE in backward pass)
  Tensor w_fq;

  /**
   * @brief Apply fake quantization to a tensor using running min/max EMA
   *
   * @param x         The tensor to fake-quantize
   * @param running_min  EMA-tracked minimum value (scalar tensor)
   * @param running_max  EMA-tracked maximum value (scalar tensor)
   * @param q_min     Minimum of the quantized integer range
   * @param q_max     Maximum of the quantized integer range
   * @param training  If true, update running stats with current batch
   * @return Tensor   The fake-quantized tensor (same shape as x)
   */
  Tensor fakeQuantize(Tensor &x, Tensor &running_min, Tensor &running_max,
                      float q_min, float q_max, bool training);
};

} // namespace nntrainer

#endif /* __QAT_FC_LAYER_H__ */
