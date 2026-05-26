// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_fc_layer.h
 * @brief  QAT Fully Connected Layer - DIAGNOSTIC VERSION
 *
 * This is a MINIMAL version with NO fake quantization.
 * It behaves identically to the built-in FullyConnectedLayer.
 * Purpose: isolate whether the crash is in the layer logic or the setup.
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

  // Added to extract properties later
  void printQATStats() const;

private:
  std::tuple<props::Unit> qat_fc_props;
  std::array<unsigned int, 2> weight_idx;
  enum FCParams { weight = 0, bias = 1 };

  // QAT specific properties
  float q_min_act;
  float q_max_act;
  float q_min_weight;
  float q_max_weight;
  float momentum;

  // Flag to track if layer was properly initialized
  bool initialized = false;

  // QAT running stats
  Tensor act_running_min;
  Tensor act_running_max;
  Tensor weight_running_min;
  Tensor weight_running_max;

  // QAT fake-quantized tensors for STE
  Tensor x_fq;
  Tensor w_fq;

  // Helper method for Fake Quantization
  Tensor fakeQuantize(Tensor &x, Tensor &running_min, Tensor &running_max, 
                      float q_min, float q_max, bool training);
};

} // namespace nntrainer

#endif /* __QAT_FC_LAYER_H__ */
