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
  ~QATFullyConnectedLayer() = default;

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

private:
  std::tuple<props::Unit> qat_fc_props;
  std::array<unsigned int, 2> weight_idx;
  enum FCParams { weight = 0, bias = 1 };
};

} // namespace nntrainer

#endif /* __QAT_FC_LAYER_H__ */
