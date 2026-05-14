// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qat_fc_layer.h
 * @brief  QAT Fully Connected Layer with optional LoRA support
 *
 * This layer supports two modes using the same code:
 *
 * MODE 1 — Full Model QAT (lora_rank not set):
 *   - Forward: output = input * fakeQuantize(W) + bias
 *   - QAT is applied to the base weight W
 *   - All weights are trainable, gradients flow via STE
 *
 * MODE 2 — LoRA QAT (lora_rank > 0):
 *   - Base weight W is FROZEN and used as vanilla FP32 (no fake-quant on W)
 *   - LoRA weights A and B are TRAINABLE and FAKE-QUANTIZED
 *   - Forward: output = input * W_frozen
 *                      + input * fakeQuant(A) * fakeQuant(B) * scaling
 *                      + bias
 *   - The LoRA adapters train aware of INT8 quantization noise
 *   - At deployment: store A, B as INT8 using the learned scale/zero-point
 *
 * Register as "qat_fully_connected". Drop-in replacement for "fully_connected".
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

  void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  inline static const std::string type = "qat_fully_connected";

  /**
   * @brief Print the learned quantization scale and zero-point
   */
  void printQATStats() const;

private:
  // Layer properties: unit, lora_rank, lora_alpha
  float lora_scaling;
  std::tuple<props::Unit, props::LoraRank, props::LoraAlpha> qat_fc_props;

  // Base weight and bias indices
  std::array<unsigned int, 2> weight_idx;
  enum FCParams { weight = 0, bias = 1 };

  // LoRA weight indices: loraA, loraB, loraTmp, loraOut
  std::array<unsigned int, 4> lora_idx;
  enum LORAParams { loraA = 0, loraB = 1, loraTmp = 2, loraOut = 3 };

  // Quantization parameters (INT8: [-128, 127])
  float q_min;
  float q_max;
  float momentum;

  bool initialized = false;

  // --- Mode 1 (full QAT): running stats for base weight W ---
  Tensor weight_running_min;
  Tensor weight_running_max;
  Tensor w_fq; // cached fake-quantized base weight

  // --- Mode 2 (LoRA QAT): running stats for LoRA weights A, B ---
  Tensor lora_a_running_min;
  Tensor lora_a_running_max;
  Tensor lora_b_running_min;
  Tensor lora_b_running_max;
  Tensor a_fq; // cached fake-quantized loraA
  Tensor b_fq; // cached fake-quantized loraB

  /**
   * @brief Apply fake quantization to a tensor using running min/max EMA
   */
  Tensor fakeQuantize(Tensor &x, Tensor &running_min, Tensor &running_max,
                      float q_min, float q_max, bool training);
};

} // namespace nntrainer

#endif /* __QAT_FC_LAYER_H__ */
