// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Sumon Nath <sumon.nath@samsung.com>
 *
 * @file   eager_attention_layer.h
 * @date   14 January 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is EagerAttention Layer Class for Neural Network with KV Cache support
 *
 */

#ifndef __EAGER_ATTENTION_LAYER_H__
#define __EAGER_ATTENTION_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_devel.h>
#include <limits>

namespace nntrainer {

/**
 * @class   EagerAttention Layer
 * @brief   EagerAttention Layer with KV Cache support
 */
class EagerAttentionLayer : public virtual Layer {
public:
  /**
   * @brief     Constructor of EagerAttention Layer
   */
  EagerAttentionLayer();

  /**
   * @brief     Destructor of EagerAttention Layer
   */
  ~EagerAttentionLayer();

  /**
   *  @brief  Move constructor of EagerAttentionLayer.
   *  @param[in] EagerAttentionLayer &&
   */
  EagerAttentionLayer(EagerAttentionLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs EagerAttentionLayer to be moved.
   */
  EagerAttentionLayer &operator=(EagerAttentionLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return EagerAttentionLayer::type; };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  static constexpr const char *type = "eager_attention";

protected:
  /**
   * @brief     Finalize the eager_attention layer with the given context
   * @param[in] context InitLayerContext
   *
   * @note This function provides the basic finalize details which can be shared
   * with derived classes as well
   */
  void finalizeCommon(InitLayerContext &context);

  std::tuple<props::ScaledDotProduct, props::CausalMask> eager_attention_props;

private:
  ActiFunc sm;                        /** softmax activation operation */
  std::array<unsigned int, 10> wt_idx; /**< indices of the weights and tensors */
  
  // Tensor indices for KV cache
  enum TensorParams {
    query = 0,
    key = 1,
    value = 2,
    weights = 3,
    cache_key = 4,
    cache_value = 5,
    present_key = 6,
    present_value = 7,
    cos = 8,
    sin = 9
  };
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __EAGER_ATTENTION_LAYER_H__ */
