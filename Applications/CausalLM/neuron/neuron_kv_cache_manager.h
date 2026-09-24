// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   neuron_kv_cache_manager.h
 * @date   24 Sep 2026
 * @brief  Host-side per-layer KV cache buffer management for LLMs running on
 *         the MediaTek NeuroPilot (Neuron) backend.
 *
 * @details Deliberately a much smaller MVP than
 * Applications/CausalLM/qnn/qnn_kv_cache_manager.h: no speculative decoding,
 * ring-buffer wraparound, or sliding-window compaction -- only what a
 * straightforward prefill-then-decode loop needs. Extend this class (or add
 * a sibling) if/when those features are needed for the neuron backend.
 *
 * Structurally this solves the same problem QnnKvCacheManager solves for
 * QNN, but the mechanism differs because the two backends expose different
 * app-facing APIs:
 *   - QNN's app layer binds `Qnn_Tensor_t`s directly against a QNN graph
 *     handle via QNN's own tensor-binding API; QnnKvCacheManager tracks raw
 *     uint8_t* buffers and `QnnKvOutputBinding`s for that.
 *   - The neuron_graph layer is driven through nntrainer's ordinary ccapi
 *     `ml::train::Model::inference(batch, input_ptrs, label_ptrs)`, which is
 *     purely positional: one raw float* per declared graph input, in graph
 *     order, and one float* per declared graph output, in graph order (see
 *     NeuralNetwork::inference(), nntrainer/models/neuralnet.cpp). So this
 *     class does not bind anything into a runtime itself -- it just owns the
 *     contiguous per-layer key/value cache storage the caller is expected to
 *     hand to `inference()` as the past_keys_L/past_values_L input pointers,
 *     and to copy the corresponding curr_keys_L/curr_values_L *output*
 *     pointers back into after each call.
 *
 * A neuron_graph .dla's cache tensors have a fixed, compile-time shape (see
 * docs/backend_guide/NEURON_BUILD.md and the config_*.yaml `cacheSize`
 * field from MediaTek's own LLM Command Line Tool) -- the caller must
 * therefore always pass the *entire* capacity_-sized buffer to `inference()`
 * as the past_keys_L/past_values_L pointer, not just the first length()
 * rows; the DLA itself only reads the rows up to whatever cache-position
 * input it was given (mask/cache-position handling is out of scope for this
 * class -- see Quick_Dot_AI_Neuron, not yet implemented, for that).
 */
#ifndef __NEURON_KV_CACHE_MANAGER_H__
#define __NEURON_KV_CACHE_MANAGER_H__

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace causallm {

/** @brief Host-owned per-layer K/V cache buffers for one neuron_graph-backed
 * LLM decode loop. Not thread-safe; one instance per in-flight generation. */
class NeuronKvCacheManager {
public:
  /**
   * @param num_layers   number of transformer layers (one key + one value
   *                     buffer each).
   * @param capacity     max cache positions the .dla was compiled for (the
   *                     YAML `cacheSize` MediaTek's own tool uses).
   * @param row_length   elements per cached row per layer, e.g.
   *                     num_key_value_heads * head_dim.
   */
  void configure(int num_layers, int capacity, int row_length) {
    if (num_layers <= 0 || capacity <= 0 || row_length <= 0) {
      throw std::invalid_argument(
        "NeuronKvCacheManager::configure: num_layers, capacity, and "
        "row_length must all be positive");
    }
    num_layers_ = num_layers;
    capacity_ = capacity;
    row_length_ = row_length;
    key_buffers_.assign(num_layers_,
                        std::vector<float>(static_cast<size_t>(capacity_) *
                                           row_length_, 0.0f));
    value_buffers_.assign(num_layers_,
                          std::vector<float>(static_cast<size_t>(capacity_) *
                                             row_length_, 0.0f));
    kv_len_ = 0;
  }

  /** @brief Drop all cached rows (start a new prompt); the underlying
   * storage is not reallocated or zeroed -- forward() calls only ever read
   * up to kv_len_ rows as "past", so stale bytes beyond that are inert. */
  void reset() { kv_len_ = 0; }

  int length() const { return kv_len_; }

  void setLength(int length) {
    if (length < 0 || length > capacity_) {
      throw std::out_of_range(
        "NeuronKvCacheManager::setLength: length out of [0, capacity] range");
    }
    kv_len_ = length;
  }

  int numLayers() const { return num_layers_; }
  int capacity() const { return capacity_; }
  int rowLength() const { return row_length_; }

  /** @brief Raw pointer to layer `layer`'s key cache buffer, sized
   * capacity() * rowLength() floats -- this is the full buffer to hand to
   * `inference()` as that layer's past_keys input pointer. */
  float *keyBuffer(int layer) {
    checkLayer(layer);
    return key_buffers_[layer].data();
  }

  /** @copydoc keyBuffer(int) but for the value cache. */
  float *valueBuffer(int layer) {
    checkLayer(layer);
    return value_buffers_[layer].data();
  }

  /**
   * @brief Append one step's newly produced key/value rows for one layer at
   * the current cache position, without advancing length() -- call this
   * once per layer, then advance() once after all layers are appended.
   *
   * @param layer          layer index.
   * @param curr_keys      row_length() * num_new_rows floats (the
   *                       curr_keys_<layer> output pointer from inference()).
   * @param curr_values    same shape, for curr_values_<layer>.
   * @param num_new_rows   rows produced this step (promptTokenBatchSize
   *                       during prefill, 1 during single-token decode).
   */
  void appendLayer(int layer, const float *curr_keys, const float *curr_values,
                   int num_new_rows) {
    checkLayer(layer);
    if (num_new_rows < 0 || kv_len_ + num_new_rows > capacity_) {
      throw std::out_of_range(
        "NeuronKvCacheManager::appendLayer: write would exceed cache "
        "capacity -- caller must evict/compact before appending further "
        "(not implemented by this MVP manager)");
    }
    const size_t dst_offset = static_cast<size_t>(kv_len_) * row_length_;
    const size_t n_elems = static_cast<size_t>(num_new_rows) * row_length_;
    std::copy(curr_keys, curr_keys + n_elems,
              key_buffers_[layer].data() + dst_offset);
    std::copy(curr_values, curr_values + n_elems,
              value_buffers_[layer].data() + dst_offset);
  }

  /** @brief Advance the shared cache position after appendLayer() has been
   * called for every layer this step. Kept as an explicit, separate step
   * (rather than folded into appendLayer()) so a caller can append all N
   * layers' outputs first and only then commit the new length once --
   * mirrors QnnKvCacheManager::advance()'s role for the same reason. */
  void advance(int num_new_rows) {
    if (num_new_rows < 0 || kv_len_ + num_new_rows > capacity_) {
      throw std::out_of_range(
        "NeuronKvCacheManager::advance: would exceed cache capacity");
    }
    kv_len_ += num_new_rows;
  }

private:
  void checkLayer(int layer) const {
    if (layer < 0 || layer >= num_layers_) {
      throw std::out_of_range("NeuronKvCacheManager: layer index out of range");
    }
  }

  int num_layers_ = 0;
  int capacity_ = 0;
  int row_length_ = 0;
  int kv_len_ = 0;

  std::vector<std::vector<float>> key_buffers_;
  std::vector<std::vector<float>> value_buffers_;
};

} // namespace causallm

#endif // __NEURON_KV_CACHE_MANAGER_H__
