// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   neuron_kv_cache_manager_test.cpp
 * @date   24 Sep 2026
 * @brief  Standalone host-side unit test for NeuronKvCacheManager. Pure
 *         bookkeeping logic -- no NPU, no .dla, no device needed to run
 *         this; it's a plain desktop executable.
 */
#include "neuron_kv_cache_manager.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

#define CHECK_THROWS(expr)                                                   \
  do {                                                                       \
    bool threw = false;                                                     \
    try {                                                                   \
      expr;                                                                 \
    } catch (const std::exception &) {                                     \
      threw = true;                                                        \
    }                                                                       \
    if (!threw) {                                                          \
      std::fprintf(stderr, "FAIL %s:%d: expected throw from: %s\n",         \
                   __FILE__, __LINE__, #expr);                             \
      ++g_failures;                                                        \
    }                                                                       \
  } while (0)

void test_basic_configure_and_bounds() {
  causallm::NeuronKvCacheManager mgr;
  mgr.configure(/*num_layers=*/2, /*capacity=*/8, /*row_length=*/4);
  CHECK(mgr.numLayers() == 2);
  CHECK(mgr.capacity() == 8);
  CHECK(mgr.rowLength() == 4);
  CHECK(mgr.length() == 0);

  CHECK_THROWS(mgr.keyBuffer(-1));
  CHECK_THROWS(mgr.keyBuffer(2));
  CHECK_THROWS(mgr.valueBuffer(2));
  CHECK_THROWS(mgr.setLength(9));
  CHECK_THROWS(mgr.setLength(-1));

  causallm::NeuronKvCacheManager bad;
  CHECK_THROWS(bad.configure(0, 8, 4));
  CHECK_THROWS(bad.configure(2, 0, 4));
  CHECK_THROWS(bad.configure(2, 8, 0));
}

// Simulates a 1-layer, row_length=2, capacity=6 prefill-then-decode loop:
// prefill appends 3 rows, then two single-token decode steps append 1 row
// each. Verifies appendLayer()+advance() lands rows at the right offsets and
// length() tracks correctly, exactly the pattern a real decode loop needs.
void test_prefill_then_decode_append_sequence() {
  causallm::NeuronKvCacheManager mgr;
  mgr.configure(/*num_layers=*/1, /*capacity=*/6, /*row_length=*/2);

  // Prefill: 3 rows of keys/values, distinct values so we can check offsets.
  std::vector<float> prefill_keys = {1, 1, 2, 2, 3, 3};
  std::vector<float> prefill_values = {10, 10, 20, 20, 30, 30};
  mgr.appendLayer(0, prefill_keys.data(), prefill_values.data(),
                  /*num_new_rows=*/3);
  mgr.advance(3);
  CHECK(mgr.length() == 3);

  const float *k = mgr.keyBuffer(0);
  const float *v = mgr.valueBuffer(0);
  CHECK(k[0] == 1 && k[1] == 1 && k[2] == 2 && k[3] == 2 && k[4] == 3 &&
        k[5] == 3);
  CHECK(v[0] == 10 && v[2] == 20 && v[4] == 30);

  // Decode step 1: append row 4 at position 3.
  std::vector<float> dec1_key = {4, 4};
  std::vector<float> dec1_val = {40, 40};
  mgr.appendLayer(0, dec1_key.data(), dec1_val.data(), 1);
  mgr.advance(1);
  CHECK(mgr.length() == 4);
  CHECK(k[6] == 4 && k[7] == 4);
  CHECK(v[6] == 40 && v[7] == 40);

  // Decode step 2: append row 5 at position 4.
  std::vector<float> dec2_key = {5, 5};
  std::vector<float> dec2_val = {50, 50};
  mgr.appendLayer(0, dec2_key.data(), dec2_val.data(), 1);
  mgr.advance(1);
  CHECK(mgr.length() == 5);
  CHECK(k[8] == 5 && k[9] == 5);

  // Capacity is 6; one more single-token step fits exactly, a second doesn't.
  std::vector<float> dec3_key = {6, 6};
  std::vector<float> dec3_val = {60, 60};
  mgr.appendLayer(0, dec3_key.data(), dec3_val.data(), 1);
  mgr.advance(1);
  CHECK(mgr.length() == 6);
  CHECK_THROWS(mgr.advance(1));
  CHECK_THROWS(mgr.appendLayer(0, dec3_key.data(), dec3_val.data(), 1));
}

void test_reset_reuses_capacity_for_a_new_prompt() {
  causallm::NeuronKvCacheManager mgr;
  mgr.configure(1, 4, 2);
  std::vector<float> key = {1, 1, 2, 2};
  std::vector<float> val = {10, 10, 20, 20};
  mgr.appendLayer(0, key.data(), val.data(), 2);
  mgr.advance(2);
  CHECK(mgr.length() == 2);

  mgr.reset();
  CHECK(mgr.length() == 0);
  // After reset, the manager must accept a full-capacity append again (the
  // stale bytes beyond length() are inert and get overwritten).
  std::vector<float> key2 = {9, 9, 9, 9, 9, 9, 9, 9};
  std::vector<float> val2 = {99, 99, 99, 99, 99, 99, 99, 99};
  mgr.appendLayer(0, key2.data(), val2.data(), 4);
  mgr.advance(4);
  CHECK(mgr.length() == 4);
}

void test_multi_layer_independence() {
  causallm::NeuronKvCacheManager mgr;
  mgr.configure(/*num_layers=*/3, /*capacity=*/4, /*row_length=*/1);
  std::vector<float> l0k = {1, 1};
  std::vector<float> l1k = {2, 2};
  std::vector<float> l2k = {3, 3};
  std::vector<float> dummy_v = {0, 0};
  mgr.appendLayer(0, l0k.data(), dummy_v.data(), 2);
  mgr.appendLayer(1, l1k.data(), dummy_v.data(), 2);
  mgr.appendLayer(2, l2k.data(), dummy_v.data(), 2);
  mgr.advance(2);

  CHECK(mgr.keyBuffer(0)[0] == 1);
  CHECK(mgr.keyBuffer(1)[0] == 2);
  CHECK(mgr.keyBuffer(2)[0] == 3);
}

} // namespace

int main() {
  test_basic_configure_and_bounds();
  test_prefill_then_decode_append_sequence();
  test_reset_reuses_capacity_for_a_new_prompt();
  test_multi_layer_independence();

  if (g_failures == 0) {
    std::printf("PASS: all NeuronKvCacheManager tests passed\n");
    return 0;
  }
  std::fprintf(stderr, "FAILED: %d check(s) failed\n", g_failures);
  return 1;
}
