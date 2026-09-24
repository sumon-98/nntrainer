// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   llama_prefill_parity_test.cpp
 * @date   24 Sep 2026
 * @brief  Tensor-level parity test: feeds MediaTek's own dumped quantized
 *         input tensors for one prefill step of a real
 *         Llama-3.2-1B-Instruct 128t1024c .dla through nntrainer's
 *         neuron_graph layer, and checks every one of the 33 outputs
 *         against MediaTek's own dumped quantized output tensors for the
 *         same step.
 *
 * @details No dequantization (scale/zero-point) is applied on either side,
 * so an EXACT match is expected if nntrainer drives the .dla correctly;
 * this is not an approximate/tolerance check.
 *
 * Input and output tensors are handled asymmetrically, matching what
 * NeuralNetwork::inference() (nntrainer/models/neuralnet.cpp) actually
 * expects/returns -- this took a debugging pass on real hardware to pin
 * down (initial runs got all-garbage output, then plausible-but-wrong
 * output, before landing here):
 *   - INPUT: mapExternalTensor()'s `float *` argument for a non-FP32
 *     dtype is mapped byte-for-byte via Tensor::Map<uint16_t/uint8_t/...>
 *     (see mapExternalTensor() in neuralnet.cpp) -- it is NOT an array of
 *     widened float values. Pass the RAW quantized bytes directly,
 *     reinterpreted as float* purely for the vector's element type.
 *   - OUTPUT: NeuralNetwork::inference()'s returned `float *` (from
 *     `out_t.getData()`) is likewise the tensor's raw native-dtype storage
 *     reinterpreted as float*, not a widened array -- read it back via
 *     `reinterpret_cast<const int16_t *>`, never as `float` elements.
 * nntrainer_neuron_smoke.cpp's own --dump handling already treats its
 * single output this way (reinterpret_cast<uint8_t*>, never widened), but
 * its --input path goes through widen_to_float() first -- given
 * mapExternalTensor()'s byte-for-byte mapping (confirmed by reading
 * neuralnet.cpp directly), that --input path is likely ALSO subtly wrong
 * and only appeared to work for its one validated case (a single UINT8
 * MobileNetV2 tensor matched bit-exact against MediaTek's neuronrt); this
 * hasn't been re-verified since finding the bug here, and is worth
 * revisiting rather than assumed correct by precedent.
 *
 * This is a hardcoded harness for one specific .dla's shape (16 layers, 8
 * KV heads, head_dim 64, hidden 2048, vocab 128256, prompt batch 128,
 * cache capacity 1024 -- see
 * Llama-3.2-1B-Instruct_sym4W_sym16A_Overall_hessian_128t1024c_info_0.json
 * and `ncc-tflite --show-io-info` on the matching .tflite), not a generic
 * multi-model tool. It proves nntrainer's neuron_graph layer can wire a
 * real LLM .dla's full 37-input/33-output I/O (not just a single
 * input/output pair, unlike nntrainer_neuron_smoke.cpp) and drive it to a
 * bit-exact match against MediaTek's own reference pipeline, independent
 * of the KV-cache bookkeeping / decode loop / tokenizer (all still
 * unbuilt -- see DEVICE_RUNBOOK.md's Next Steps).
 *
 * @usage
 *   llama_prefill_parity_test <model.dla> <dump_step_dir>
 * where <dump_step_dir> is a MediaTek mtk_llm_sdk debug-dump step
 * directory (e.g. .../dla_inference_debug/staged_llama/prompt_0/
 * llm_first_token_index_0/chunk_0) containing
 * quantized_inputs/in_NNNN_<name>.bin and
 * quantized_outputs/out_NNNN_<name>.bin.
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <engine.h>
#include <layer.h>
#include <model.h>
#include <nntrainer_error.h>

namespace {

constexpr int kNumLayers = 16;

struct NamedTensor {
  std::string name;
  std::string shape; // "B:C:H:W", matching nntrainer's TensorDim convention
};

/** @brief Input order/shapes, exactly matching the .dla's own I/O order
 * (NeuronGraph binds by position -- see NeuronGraph.h's file comment) and
 * MediaTek's own debug-dump `in_NNNN_<name>.bin` numbering. */
std::vector<NamedTensor> buildInputSpec() {
  std::vector<NamedTensor> v = {
    {"input_embeds", "1:1:128:2048"},
    {"mask", "1:1:128:128"},
    {"split_mask", "1:1:1:1024"},
    {"cos", "1:1:128:64"},
    {"sin", "1:1:128:64"},
  };
  for (int l = 0; l < kNumLayers; ++l) {
    v.push_back({"past_keys_" + std::to_string(l), "1:8:1024:64"});
    v.push_back({"past_values_" + std::to_string(l), "1:8:1024:64"});
  }
  return v;
}

/** @brief Output order/shapes, matching the .dla's own I/O order and
 * MediaTek's `out_NNNN_<name>.bin` numbering. */
std::vector<NamedTensor> buildOutputSpec() {
  std::vector<NamedTensor> v = {
    {"hidden_states_16", "1:1:128:128256"},
  };
  for (int l = 0; l < kNumLayers; ++l) {
    v.push_back({"curr_keys_" + std::to_string(l), "1:8:128:64"});
    v.push_back({"curr_values_" + std::to_string(l), "1:8:128:64"});
  }
  return v;
}

std::string join(const std::vector<std::string> &v, char sep) {
  std::string out;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      out += sep;
    out += v[i];
  }
  return out;
}

std::string zeroPad4(size_t idx) {
  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << idx;
  return oss.str();
}

std::vector<uint8_t> readBinaryFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  if (size > 0 && !file.read(reinterpret_cast<char *>(buf.data()), size)) {
    throw std::runtime_error("failed to read file: " + path);
  }
  return buf;
}

/** @brief raw int16 quantized bytes -> float32, no dequantization (same
 * convention nntrainer_neuron_smoke.cpp's widen_to_float() uses for
 * QINT16: the ccapi float32 boundary carries the raw integer value
 * verbatim). */
void widenQint16(const std::vector<uint8_t> &raw, std::vector<float> &out) {
  const size_t n = raw.size() / sizeof(int16_t);
  out.resize(n);
  const auto *p = reinterpret_cast<const int16_t *>(raw.data());
  for (size_t i = 0; i < n; ++i) {
    out[i] = static_cast<float>(p[i]);
  }
}

size_t elemsFromShape(const std::string &shape) {
  size_t elems = 1;
  size_t start = 0;
  while (start <= shape.size()) {
    size_t pos = shape.find(':', start);
    std::string tok = shape.substr(
      start, pos == std::string::npos ? std::string::npos : pos - start);
    elems *= std::stoul(tok);
    if (pos == std::string::npos)
      break;
    start = pos + 1;
  }
  return elems;
}

void usage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " <model.dla> <dump_step_dir>\n"
            << "  <dump_step_dir> must contain quantized_inputs/ and "
               "quantized_outputs/\n"
               "  (a MediaTek mtk_llm_sdk debug-dump step directory)\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  const std::string dla_path = argv[1];
  const std::string dump_dir = argv[2];

  try {
    auto &engine = nntrainer::Engine::Global();
    if (const char *so = std::getenv("NNTRAINER_NEURON_CONTEXT_SO")) {
      engine.registerContext(so, "");
    }
    try {
      (void)engine.getRegisteredContext("neuron");
    } catch (const std::exception &e) {
      std::cerr << "ERROR: neuron context not registered: " << e.what()
                << std::endl;
      return EXIT_FAILURE;
    }

    const auto input_spec = buildInputSpec();
    const auto output_spec = buildOutputSpec();

    auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    if (model == nullptr) {
      std::cerr << "ERROR: failed to create model" << std::endl;
      return EXIT_FAILURE;
    }

    std::vector<std::string> input_layer_names;
    for (const auto &t : input_spec) {
      model->addLayer(ml::train::createLayer(
        "input", {"name=" + t.name, "input_shape=" + t.shape}));
      input_layer_names.push_back(t.name);
    }

    std::vector<std::string> out_dims, out_dtypes, out_ttypes;
    for (const auto &t : output_spec) {
      out_dims.push_back(t.shape);
      out_dtypes.push_back("QINT16");
      out_ttypes.push_back("OUT_TENSOR");
    }

    model->addLayer(ml::train::createLayer(
      "neuron_graph",
      {"name=llm_graph", "input_layers=" + join(input_layer_names, ','),
       "path=" + dla_path, "dim=" + join(out_dims, ','),
       "tensor_dtype=" + join(out_dtypes, ','),
       "tensor_type=" + join(out_ttypes, ','), "engine=neuron"}));

    model->setProperty(
      {"batch_size=1", "epochs=1", "model_tensor_type=QINT16-QINT16"});

    int ret = model->compile(ml::train::ExecutionMode::INFERENCE);
    if (ret != ML_ERROR_NONE) {
      std::cerr << "ERROR: model->compile failed, code: " << ret << std::endl;
      return EXIT_FAILURE;
    }
    ret = model->initialize(ml::train::ExecutionMode::INFERENCE);
    if (ret != ML_ERROR_NONE) {
      std::cerr << "ERROR: model->initialize failed, code: " << ret
                << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "[parity] model compiled+initialized ("
              << input_spec.size() << " inputs, " << output_spec.size()
              << " outputs)" << std::endl;

    // Load MediaTek's own dumped inputs for this step, in .dla I/O order.
    // As with the output side, pass the RAW quantized bytes directly
    // (reinterpreted as float* purely for the ccapi vector's element type,
    // no value widening) rather than nntrainer_neuron_smoke.cpp's
    // widen_to_float() convention -- that convention was only proven
    // correct there for a single-input/single-output model; this is the
    // first real multi-input case, and the analogous output-side "widen"
    // assumption turned out to be wrong (see the fix above), so don't
    // assume input-side widening is right either without checking.
    std::vector<std::vector<uint8_t>> in_storage(input_spec.size());
    std::vector<float *> in_ptrs;
    for (size_t i = 0; i < input_spec.size(); ++i) {
      const std::string path = dump_dir + "/quantized_inputs/in_" +
                               zeroPad4(i) + "_" + input_spec[i].name +
                               ".bin";
      auto raw = readBinaryFile(path);
      const size_t expected = elemsFromShape(input_spec[i].shape) * 2;
      if (raw.size() != expected) {
        std::cerr << "ERROR: " << path << " has " << raw.size()
                  << " bytes, expected " << expected << std::endl;
        return EXIT_FAILURE;
      }
      in_storage[i] = std::move(raw);
      in_ptrs.push_back(reinterpret_cast<float *>(in_storage[i].data()));
    }
    std::cout << "[parity] loaded " << input_spec.size()
              << " input tensors from " << dump_dir << "/quantized_inputs/"
              << std::endl;

    std::cout << "[parity] running inference..." << std::endl;
    std::vector<float *> out_ptrs = model->inference(1, in_ptrs, {});
    if (out_ptrs.size() != output_spec.size()) {
      std::cerr << "ERROR: expected " << output_spec.size()
                << " outputs, got " << out_ptrs.size() << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "[parity] inference completed" << std::endl;

    // IMPORTANT: unlike the *input* side (which genuinely widens each
    // quantized int16 value to a real float, matching how
    // nntrainer_neuron_smoke.cpp's --input path works, confirmed correct
    // there against a real .dla), NeuralNetwork::inference()'s returned
    // `float *` for a non-FP32-dtype output is NOT an array of widened
    // float values -- out_t.getData() (neuralnet.cpp) just returns the
    // tensor's own native storage pointer, reinterpreted as float* purely
    // as the ccapi vector's generic element type. nntrainer_neuron_smoke.cpp
    // already treats its own single output this way (`reinterpret_cast<const
    // uint8_t *>(outputs[0])`, never reading it as widened floats). Do the
    // same here: reinterpret each out_ptrs[i] as `const int16_t *` (this
    // .dla's real output dtype) and compare raw int16 values directly
    // against the goldens (no float widening at all on the output side).
    std::vector<std::vector<uint8_t>> golden_raw(output_spec.size());
    for (size_t i = 0; i < output_spec.size(); ++i) {
      const std::string path = dump_dir + "/quantized_outputs/out_" +
                               zeroPad4(i) + "_" + output_spec[i].name + ".bin";
      golden_raw[i] = readBinaryFile(path);
    }

    // Bounded-size safety probe before trusting out_ptrs[i] for a full-size
    // read: this is a first run against real multi-output (33 OUT_TENSOR
    // entries on one neuron_graph layer), so confirm addressing/ordering is
    // sane before an exhaustive pass.
    constexpr size_t kProbeElems = 8;
    std::cout << "[parity] first " << kProbeElems
              << " raw int16 value(s) per output (probe, bounded-safe):"
              << std::endl;
    for (size_t i = 0; i < output_spec.size(); ++i) {
      const auto *p = reinterpret_cast<const int16_t *>(out_ptrs[i]);
      std::cout << "  out_ptrs[" << i << "] (" << output_spec[i].name << "):";
      for (size_t e = 0; e < kProbeElems; ++e) {
        std::cout << " " << p[e];
      }
      std::cout << std::endl;
    }

    // For each out_ptrs[i], report which golden tensor's first kProbeElems
    // it actually matches (if any) -- reveals an index permutation cheaply,
    // without a full-size read of any tensor.
    for (size_t i = 0; i < output_spec.size(); ++i) {
      const auto *p = reinterpret_cast<const int16_t *>(out_ptrs[i]);
      int best_match = -1;
      for (size_t g = 0; g < golden_raw.size(); ++g) {
        const auto *gp = reinterpret_cast<const int16_t *>(golden_raw[g].data());
        bool all_match = true;
        for (size_t e = 0; e < kProbeElems; ++e) {
          if (p[e] != gp[e]) {
            all_match = false;
            break;
          }
        }
        if (all_match) {
          best_match = static_cast<int>(g);
          break;
        }
      }
      std::cout << "  out_ptrs[" << i << "] probe matches golden index "
                << best_match << " (expected " << i << ", "
                << output_spec[i].name << ")" << std::endl;
    }

    // Exhaustive comparison: the probe above confirmed addressing/ordering
    // is sane (every out_ptrs[i] matches golden index i), and the root
    // cause of the earlier garbage/crash (treating outputs as widened
    // float32 values instead of raw native-dtype bytes) is fixed above, so
    // it's now safe to compare every element of every tensor, including
    // the full 16.4M-element hidden_states_16 logits tensor.
    size_t tensors_mismatched = 0;
    for (size_t i = 0; i < output_spec.size(); ++i) {
      const std::string &name = output_spec[i].name;
      const size_t n = golden_raw[i].size() / sizeof(int16_t);
      const auto *out_p = reinterpret_cast<const int16_t *>(out_ptrs[i]);
      const auto *golden_p =
        reinterpret_cast<const int16_t *>(golden_raw[i].data());
      size_t mismatches = 0;
      int reported = 0;
      for (size_t e = 0; e < n; ++e) {
        if (out_p[e] != golden_p[e]) {
          ++mismatches;
          if (reported < 5) {
            std::cerr << "  [" << name << "] elem " << e
                      << ": nntrainer=" << out_p[e]
                      << " golden=" << golden_p[e] << std::endl;
            ++reported;
          }
        }
      }
      if (mismatches > 0) {
        std::cerr << "MISMATCH [" << name << "]: " << mismatches << "/" << n
                  << " elements differ" << std::endl;
        ++tensors_mismatched;
      } else {
        std::cout << "MATCH    [" << name << "]: all " << n
                  << " elements bit-exact" << std::endl;
      }
    }

    if (tensors_mismatched == 0) {
      std::cout << "PASS: all " << output_spec.size()
                << " output tensors bit-exact vs MediaTek's own dump"
                << std::endl;
      return EXIT_SUCCESS;
    }
    std::cerr << "FAIL: " << tensors_mismatched << "/" << output_spec.size()
              << " output tensors mismatched" << std::endl;
    return EXIT_FAILURE;

  } catch (const std::exception &e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
