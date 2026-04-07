// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lora_train.cpp
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  LoRA training data pipeline implementation
 */

#include "lora_train.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace causallm {

TrainingDataGenerator::TrainingDataGenerator(tokenizers::Tokenizer *tokenizer,
                                             unsigned int seq_len,
                                             unsigned int vocab_size) :
  tokenizer_(tokenizer),
  seq_len_(seq_len),
  vocab_size_(vocab_size),
  current_idx_(0) {}

void TrainingDataGenerator::loadTextFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open training data file: " + path);
  }

  std::string line;
  int count = 0;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    auto ids = tokenizer_->Encode(line);
    samples_.push_back(ids);
    count++;
  }

  std::cout << "[TrainingData] Loaded " << path
            << " line by line, total: " << count << " samples." << std::endl;
}

void TrainingDataGenerator::addTokenIds(const std::vector<int> &ids) {
  samples_.push_back(ids);
}

unsigned int TrainingDataGenerator::getNumSamples() const {
  return static_cast<unsigned int>(samples_.size());
}

void TrainingDataGenerator::reset() { current_idx_ = 0; }

int TrainingDataGenerator::dataCb(float **input, float **label, bool *last,
                                  void *user_data) {
  auto *self = static_cast<TrainingDataGenerator *>(user_data);

  if (self->current_idx_ >= self->samples_.size()) {
    *last = true;
    self->reset();
    return 0;
  }

  const auto &ids = self->samples_[self->current_idx_];
  unsigned int available = ids.size();

  // Input: [t_0, t_1, ..., t_{L-1}, pad, ..., pad]
  // Label: [t_1, t_2, ..., t_L,     pad, ..., pad]
  for (unsigned int j = 0; j < self->seq_len_; j++) {
    // Input
    if (j < available) {
      input[0][j] = static_cast<float>(ids[j]);
    } else {
      input[0][j] = 0.0f; // pad
    }

    // // Label: zero out first
    // for (unsigned int v = 0; v < self->vocab_size_; ++v) {
    //   label[0][j * self->vocab_size_ + v] = 0.0f;
    // }

    // // Label: set one-hot
    // if (j + 1 < available) {
    //   label[0][j * self->vocab_size_ + ids[j + 1]] = 1.0f;
    // } else {
    //   label[0][j * self->vocab_size_ + 0] = 1.0f; // pad or eos
    // }
    // CLINE FIX SUGGESTION
    // Label: just the next token ID (not one-hot)
    if (j + 1 < available) {
      label[0][j] = static_cast<float>(ids[j + 1]);
    } else {
      label[0][j] = 0.0f; // pad
    }
  }

  self->current_idx_++;
  *last = false;
  return 0;
}

} // namespace causallm
