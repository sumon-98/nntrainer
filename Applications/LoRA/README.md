# LoRA Implementation in NNTrainer

This directory contains different implementations of LoRA (Low-Rank Adaptation) training to demonstrate the performance differences.

## Files

- `lora_train.cpp`: Standard backpropagation training (baseline)
- `lora_test.cpp`: Incorrect LoRA implementation (slower due to training both original weights and adapters)
- `lora_train_correct.cpp`: Correct LoRA implementation (with frozen base weights, training only adapters)
- `lora_run.cpp`: Inference pipeline for trained models

## Performance Comparison

### Why lora_test.cpp is Slower than lora_train.cpp

The `lora_test.cpp` implementation is slower than standard backpropagation (`lora_train.cpp`) for the following reasons:

1. **Double Computation**: It computes both the standard forward/backward passes AND the additional LoRA adaptations, essentially doing twice the work.

2. **Parameter Update Overhead**: Instead of the typical LoRA approach (freezing original weights, training only adapters), this implementation trains both, negating LoRA's efficiency benefits.

3. **Memory Overhead**: Additional tensors for LoRA computations (`loraTmp`, `loraOut`) consume more memory and require additional memory operations.

### Correct LoRA Implementation

The `lora_train_correct.cpp` file demonstrates the proper way to implement LoRA:

1. **Frozen Base Weights**: Original weights are frozen after pre-training
2. **Train Only Adapters**: Only the low-rank adapter matrices (loraA, loraB) are trained
3. **Efficient Computation**: During inference, adapters can be merged with original weights

## Building and Running

To build the LoRA examples:

```bash
meson setup build
ninja -C build
```

To run the different implementations:

```bash
# Standard training (baseline)
./build/Applications/LoRA/jni/lora_train

# Incorrect LoRA implementation (slower)
./build/Applications/LoRA/jni/lora_test

# Correct LoRA implementation (faster)
./build/Applications/LoRA/jni/lora_train_correct

# Inference with trained model
./build/Applications/LoRA/jni/lora_run
```

## Key Differences in Implementation

### Incorrect Implementation (lora_test.cpp)
- Trains both original weights and LoRA adapters
- Computes gradients for all parameters
- No performance benefit over standard training

### Correct Implementation (lora_train_correct.cpp)
- Freezes original weights during training
- Only trains LoRA adapter matrices
- Significantly fewer parameters to train
- True parameter-efficient fine-tuning

## Performance Expectations

When properly implemented, LoRA should be:
- Faster to train than full fine-tuning
- Use less memory
- Require fewer trainable parameters
- Achieve comparable accuracy to full fine-tuning

The current `lora_test.cpp` implementation does not follow these principles, which is why it's slower than standard backpropagation.