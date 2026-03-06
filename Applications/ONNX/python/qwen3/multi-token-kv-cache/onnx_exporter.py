import torch
import argparse
from transformers import AutoTokenizer, Qwen3Config, AutoModelForCausalLM
from custom_qwen3 import  NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding
import onnx
import numpy as np
# from transformers.cache_utils import DynamicCache

# Parse command line arguments
parser = argparse.ArgumentParser(description='ONNX Exporter for Qwen3')
parser.add_argument('--seq-len', type=int, default=1, 
                    help='Sequence length for the model export (>1 for prefill, 1 for decode/generation)')
parser.add_argument('--model-name', type=str, default='qwen3.onnx',
                    help='Name of the output ONNX model file')
args = parser.parse_args()

SEQ_LEN = args.seq_len
MODEL_NAME = args.model_name

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

####Offical Model####

config = Qwen3Config.from_pretrained(model_name,attn_implementation="eager")  
official_model = AutoModelForCausalLM.from_pretrained(model_name,config=config).eval()

qwenConfig = official_model.config
custom_model =  NNTrainerQwen3ForCausalLM(qwenConfig).eval()
custom_model.load_state_dict(official_model.state_dict(),strict=False)

head_dim = config.head_dim
num_layers = 28 

# =========================
# Dummy input config
# =========================
BATCH = 1
PAST_LEN = 1024
NUM_HEADS = config.num_attention_heads
HEAD_DIM = config.head_dim
VOCAB_SIZE = config.vocab_size
DTYPE = torch.float32
DEVICE = "cpu"

# input_ids
input_ids = torch.randint(
    0, VOCAB_SIZE,
    (BATCH, SEQ_LEN),
    dtype=torch.long,
    device=DEVICE
)

# RoPE cos/sin (Qwen/LLaMA style)
cos = torch.randn(
    SEQ_LEN, HEAD_DIM,
    dtype=DTYPE,
    device=DEVICE
)

sin = torch.randn(
    SEQ_LEN, HEAD_DIM,
    dtype=DTYPE,
    device=DEVICE
)

eps=torch.tensor([1e-6],dtype=DTYPE,device=DEVICE) 

# past keys / values 
past_keys = []
past_values = []

for _ in range(num_layers):
    past_keys.append(
        torch.randn(
            BATCH, NUM_HEADS, PAST_LEN, HEAD_DIM,
            dtype=DTYPE,
            device=DEVICE
        )
    )
    past_values.append(
        torch.randn(
            BATCH, NUM_HEADS, PAST_LEN, HEAD_DIM,
            dtype=DTYPE,
            device=DEVICE
        )
    )
dummy_inputs = (
    input_ids.transpose(0,1),
    cos,
    sin,
    eps,
    past_keys,
    past_values
)

input_names = ["input_ids", "cos", "sin","variance_epsilion"]

for i in range(num_layers):
    input_names.append(f"past_key_{i}")

for i in range(num_layers):
    input_names.append(f"past_value_{i}")

output_names = ["logits"]

for i in range(num_layers):
    output_names.append(f"present_key_{i}")

for i in range(num_layers):
    output_names.append(f"present_value_{i}")

dynamic_axes = {}

for i in range(num_layers):
    dynamic_axes[f"past_key_{i}"] = { 2: "past_seq"}
    dynamic_axes[f"past_value_{i}"] = { 2: "past_seq"}
    dynamic_axes[f"present_key_{i}"] = { 2: "total_seq"}
    dynamic_axes[f"present_value_{i}"] = { 2: "total_seq"}

torch.onnx.export(
    custom_model,
    dummy_inputs,
    MODEL_NAME,
    input_names=input_names,
    output_names=output_names,
    opset_version=23,
    keep_initializers_as_inputs=False,
    export_params=True,
    dynamic_axes=dynamic_axes,
    dynamo=False,
    
)

print("ONNX export completed successfully")