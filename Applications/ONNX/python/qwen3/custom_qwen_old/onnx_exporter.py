from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_qwen3 import  NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding
import torch
import numpy as np

model_name = "Qwen/Qwen3-1.7B"

print("Loading model on CPU...")
# Load official model in FP32 first to get weights
official_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='cpu')

qwenConfig = official_model.config
# Create custom model directly in FP16
custom_model = NNTrainerQwen3ForCausalLM(qwenConfig)
# Load state dict in FP32, then convert to FP16
custom_model.load_state_dict(official_model.state_dict(), strict=False)
# Convert to FP16 immediately after loading
custom_model = custom_model.half()
print("<Model converted to fp16>")

# Force all model parameters to FP16 to ensure no FP32 remnants
for param in custom_model.parameters():
    param.data = param.data.half()
print("<All model parameters forced to FP16>")

rotary_emb = Qwen3RotaryEmbedding(qwenConfig)

x = torch.tensor([[52,],]).view(-1, 1)
position_ids = torch.arange(1).reshape(1, -1)
cos, sin = rotary_emb(x, position_ids)
# Convert to tensors and immediately to FP16 (except position_ids which is used for indexing)
cos = torch.tensor(cos.numpy()).half()
sin = torch.tensor(sin.numpy()).half()
variance_epsilon = torch.tensor([[1e-6,]]).half()

# Ensure all inputs are on the same device as the model first
device = next(custom_model.parameters()).device
x = x.to(device)
cos = cos.to(device)
sin = sin.to(device)
variance_epsilon = variance_epsilon.to(device)

# Convert inputs to fp16 for consistency (except input_ids which must remain integer)
cos = cos.half()
sin = sin.half()
variance_epsilon = variance_epsilon.half()
# Note: x (input_ids) remains as integer type for gather operation

logits_of_custom_model = custom_model(x,cos,sin,variance_epsilon)
logits_of_official_model = official_model(x).logits
print(logits_of_custom_model)
print(logits_of_official_model)

if (torch.allclose(logits_of_custom_model.float(), logits_of_official_model,atol=1e-3)):
    print("<All logits matched successfully (with fp16 tolerance)>")
else:
    print("<Some logits do not match>")
    print(f"Max difference: {torch.max(torch.abs(logits_of_custom_model.float() - logits_of_official_model))}")

logits_of_custom_model = logits_of_custom_model.detach().numpy()
print("saving logits in bin")
logits_of_custom_model.tofile("./modelling_logits.bin")

# Export with FP16 precision maintained throughout
torch.onnx.export(
    custom_model, (x, cos, sin, variance_epsilon),
    'qwen3_model.onnx',
    export_params=True,
    opset_version=17,  # Use even older opset version for maximum compatibility
    input_names=['input', 'cos', 'sin', 'variance_epsilon'],
    output_names=['output'],
    keep_initializers_as_inputs=False,  # Change back to default
    dynamic_axes=None,
    do_constant_folding=True,
    # Ensure FP16 export
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    # Add options to maintain FP16 precision
    training=torch.onnx.TrainingMode.EVAL,
    # Critical: Maintain FP16 precision throughout export
    # This ensures all tensors stay in FP16 format
    # Note: torch.onnx.export doesn't have a direct fp16 parameter,
    # but the model and inputs being FP16 should preserve the format
)

print("<FP16 Model exported successfully>")

# Try to fix the ONNX file by removing problematic exponent attributes
# try:
#     import onnx
#     print("Loading ONNX model to fix exponent attributes...")
#     model = onnx.load('qwen3_model.onnx')
    
#     # Remove problematic exponent attributes from nodes
#     for node in model.graph.node:
#         if hasattr(node, 'attribute'):
#             attrs_to_remove = []
#             for attr in node.attribute:
#                 if attr.name == 'exponent' and (not hasattr(attr, 'i') or attr.i is None):
#                     attrs_to_remove.append(attr)
#             for attr in attrs_to_remove:
#                 node.attribute.remove(attr)
    
#     # Save the fixed model
#     onnx.save(model, 'qwen3_model_fixed.onnx')
#     print("Fixed ONNX model saved as qwen3_model_fixed.onnx")
    
# except Exception as e:
#     print(f"Could not fix ONNX file: {e}")
#     print("Using original exported file")

# print("<FP16 Model exported successfully>")
