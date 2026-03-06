import torch,onnx

def make_dummy_tensors(batch, num_heads, seq_len, past_seq_len, head_dim, device="cpu", dtype=torch.float32):
    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    past_k = torch.randn(batch, num_heads, past_seq_len, head_dim, device=device, dtype=dtype)
    past_v = torch.randn(batch, num_heads, past_seq_len, head_dim, device=device, dtype=dtype)
    return q, k, v, past_k, past_v

class Model(torch.nn.Module):
    def forward(
        self, q, k, v, past_k, past_v
    ):  
       if torch.onnx.is_in_onnx_export():
            o1,o2,o3,_ = torch.onnx.ops.attention(
                q,
                k,
                v,
                None,
                past_k,
                past_v,
            )
            return o1,o2,o3
       else:
           print("Hello")

batch = 1
num_heads = 16
seq_len = 8          # dynamic axis will allow other seq lengths
past_seq_len = 4     # dynamic axis will allow other past seq lengths
head_dim = 64

device = "cpu"
q, k, v, past_k, past_v = make_dummy_tensors(batch, num_heads, seq_len, past_seq_len, head_dim, device=device)

input_names = ["q", "k", "v", "past_k", "past_v"]
output_names = ["out", "present_k", "present_v"]

past_seq = torch.export.Dim("past_seq_len")
    
dynamic_shapes = {
        "q":None,
        "k":None,
        "v":None,
        "past_k": {2: past_seq},
        "past_v": {2: past_seq},
        # "present_k": {2: "past_seq_len_plus_seq"},
        # "present_v": {2: "past_seq_len_plus_seq"},
}

model = Model()

torch.onnx.export(
        model.eval(),
        (q, k, v, past_k, past_v),
        "testopset23_attention.onnx",
        opset_version=23,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        # dynamo=True is recommended for newer PyTorch; remove if unsupported in your version
        dynamo=True,
)

 