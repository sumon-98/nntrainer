import torch
from transformers import AutoTokenizer, Qwen3Config, AutoModelForCausalLM, DynamicCache
from custom_qwen3 import  NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding
import onnx
import numpy as np

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = Qwen3Config.from_pretrained(model_name,attn_implementation="eager")
official_model = AutoModelForCausalLM.from_pretrained(model_name,config = config).eval()

num_tokens_to_generate = 20

# Prompt
prompt = "Tell me a dad joke about a computer: "
print("\nInput prompt: ",prompt)

input_encoding = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids

####################### Official Model #############################

# Prefill Stage

past_cache = DynamicCache()

outputs = official_model(
    input_encoding,
    past_key_values=past_cache,    
)

#KV cache of input prompt done. Calculating token id of the lastest token generated
output_token = outputs.logits[:, -1, :].argmax(dim=-1)

#Decode Stage

response = []

# for step in range(num_tokens_to_generate):
    
#     outputs = official_model(
#         output_token.unsqueeze(0),
#         past_key_values=past_cache,          
#     )
    
#     output_token = outputs.logits[:, -1, :].argmax(dim=-1)
#     response.append(output_token.item())
       
# decoded = tokenizer.decode(response, skip_special_tokens=True)
# print("Official model response: ",decoded)

####################### NNTrainer Model #############################

qwenConfig = official_model.config
custom_model =  NNTrainerQwen3ForCausalLM(qwenConfig)
custom_model.load_state_dict(official_model.state_dict(),strict=False)

# Calculating rotary embeddings for input tokens and positions

rotary_emb = Qwen3RotaryEmbedding(qwenConfig)
position_ids = torch.arange(input_encoding.shape[1]).reshape(1, -1).repeat(input_encoding.shape[0], 1)
cos, sin = rotary_emb(input_encoding.to(torch.float32), position_ids)
cos, sin = torch.tensor(cos.numpy()), torch.tensor(sin.numpy())
variance_epsilon = torch.tensor([[1e-6,]])
slen = input_encoding.shape[1]

# Prefill Stage

custom_model_past_cache = DynamicCache()

outputs = custom_model(
        input_encoding.reshape(input_encoding.shape[1],1), 
        cos,
        sin,
        variance_epsilon,
        custom_model_past_cache
    )

output_token = outputs[:, :, -1, :].argmax(dim=-1)

       
#Decode stage

response = []

for step in range(num_tokens_to_generate):
    
    # For decode stage, we only need position for the current token
    
    position_ids = torch.arange(slen,slen+1).reshape(1, 1)
    cos, sin = rotary_emb(output_token.to(torch.float32), position_ids)
    cos, sin = torch.tensor(cos.numpy()), torch.tensor(sin.numpy())
    cos,sin = cos[:, -1:, :], sin[:, -1:, :]
      
    outputs = custom_model(
        output_token.reshape(1,1),
        cos,
        sin,
        variance_epsilon,
        custom_model_past_cache
    )
    
    output_token = outputs[:, :, -1, :].argmax(dim=-1)
    response.append(output_token.item())
    slen += 1

decoded = tokenizer.decode(response, skip_special_tokens=True)
print("NNTrainer model response: ", decoded)