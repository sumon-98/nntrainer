import torch
from transformers import AutoTokenizer, Qwen3Config, AutoModelForCausalLM
from custom_qwen3 import  NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding
from transformers.cache_utils import DynamicCache

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

####Offical Model####

config = Qwen3Config.from_pretrained(model_name,attn_implementation="eager")  
official_model = AutoModelForCausalLM.from_pretrained(model_name,config=config).eval()

# Prompt
prompt = "Tell me a dad joke about a computer: "

# messages = [
    # {"role": "user", "content": prompt}
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# )

# Tokens to generate
num_tokens_to_generate = 20

print("\nInput prompt:",prompt)
max_len = 256
# enc = tokenizer(text, return_tensors="pt")
enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
input_ids=enc.input_ids
generated = input_ids.clone()
    
# output = official_model.generate(
#     input_ids=input_ids,
#     do_sample=False,
#     max_new_tokens=num_tokens_to_generate,
#     use_cache=True,
# )

# decoded = tokenizer.decode(output[0], skip_special_tokens=True)
# print("Official model output: ",decoded,"\n")

####Custom Model####

qwenConfig = official_model.config
custom_model =  NNTrainerQwen3ForCausalLM(qwenConfig).eval()
custom_model.load_state_dict(official_model.state_dict(),strict=False)

rotary_emb = Qwen3RotaryEmbedding(qwenConfig)

generated = input_ids.clone() # input to official Qwen model
cur_len =  enc.input_ids.size(1)

position_ids = torch.arange(cur_len).unsqueeze(0)
cos, sin = rotary_emb(generated.to(torch.float32), position_ids)
variance_epsilon = torch.tensor([[1e-6,]])

causal_mask = torch.full((1, 1, generated.shape[1], generated.shape[1]), float(-3.4028e+38))
causal_mask = torch.triu(causal_mask, diagonal=1) 
past_values = [torch.rand(1,8,0,128) for _ in range(qwenConfig.num_hidden_layers)]
past_keys = [torch.rand(1,8,0,128) for _ in range(qwenConfig.num_hidden_layers)]
response = []

outputs,past_keys,past_values = custom_model(
        generated.transpose(0,1), 
        cos,
        sin,
        variance_epsilon,
        past_keys,
        past_values,
        causal_mask,
    )

print(outputs)
print(cur_len)
next_token_logits = outputs[0][:, cur_len - 1, :]
print(next_token_logits)
next_token_id = torch.argmax(next_token_logits,dim=-1)
print(next_token_id)
response.append(next_token_id.item())

for step in range(num_tokens_to_generate - 1):  
    
    position_ids = torch.arange(cur_len,cur_len+1).unsqueeze(0)
    cos, sin = rotary_emb(next_token_id.to(torch.float32), position_ids)
    print(sin)
    exit()
    
    attention_mask = torch.zeros((1,cur_len+1)).to(torch.int64)
  
    outputs,past_keys,past_values = custom_model(
        next_token_id.unsqueeze(1), 
        cos,
        sin,
        variance_epsilon,
        past_keys,
        past_values,
        attention_mask,
    )

    next_token_logits = outputs[0][:,-1, :]
    next_token_id = torch.argmax(next_token_logits,dim=-1)
    if next_token_id == tokenizer.eos_token_id:
        break
    response.append(next_token_id.item())
    cur_len+=1

decoded = tokenizer.decode(response, skip_special_tokens=True)

print("\nCustom Model: ",decoded)