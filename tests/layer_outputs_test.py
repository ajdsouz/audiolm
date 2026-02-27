from audiolm.config import QwenConfig
from audiolm.qwen import QwenCausalLM
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

activation = {
    "hf": {},
    "mine": {}
}


def get_hook(name, model_name):
    def hook(module, input, output):
        activation[model_name][name] = output.detach().cpu()
    return hook

cfg = QwenConfig(
    block_size=64,
    d_model=896,
    d_ffn=4864,
    n_layers=24,
    n_heads=14,
    n_kv_heads=2,
    max_positional_embed=32768,
    rmsnorm_eps=1e-06,
    rope_theta=1000000.0,
    dropout=0.0,
    vocab_size=151936,
    activation='silu',
    pad_token_id=151643,
    tie_word_embeddings=True
)

hf_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(hf_name)
hf_model = AutoModelForCausalLM.from_pretrained(hf_name)
custom_model = QwenCausalLM(cfg)
sd = torch.load("ckpt/qwen.bin")
print("loading sq into model")
missing, unexpected = custom_model.load_state_dict(sd)

prompt = "The quick brown fox"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# hook_points = [
#     "embed_tokens",
#     # "rotary_emb",
#     "input_layernorm",
#     "self_attn.q_proj",
#     "self_attn.k_proj",
#     "self_attn.v_proj",
#     "post_attention_layernorm",
#     "mlp.gate_proj",
#     "mlp.up_proj",
#     "mlp.down_proj",
#     ]

hook_points = [
    "embed_tokens",
    # "rotary_emb",
    "input_layernorm",
    "self_attn",
    "post_attention_layernorm",
    "mlp",
    ]

def attach_hooks(model: nn.Module, model_name: str):
    
    for name, module in model.named_modules():
        if any(name.endswith(hp) for hp in hook_points):
            module.register_forward_hook(get_hook(name, model_name))

attach_hooks(hf_model, 'hf')
attach_hooks(custom_model, 'mine')

with torch.no_grad():
    hf_out = hf_model(input_ids)
    mine_out = custom_model(input_ids)


for _, key2 in activation.items():
    for key in key2.keys():
        assert torch.allclose(activation['hf'][key], activation['mine'][key]), f"hf-{key} and custom-{key} dont match!"
