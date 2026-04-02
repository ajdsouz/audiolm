from transformers import AutoTokenizer
from .config import SpeechLLMConfig

config = SpeechLLMConfig
# SEMANTIC_TOKENS = [f"semantic_{idx}" for idx in range(config.semantic_vocab_size)]

SPECIAL_TOKENS_DICT = {
    "input_token": "<|INPUT|>",
    "output_token": "<|OUTPUT|>",
    "text_bos_token": "<|text_start|>",
    "text_eos_token": "<|text_end|>",
    "audio_bos_token": "<|audio_start|>",
    "audio_eos_token": "<|audio_end|>",
    "pad_token": "<|pad_token|>"
}

def get_tokenizer(model_name: str, special_tokens_dict: dict = SPECIAL_TOKENS_DICT):

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", extra_special_tokens = special_tokens_dict)
