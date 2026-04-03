from transformers import AutoTokenizer
# from speechtokenizer import SpeechTokenizer
#from .config import SpeechLLMConfig

#config = SpeechLLMConfig
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

def get_text_tokenizer(model_name: str, special_tokens_dict: dict = SPECIAL_TOKENS_DICT):

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", extra_special_tokens = special_tokens_dict)

# def get_audio_tokenizer(model_path):

#     return SpeechTokenizer.load_from_checkpoint(config_path=model_path + 'config.json', ckpt_path=model_path + 'ckpt.dev')
    

if __name__ == "__main__":
    tokenizer = get_text_tokenizer("Qwen/Qwen2.5-0.5B")
    tokenizer.save_pretrained('ckpt/pretrained/tokenizer')