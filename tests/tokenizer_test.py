from audiolm.tokenizer import get_tokenizer

tok = get_tokenizer("Qwen/Qwen2.5-0.5B")

print(tok.all_special_tokens)