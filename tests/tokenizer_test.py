from audiolm.tokenizer import get_text_tokenizer

tok = get_text_tokenizer("Qwen/Qwen2.5-0.5B")

print(tok.all_special_tokens)
print(len(tok))

tok.push_to_hub("ajdsouza/audiollm")