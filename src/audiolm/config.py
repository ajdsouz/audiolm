from dataclasses import dataclass

@dataclass
class QwenConfig:
    block_size: int
    d_model: int
    d_ffn: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    max_positional_embed: int
    rmsnorm_eps: float
    rope_theta: float
    dropout: float
    vocab_size: int
    activation: str
    pad_token_id: int
    tie_word_embeddings: bool