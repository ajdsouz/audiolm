import math
import torch
from torch import Tensor 
import torch.nn.functional as F 
from contextlib import nullcontext
from sacrebleu import BLEU
from transformers import PreTrainedTokenizer
from evaluate import load




def attention(
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        scale: float | None = None, 
        dropout: float = 0.0,
        mask : Tensor | None = None
) -> tuple[Tensor, Tensor]:
    

    if scale is None:
        scale = math.sqrt(key.size(-1))

    scores = (query @ key.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_weights = F.dropout(attention_weights, p=dropout)
    attention = attention_weights @ value 

    return attention, attention_weights


def split_heads(x: Tensor, n_heads: int) -> Tensor:
    """Split tensor into n_heads tensors for individual attention heads

    Args:
        x (Tensor): tensor to be split
        n_heads (int): number of heads

    Returns:
        Tensor: A different view of the tensor of form [Batch n_heads Sequence d_head]
    """
    # [batch sequence d_model]
    B, S, D = x.shape
    d_head = D // n_heads
    # [batch sequence n_heads d_head] => [batch n_heads sequence d_head]
    return x.view(B, S, n_heads, d_head).transpose(1, 2)

def merge_heads(x: Tensor) -> Tensor:
    """Merge tensor after after attention calculation

    Args:
        x (Tensor): Tensor [Batch n_heads Sequence d_head] to be merged

    Returns:
        Tensor: Merged tensor of shape [Batch Sequence d_model]
    """
    B, Nh, S, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, Nh * Dh)
 
def rotate_half(x:Tensor) -> Tensor:
    """ Rotate the tensor by 90° on last dim

    Args:
        x (Tensor): tensor to be rotated

    Returns:
        Tensor: rotated tensor
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(
        q: Tensor, 
        k: Tensor, 
        cos: Tensor, 
        sin: Tensor, 
        unsqueeze_dim: int = 1
) -> tuple[Tensor, Tensor]:
    """Apply RoPE

    Args:
        q (Tensor): query tensor
        k (Tensor): key tensor
        cos (Tensor): cosine part of rope
        sin (Tensor): sine part of rope
        unsqueeze_dim (int, optional): unsqueeze sine and cosine tensors across specific dimension. Defaults to 1.

    Returns:
        tuple[Tensor, Tensor]: query and key embeddings with rope
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
 
def maybe_autocast(
    device_type: str,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()
 
 
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def compute_bleu(predicted: torch.Tensor, ground_truth: torch.Tensor, tokenizer: PreTrainedTokenizer) -> float:
    """
    Compute BLEU score on translation validation set.
    Returns:
        float: BLEU score
    """
    predictions = tokenizer.batch_decode(predicted, skip_special_tokens=True)
    references = tokenizer.batch_decode(ground_truth, skip_special_tokens=True)

    score = BLEU().corpus_score(predictions, [references])
    return score.score

def compute_wer(predicted: torch.Tensor, ground_truth: torch.Tensor, tokenizer: PreTrainedTokenizer)    -> float:
    """
    Compute WER score on translation validation set.
    Returns:
        float: WER score
    """
    wer_metric = load("wer")

    predictions = tokenizer.batch_decode(predicted, skip_special_tokens=True)
    references = tokenizer.batch_decode(ground_truth, skip_special_tokens=True)

    score = wer_metric.compute(predictions=predictions, references=references)

    return score
