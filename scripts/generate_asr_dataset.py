from audiolm.tokenizer import get_text_tokenizer

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
import argparse
from functools import partial
from speechtokenizer import SpeechTokenizer
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--model_path"m type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--subset", type=str)
parser.add_argument("--split", type=int)
parser.add_argument("--audio_column", type=str)
parser.add_argument("--stride", type=int)
parser.add_argument("--device", type=str)
parser.add_argument("--save_dir", type=str)

args = parser.parse_args()

model = SpeechTokenizer.load_from_checkpoint(f"{args.model_path}/config.json", f"{args.base_path}/SpeechTokenizer.pt")
model = model.to(args.device)

tokenizer = AutoTokenizer.from_pretrained("ajdsouz/audiollm")
dataset = load_dataset(args.dataset, args.subset, args.split)
offset = len(tokenizer)




def encode_audio(example, audio_tokenizer, audio_column, offset, stride):
    def tokenize_semantic(audio, audio_tokenizer, offset, stride):
        with torch.no_grad():
            tokens = audio_tokenizer.encode(audio)

        tokens = tokens[..., ::stride]
        semantic_tokens = tokens[0, 0, :]
        offset_semantic_tokens = semantic_tokens.add(offset)

        return semantic_tokens, offset_semantic_tokens

    audio_array = torch.tensor(example[audio_column]['array'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    semantic_tokens, offset_semantic_tokens = tokenize_semantic(audio_array.to(args.device), audio_tokenizer, offset=offset, stride=stride)
    return {'semantic_tokens': semantic_tokens.tolist(), 'offset_semantic_tokens': offset_semantic_tokens.tolist()}

def make_dataset(args):
    model = SpeechTokenizer.load_from_checkpoint(f"{args.model_path}/config.json", f"{args.base_path}/SpeechTokenizer.pt")
    model = model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained("ajdsouz/audiollm")
    dataset = load_dataset(args.dataset, args.subset, args.split)
    offset = len(tokenizer)
    make_fn = partial(
        encode_audio,
        audio_tokenizer=model,
        audio_column=args.audio_column,
        offset=offset,
        stride=2
    )

    dataset = dataset.map(
        make_fn,
        batched=False,
        num_proc=1,
    )

    dataset.save_to_disk(args.save_dir)