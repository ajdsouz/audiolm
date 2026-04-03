from datasets import load_dataset, Audio
import torchaudio
import torch

from speechtokenizer import SpeechTokenizer

SPEECHTOKENIZER_PATH = "ckpt/pretrained/speechtokenizer"
tokenizer = SpeechTokenizer.load_from_checkpoint(
    config_path=SPEECHTOKENIZER_PATH + '/config.json',
    ckpt_path=SPEECHTOKENIZER_PATH + '/SpeechTokenizer.pt'
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy")
#dataset = dataset['validation'].cast_column("audio", Audio(sampling_rate=16000))
# print(dataset.column_names)

for example in dataset['validation']:
    audio, sr = torchaudio.load(example["audio"]["path"])

    if sr != tokenizer.sample_rate:
        audio = torchaudio.functional.resample(audio, sr, tokenizer.sample_rate)

    with torch.no_grad():
        tokens = tokenizer.encode(audio)
    break

print(tokens.shape)
print(tokens[:1, :, :])