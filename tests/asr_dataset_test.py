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
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
print(dataset.column_names)
tokenizer = tokenizer.to('mps')
for example in dataset['validation']:
    #audio, sr = torchaudio.load(example["audio"]["path"])
    audio = torch.tensor(example['audio']['array']).to('mps')
    print(audio.shape)
    

    audio = audio.unsqueeze(0).unsqueeze(1)
    # if sr != tokenizer.sample_rate:
    #     audio = torchaudio.functional.resample(audio, sr, tokenizer.sample_rate)
    print(audio.shape)
    with torch.no_grad():
       tokens = tokenizer.encode(audio)
    break

print(tokens.shape)
print(tokens[:1, :, :])