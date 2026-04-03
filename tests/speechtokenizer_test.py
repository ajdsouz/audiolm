from speechtokenizer import SpeechTokenizer

SPEECHTOKENIZER_PATH = "ckpt/pretrained/speechtokenizer"
audio_tokenizer = SpeechTokenizer.load_from_checkpoint(
    config_path=SPEECHTOKENIZER_PATH + '/config.json',
    ckpt_path=SPEECHTOKENIZER_PATH + '/SpeechTokenizer.pt'
)

print("Sample Rate: ", audio_tokenizer.sample_rate)

