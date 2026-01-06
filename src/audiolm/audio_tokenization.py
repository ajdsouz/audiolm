"""
Audio tokenization using a pretrained encodec model.
Extracts discrete audio tokens.
"""

from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
librispeech = load_dataset("hf-internal-testing/librispeech_asr", "clean", split="validation")

model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
librispeech = librispeech.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech[-1]["audio"]["array"]
inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
# `encoder_outputs.audio_codes` contains discrete codes
audio_values = model.decode(**encoder_outputs, padding_mask=inputs["padding_mask"])[0]
# or the equivalent with a forward pass
audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values


