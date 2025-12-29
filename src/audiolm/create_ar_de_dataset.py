"""Script to download and extract the Arabic-German Tatoeba dataset."""
import gzip
import tarfile
import requests
from datasets import DatasetDict, Dataset
from datasets import load_from_disk

url = "https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/ara-deu.tar"
filename = "ara_deu.tar"

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

with tarfile.open(filename, mode="r:") as tar:
    tar.extractall(path="ara-deu")
print("Extracted.")

def load_parallel(src_path, tgt_path):
  src, tgt = [], []
  with gzip.open(src_path, "rt", encoding="utf-8") as src_file, gzip.open(tgt_path, "rt", encoding="utf-8") as tgt_file:
    for src_line, tgt_line in zip(src_file, tgt_file):
      src.append(src_line.strip())
      tgt.append(tgt_line.strip())
  return src, tgt

train_src, train_tgt = load_parallel("./ara-deu/data/release/v2023-09-26/ara-deu/train.src.gz", "./ara-deu/data/release/v2023-09-26/ara-deu/train.trg.gz")
dev_src, dev_tgt = load_parallel("./ara-deu/data/release/v2023-09-26/ara-deu/dev.src", "./ara-deu/data/release/v2023-09-26/ara-deu/dev.trg")
test_src, test_tgt = load_parallel("./ara-deu/data/release/v2023-09-26/ara-deu/test.src", "./ara-deu/data/release/v2023-09-26/ara-deu/test.trg")


train_ds = Dataset.from_dict({"src": train_src, "tgt": train_tgt})
dev_ds   = Dataset.from_dict({"src": dev_src, "tgt": dev_tgt})
test_ds  = Dataset.from_dict({"src": test_src, "tgt": test_tgt})

dataset = DatasetDict({
    "train": train_ds,
    "validation": dev_ds,
    "test": test_ds,
})

dataset.save_to_disk("ara_de_dataset_hf")
print("Dataset saved to 'ara_de_dataset_hf' directory.")

dataset = load_from_disk("ara_de_dataset_hf")
dataset.push_to_hub("username/ara_de_tatoeba")

