from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

dset = load_from_disk('data/')
tokenizer = AutoTokenizer.from_pretrained('ckpt/pretrained/tokenizer')
#print(dset)
#print(dset['train'])

example = dset['train'][0]

#print(example)

dset['train'].set_format(type="torch", columns=['input_ids'])

train_dl = DataLoader(dset['train'], batch_size=1, shuffle=True)

for batch in train_dl:
    print(batch)
    print(tokenizer.decode(batch['input_ids'], skip_special_tokens=False))
    break