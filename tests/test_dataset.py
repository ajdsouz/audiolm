from datasets import load_from_disk

dset = load_from_disk('data/')

print(dset)
print(dset['train'])

example = dset['train'][0]

print(example)