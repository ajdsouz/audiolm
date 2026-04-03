from audiolm.tokenizer import get_text_tokenizer

from datasets import load_dataset, Dataset, concatenate_datasets
import argparse
from functools import partial
from transformers import AutoTokenizer, PreTrainedTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
# parser.add_argument("--subset", type=str)
parser.add_argument("--split", type=float)
parser.add_argument("--flip_ratio", type=float)
parser.add_argument("--num_proc", type=int)
parser.add_argument("--src_column", type=str)
parser.add_argument("--tgt_column", type=str)
parser.add_argument("--train_test_ratio", type=float)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--data_dir", type=str)

args = parser.parse_args()


def preprocess_dataset(
        dataset: Dataset, 
        tokenizer: PreTrainedTokenizer, 
        task: str,
        src_column: str,
        tgt_column: str,
        num_proc: int,
        flip=False,
    ) -> Dataset:
    
    def apply_template( example: dict, task: str, src: str, tgt: str, tokenizer: PreTrainedTokenizer) -> dict:
        """
        A general template for multi-task formatting.

        <|INPUT|>[MT en de]<|text_start|>Resumption of the session<|text_end|><|OUTPUT|><|text_start|>Wiederaufnahme der Sitzungsperiode<|text_end|><|endoftext|>
           |       |  |  |                    |                                                                          |                              |
         input   task src tgt              src sentnece                                                              tgt sentence                     eos_token
        token

        Args:
            task (str): Task abreviation
            src (str): source language
            tgt (str): target language
            example (DatasetDict): a row from the dataset to be mapped
            tokenizer (AutoTokenizer): the tokenizer

        Returns:
            dict: text formatted to the template
        """
        text = f"<|INPUT|>[{task} {src} {tgt}]<|text_start|>{example[src]}<|text_end|><|OUTPUT|><|text_start|>{example[tgt]}<|text_end|><|endoftext|>"
        tokenized = tokenizer(text)
        return {
            'input_ids': tokenized['input_ids']
        }
    
    if flip:
        src, tgt = tgt_column, src_column
    else:
        src, tgt = src_column, tgt_column

    map_fn = partial(
        apply_template,
        task=task,
        src=src,
        tgt=tgt,
        tokenizer=tokenizer,
    )
    
    dataset = dataset.map(
        map_fn,
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=num_proc
    )

    return dataset


def create_datasets(args: argparse.Namespace):
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer = get_text_tokenizer(args.tokenizer)
    dataset = load_dataset(args.dataset_name)
    print(f"using {args.tokenizer} on {args.dataset_name}")
    dataset = dataset['train'].select(range(int(args.split * len(dataset['train']))))
    flip_split = int(args.flip_ratio * len(dataset))
    indices = list(range(len(dataset)))
    flip = dataset.select(indices[:flip_split])
    no_flip = dataset.select(indices[flip_split:])

    flipped_dataset = preprocess_dataset(
        flip,
        tokenizer,
        task = 'MT',
        src_column=args.src_column,
        tgt_column=args.tgt_column,
        num_proc=args.num_proc,
        flip=True
    )

    unflipped_dataset = preprocess_dataset(
        no_flip,
        tokenizer,
        task = 'MT',
        src_column=args.src_column,
        tgt_column=args.tgt_column,
        num_proc=args.num_proc,
        flip=False
    )

    generated_dataset = concatenate_datasets(
        [flipped_dataset, unflipped_dataset]
    ).shuffle(seed=1337)

    split_dataset = generated_dataset.train_test_split(test_size=args.train_test_ratio, seed=1337)
    split_dataset['validation'] = split_dataset.pop('test')
    
    split_dataset.save_to_disk(args.data_dir)


create_datasets(args=args)
