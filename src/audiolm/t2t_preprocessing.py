"""Preprocessing pipeline for translation datasets used in Text-to-Text tasks."""

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import argparse

# Reduce dataset to necessary columns
def reduce_columns(dataset, columns_to_keep):
    '''Reduce dataset to only specified columns.'''
    all_columns = dataset.column_names
    columns_to_remove = set(all_columns) - set(columns_to_keep)

    return dataset.remove_columns(columns_to_remove)


def translation_direction(src_lang, tgt_lang):
    """Add explicit translation direction prefix."""
    prefix = f"<{src_lang}2{tgt_lang}> "
    return prefix


def preprocess_function(batch, tokenizer, max_length, src_lang, tgt_lang):
    """Tokenize source and target text."""
    inputs = [translation_direction(src_lang, tgt_lang) + example for example in batch[src_lang]]
    targets = [example for example in batch[tgt_lang]]
    model_inputs = tokenizer(inputs, text_target =targets, max_length=max_length, truncation=True)
    return model_inputs


def main(language):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    if language in ["arabic_english", "all"]:
        # Load English translation datasets
        arabic_english_1 = load_dataset("fr3on/egyptian-dialogue")
        arabic_english_1 = reduce_columns(arabic_english_1, ["arabic", "english"])
        arabic_english_1 = arabic_english_1.rename_column("arabic", "ar")
        arabic_english_1 = arabic_english_1.rename_column("english", "en")

        arabic_english_1 = arabic_english_1.map(lambda batch: preprocess_function(tokenizer, max_length=128, src_lang="ar", tgt_lang="en"), batched=True)

        arabic_english_2 = load_dataset("ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts")
        arabic_english_2 = arabic_english_2.map(lambda batch: preprocess_function(tokenizer, max_length=128, src_lang="ar", tgt_lang="en"), batched=True)

        arabic_english_3 = load_dataset("ImruQays/Thaqalayn-Classical-Arabic-English-Parallel-texts")
        arabic_english_3 = arabic_english_3.map(lambda batch: preprocess_function(tokenizer, max_length=128, src_lang="ar", tgt_lang="en"), batched=True)

        arabic_english_4 = load_dataset("majedk01/english-arabic-text")
        arabic_english_4 = arabic_english_4.map(lambda batch: tokenizer(
        [f"<ar2en>" + example["ar"] for example in batch["translation"]],
        text_target=[example["en"] for example in batch["translation"]],
        max_length=128,
        truncation=True
    ),
    batched=True)

        arabic_english_5 = load_dataset("salehalmansour/english-to-arabic-translate")
        arabic_english_5 = arabic_english_5.map(lambda batch: preprocess_function(tokenizer, max_length=128, src_lang="en", tgt_lang="ar"), batched=True)

        concatenated_arabic_english = concatenate_datasets([arabic_english_1, arabic_english_2, arabic_english_3, arabic_english_4, arabic_english_5])

        # save dataset
        concatenated_arabic_english.save_to_disk("./preprocessed/arabic_english")


    if language in ["german_english", "all"]:
        german_english_1 = load_dataset("Darth-Vaderr/English-German")
        german_english_1 = german_english_1.rename_column("English", "en")
        german_english_1 = german_english_1.rename_column("German", "de")
        german_english_1 = german_english_1.map(preprocess_function(tokenizer, max_length=128, src_lang="en", tgt_lang="de"), batched=True)

        german_english_2 = load_dataset("nvidia/granary", "de", split="ast")
        german_english_2 = german_english_2.map(preprocess_function(tokenizer, max_length=128, src_lang="en", tgt_lang="de"), batched=True)

        concatenated_german_english = concatenate_datasets([german_english_1, german_english_2])

        # save dataset
        concatenated_german_english.save_to_disk("./preprocessed/german_english")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio datasets.")
    parser.add_argument("--language", type=str, choices=[ "german_english", "arabic_english", "german_arabic","all"],
                        help="Choose one more more translation pairs.")
    args = parser.parse_args()
    main(args.language)

