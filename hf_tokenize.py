from transformers import GPT2TokenizerFast
from datasets import load_dataset
import os


tokenizer = GPT2TokenizerFast.from_pretrained("esenergun/wikitext_tokenizer")
tokenizer.pad_token = tokenizer.eos_token
def tokenize(examples):
    outputs = tokenizer(examples['text'], truncation = False)
    return outputs


raw_datasets = load_dataset("wikitext","wikitext-103-raw-v1")

tokenized_datasets = {split:raw_datasets[split].map(tokenize, batched = True, remove_columns = raw_datasets[split].column_names) for split in raw_datasets.keys()}


os.makedirs("../datasets", exist_ok=True)

for split in ['train', 'validation','test']:

    with open('../datasets/wikitext_tokenized_' + split +'.txt', 'w') as file: