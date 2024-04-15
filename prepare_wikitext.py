from datasets import load_dataset
import os

raw_datasets = load_dataset("wikitext","wikitext-103-raw-v1")

os.makedirs("../datasets", exist_ok=True)

for split in ['train', 'validation','test']:

    with open('../datasets/wikitext_' + split +'.txt', 'w') as file:
        file.write(''.join(raw_datasets[split]['text']))

