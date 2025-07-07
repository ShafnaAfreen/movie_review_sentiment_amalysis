# data_preprocessing.py

from datasets import load_dataset
from config import tokenizer, MAX_LENGTH, SEED
import torch

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

def get_dataset(subset_size=5000):
    dataset = load_dataset("imdb")["train"].shuffle(seed=SEED)
    subset = dataset.select(range(subset_size))

    tokenized = subset.map(tokenize_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

    return tokenized.train_test_split(test_size=0.2, seed=SEED)
