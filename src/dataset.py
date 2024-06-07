from abc import ABC, abstractmethod

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from config import config


def load_data(
    file_path: str,
    tokenizer,
    preprocess_fn=None,
):
    df = pd.read_csv(file_path)
    train_df, val_df = train_test_split(df, test_size=0.2)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH,
        )

    train_dataset = train_dataset.map(preprocess_fn, batched=True)
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    val_dataset = val_dataset.map(preprocess_fn, batched=True)
    val_dataset = val_dataset.map(preprocess_fn, batched=True)

    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    return dataset
