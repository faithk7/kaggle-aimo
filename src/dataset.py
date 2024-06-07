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
    df = _preprocess_df(df)

    train_df, val_df = train_test_split(df, test_size=0.2)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["problem"],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH,
        )
        tokenized_inputs["labels"] = examples["answer"]
        return tokenized_inputs

    train_dataset = train_dataset.map(preprocess_fn, batched=True)
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    val_dataset = val_dataset.map(preprocess_fn, batched=True)
    val_dataset = val_dataset.map(preprocess_fn, batched=True)

    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    return dataset


def _preprocess_df(df):
    df["answer"] = df["answer"].astype(int)
    return df
