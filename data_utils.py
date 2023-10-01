# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import datasets
from torch.utils.data import Dataset, Subset


def get_raw_dataset_split_index(data_split, split_index, data_size):
    splits = [float(s) for s in data_split.split(',')]
    splits_sum = sum(splits)
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(data_size))))
    res = []
    for i in range(data_size):
        if i >= splits_index[split_index] and i < splits_index[split_index +
                                                               1]:
            res.append(i)
    return res


class PromptDataset(Dataset):

    def __init__(self, chosen_dataset) -> None:
        super().__init__()
        self.chosen_dataset = chosen_dataset

    def __len__(self):
        length = len(self.chosen_dataset)
        return length

    def __getitem__(self, idx):
        return {
            "input_ids": self.chosen_dataset[idx]["input_ids"],
            "attention_mask": self.chosen_dataset[idx]["attention_mask"],
            "labels": self.chosen_dataset[idx]["input_ids"]
        }
       


def create_dataset_split(current_dataset, raw_dataset, tokenizer,
                         end_of_conversation_token, max_seq_len):
    chosen_dataset = []
    for i, tmp_data in enumerate(current_dataset):
        # tokenize the text
        chosen_sentence = tmp_data["prompt"] + tmp_data["chosen"]  # the accept response
        if chosen_sentence is not None:
            chosen_sentence += end_of_conversation_token
            chosen_token = tokenizer(chosen_sentence,
                                        max_length=max_seq_len,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")
            chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                0)
            chosen_token["attention_mask"] = chosen_token[
                "attention_mask"].squeeze(0)
            chosen_dataset.append(chosen_token)

    return PromptDataset(chosen_dataset)


def create_dataset(dataset_name, data_split,
                   train_phase, tokenizer, end_of_conversation_token,
                   max_seq_len):

    raw_dataset = datasets.load_from_disk(dataset_name)
    train_dataset = raw_dataset["train"]
    train_index = get_raw_dataset_split_index(data_split,
                                              train_phase - 1,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset["test"]
    eval_index = get_raw_dataset_split_index(data_split, train_phase - 1,
                                             len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset
