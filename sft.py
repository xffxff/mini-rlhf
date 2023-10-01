from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import math
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import default_data_collator

import datasets
from torch.utils.data import Dataset, Subset


def get_raw_dataset_split_index(data_split, split_index, data_size):
    """
    params:
        data_split: a string of comma separated numbers, e.g. "2,4,4",
            which means the dataset will be split into 3 parts, and the first
            part will be 2/(2+4+4) of the whole dataset, and the second part
            will be 4/(2+4+4) of the whole dataset, and so on.
        split_index: the index of the split to be returned, starting from 0, and should
            be smaller than the number of splits. This is used to determine which
            part of the split to be returned.
        data_size: the size of the whole dataset.
    return:
        a list of indices of the split to be returned.
    """
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


def create_dataset_split(dataset, tokenizer,
                         end_of_conversation_token, max_seq_len):
    chosen_dataset = []
    for i, tmp_data in enumerate(dataset):
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
                   tokenizer, end_of_conversation_token,
                   max_seq_len):
    """
    create the dataset for sft based on the data_split.
    For example, if data_split is "2,4,4", then the dataset will be split into 3 parts,
    and the first part will be 2/(2+4+4) of the whole dataset, and the second part
    will be 4/(2+4+4) of the whole dataset, and so on. For sft, we only use the first
    part of the split.
    """
    raw_dataset = datasets.load_from_disk(dataset_name)
    train_dataset = raw_dataset["train"]
    train_index = get_raw_dataset_split_index(data_split,
                                              0,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset,
                                         tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset["test"]
    eval_index = get_raw_dataset_split_index(data_split, 0,
                                             len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset


data_path = "../huggingface/datasets/Dahoas/rm-static"
model_path = "../huggingface/models/facebook/opt-350m"

tokenizer = AutoTokenizer.from_pretrained(model_path, fast_tokenizer=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(int(
    8 *
    math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

# prepare the dataset
device = "cuda"
train_dataset, eval_dataset = create_dataset(
    data_path,
    "2,4,4",
    tokenizer,
    "<|endoftext|>",
    512,
)

per_device_train_batch_size = 16
per_device_eval_batch_size = 16
train_sampler = RandomSampler(train_dataset)
eval_sampler = SequentialSampler(eval_dataset)
train_dataloader = DataLoader(train_dataset,
                                collate_fn=default_data_collator,
                                sampler=train_sampler,
                                batch_size=per_device_train_batch_size)
eval_dataloader = DataLoader(eval_dataset,
                                collate_fn=default_data_collator,
                                sampler=eval_sampler,
                                batch_size=per_device_eval_batch_size)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def evaluation(model, eval_dataloader):
    model.eval()
    losses = 0
    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses += loss.float()
    losses = losses / (step + 1)
    try:
        perplexity = torch.exp(losses)
    except OverflowError:
        perplexity = float("inf")
    return perplexity


num_train_epochs = 1
print("***** Running training *****")
print(
    f"***** Evaluating perplexity, Epoch {0}/{num_train_epochs} *****",)
perplexity = evaluation(model, eval_dataloader)
print(f"ppl: {perplexity}")

for epoch in range(num_train_epochs):
    print(
        f"Beginning of Epoch {epoch+1}/{num_train_epochs}, Total Micro Batches {len(train_dataloader)}")
    model.train()
    import time
    for step, batch in enumerate(train_dataloader):
        start = time.time()
        batch = to_device(batch, device)
        outputs = model(**batch, use_cache=False) # CasualLMOutputWithPast
        loss = outputs.loss
        loss.backward()
        print(
            f"Epoch: {epoch}, Step: {step}, loss = {loss}"
        )
        optimizer.step()
        optimizer.zero_grad()
        end = time.time()

    # Evaluate perplexity on the validation set.
    print(
        f"***** Evaluating perplexity, Epoch {epoch+1}/{num_train_epochs} *****")
    perplexity = evaluation(model, eval_dataloader)
    print(f"ppl: {perplexity}")


def save_hf_format(model, tokenizer, output_dir, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


output_dir = "./output/model"
if output_dir is not None:
    print('saving the final model ...')

    save_hf_format(model, tokenizer, output_dir)

