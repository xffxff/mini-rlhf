from transformers import AutoTokenizer
import sys
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
    splits = [float(s) for s in data_split.split(",")]
    splits_sum = sum(splits)
    splits = [split / splits_sum for split in splits]

    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(data_size))))

    res = []
    for i in range(data_size):
        if i >= splits_index[split_index] and i < splits_index[split_index + 1]:
            res.append(i)
    return res


class PromptDataset(Dataset):
    def __init__(self, prompt_dataset) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset

    def __len__(self):
        length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        return (
            self.prompt_dataset[idx]["input_ids"],
            self.prompt_dataset[idx]["attention_mask"],
            self.pad_token_id,
        )


def create_dataset_split(dataset, tokenizer, end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    for i, tmp_data in enumerate(dataset):
        # tokenize the text
        prompt = tmp_data["prompt"]
        if prompt is not None:
            prompt_token = tokenizer(prompt, return_tensors="pt")
            prompt_token["input_ids"] = prompt_token["input_ids"]
            prompt_token["attention_mask"] = prompt_token["attention_mask"]
            for key_word in ["input_ids", "attention_mask"]:
                length = prompt_token[key_word].size()[-1]
                if length > max_seq_len:
                    y = (
                        prompt_token[key_word]
                        .squeeze(0)[length - (max_seq_len - 1) :]
                        .flip(0)
                    )
                else:
                    y = prompt_token[key_word].squeeze(0).flip(0)
                prompt_token[key_word] = y
            prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset)


def create_dataset(
    dataset_name, data_split, tokenizer, end_of_conversation_token, max_seq_len
):
    raw_dataset = datasets.load_from_disk(dataset_name)
    train_dataset = raw_dataset["train"]
    train_index = get_raw_dataset_split_index(data_split, 2, len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(
        train_dataset, tokenizer, end_of_conversation_token, max_seq_len
    )

    eval_dataset = raw_dataset["test"]
    eval_index = get_raw_dataset_split_index(data_split, 2, len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(
        eval_dataset, tokenizer, end_of_conversation_token, max_seq_len
    )
    return train_dataset, eval_dataset


assert len(sys.argv) == 3, "Please provide the model path and data path."

model_path = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_path, fast_tokenizer=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################### prepare dataset ####################
data_path = sys.argv[2]

# `data_split` is a string of comma separated numbers, e.g. "2,4,4",
# which means the dataset will be split into 3 parts, and the first
# part will be 2/(2+4+4) of the whole dataset, and the second part
# will be 4/(2+4+4) of the whole dataset, and so on.
# For sft, we only use the first part of the split.
data_split = "2,4,4"
end_of_conversation_token = "<|endoftext|>"
max_seq_len = 512
train_dataset, eval_dataset = create_dataset(
    data_path, data_split, tokenizer, end_of_conversation_token, max_seq_len
)
