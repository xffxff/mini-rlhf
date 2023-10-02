from transformers import AutoTokenizer, AutoConfig, AutoModel
import math
import sys
import torch
from torch import nn
from torch.utils.data import Dataset, Subset
import datasets


## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):
    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim, 1, bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = (
                self.config.hidden_size
                if hasattr(self.config, "hidden_size")
                else self.config.n_embd
            )
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=False,
    ):
        loss = None

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = (
                c_inds[self.num_padding_at_beginning].item()
                if len(c_inds) > self.num_padding_at_beginning
                else seq_len
            )  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = (
                    r_inds[self.num_padding_at_beginning].item()
                    if len(r_inds) > self.num_padding_at_beginning
                    else seq_len
                )
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1]
            )  # use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.nn.functional.logsigmoid(
                c_truncated_reward - r_truncated_reward
            ).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        return_value_only=False,
        prompt_length=0,
        use_cache=False,
    ):
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert (
                prompt_length > 1
            ), "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = (
                []
            )  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }


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
    def __init__(self, chosen_dataset, reject_dataset) -> None:
        super().__init__()
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset

    def __len__(self):
        length = len(self.chosen_dataset)
        return length

    def __getitem__(self, idx):
        return (
            self.chosen_dataset[idx]["input_ids"],
            self.chosen_dataset[idx]["attention_mask"],
            self.reject_dataset[idx]["input_ids"],
            self.reject_dataset[idx]["attention_mask"],
        )


def create_dataset_split(dataset, tokenizer, end_of_conversation_token, max_seq_len):
    chosen_dataset = []
    reject_dataset = []
    for i, tmp_data in enumerate(dataset):
        # tokenize the text
        chosen_sentence = tmp_data["prompt"] + tmp_data["chosen"]  # the accept response
        reject_sentence = (
            tmp_data["prompt"] + tmp_data["rejected"]
        )  # the accept response
        if chosen_sentence is not None and reject_sentence is not None:
            chosen_sentence += end_of_conversation_token  # the accept response
            reject_sentence += end_of_conversation_token
            chosen_token = tokenizer(
                chosen_sentence,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            reject_token = tokenizer(
                reject_sentence,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            chosen_token["input_ids"] = chosen_token["input_ids"]
            chosen_token["attention_mask"] = chosen_token["attention_mask"]
            chosen_dataset.append(chosen_token)

            reject_token["input_ids"] = reject_token["input_ids"]
            reject_token["attention_mask"] = reject_token["attention_mask"]
            reject_dataset.append(reject_token)

    return PromptDataset(chosen_dataset, reject_dataset)


def create_dataset(
    dataset_name, data_split, tokenizer, end_of_conversation_token, max_seq_len
):
    raw_dataset = datasets.load_from_disk(dataset_name)
    train_dataset = raw_dataset["train"]
    train_index = get_raw_dataset_split_index(data_split, 1, len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(
        train_dataset, tokenizer, end_of_conversation_token, max_seq_len
    )

    eval_dataset = raw_dataset["test"]
    eval_index = get_raw_dataset_split_index(data_split, 1, len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(
        eval_dataset, tokenizer, end_of_conversation_token, max_seq_len
    )
    return train_dataset, eval_dataset


model_path = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_path, fast_tokenizer=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################### prepare model ####################
model_config = AutoConfig.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, config=model_config)
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(
    int(8 * math.ceil(len(tokenizer) / 8.0))
)  # make the vocab size multiple of 8

reward_model = RewardModel(model, tokenizer, num_padding_at_beginning=1)

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
