from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel
import torch
import math
import sys
import datasets
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from rm import RewardModel


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
    def __init__(self, prompt_dataset, pad_token_id) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.pad_token_id = pad_token_id

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
    return PromptDataset(prompt_dataset, tokenizer.pad_token_id)


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
    return train_dataset


class DataCollatorRLHF:
    def __init__(self, max_token_len):
        self.max_token_len = max_token_len

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence(
            [f[0] for f in data], padding_value=pad_token_id, batch_first=True
        )
        prompt_mask = pad_sequence(
            [f[1] for f in data], padding_value=0, batch_first=True
        )

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(
                prompt, pad=(0, pad_length), mode="constant", value=pad_token_id
            )
            batch["prompt_att_mask"] = F.pad(
                prompt_mask, pad=(0, pad_length), mode="constant", value=0
            )
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


actor_model_name_or_path = sys.argv[1]
critic_model_name_or_path = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(actor_model_name_or_path, fast_tokenizer=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################### prepare dataset ####################
data_path = sys.argv[3]

# `data_split` is a string of comma separated numbers, e.g. "2,4,4",
# which means the dataset will be split into 3 parts, and the first
# part will be 2/(2+4+4) of the whole dataset, and the second part
# will be 4/(2+4+4) of the whole dataset, and so on.
# For sft, we only use the first part of the split.
data_split = "2,4,4"
end_of_conversation_token = "<|endoftext|>"
max_seq_len = 256
train_dataset = create_dataset(
    data_path, data_split, tokenizer, end_of_conversation_token, max_seq_len
)

data_collector = DataCollatorRLHF(max_seq_len)
train_sampler = RandomSampler(train_dataset)
per_device_generation_batch_size = 16
train_dataloader = DataLoader(
    train_dataset,
    collate_fn=data_collector,
    sampler=train_sampler,
    batch_size=per_device_generation_batch_size,
)


#################### prepare model ####################
def create_hf_model(model_class, model_name_or_path, tokenizer):
    model_config = AutoConfig.from_pretrained(actor_model_name_or_path)
    model = model_class.from_pretrained(actor_model_name_or_path, config=model_config)
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )  # make the vocab size multiple of 8
    return model


num_padding_at_beginning = 1
actor_learning_rate = 1e-5
critic_learning_rate = 1e-5
device = "cuda"

actor_model = create_hf_model(AutoModelForCausalLM, actor_model_name_or_path, tokenizer)
actor_optimizer = torch.optim.AdamW(actor_model.parameters(), lr=actor_learning_rate)
actor_model.optimizer = actor_optimizer
actor_model.to(device)

ref_model = create_hf_model(AutoModelForCausalLM, actor_model_name_or_path, tokenizer)
ref_model.to(device)

critic_model = create_hf_model(AutoModel, critic_model_name_or_path, tokenizer)
critic_model = RewardModel(critic_model, tokenizer, num_padding_at_beginning)
critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=critic_learning_rate)
critic_model.optimizer = critic_optimizer
critic_model.to(device)

reward_model = create_hf_model(AutoModel, critic_model_name_or_path, tokenizer)
reward_model = RewardModel(reward_model, tokenizer, num_padding_at_beginning)
reward_model.to(device)


#################### training loop ####################


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class PPOTrainer:
    def __init__(
        self,
        actor_model,
        ref_model,
        critic_model,
        reward_model,
        tokenizer,
        max_answer_seq_len,
    ):
        self.actor_model = actor_model
        self.ref_model = ref_model
        self.critic_model = critic_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.max_answer_seq_len = max_answer_seq_len

        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

    def eval(self):
        self.actor_model.eval()
        self.ref_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()

    def train(self):
        self.actor_model.train()
        self.ref_model.train()
        self.critic_model.train()
        self.reward_model.train()

    def _generate_sequence(self, prompts, mask):
        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        with torch.no_grad():
            seq = self.actor_model.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i : i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq

    def generate_experience(self, prompts, mask):
        self.eval()
        seq = self._generate_sequence(prompts, mask)
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            reward_score = self.reward_model.forward_value(
                seq, attention_mask, prompt_length=self.prompt_length
            )["chosen_end_scores"].detach()
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True
            ).detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits

        return {
            "prompts": prompts,
            "logprobs": gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            "ref_logprobs": gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]),
            "value": values,
            "rewards": reward_score,
            "input_ids": seq,
            "attention_mask": attention_mask,
        }

    def compute_rewards(
        self, prompts, log_probs, ref_log_probs, reward_score, action_mask
    ):
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(
            reward_score, -self.clip_reward_value, self.clip_reward_value
        )
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start : ends[j]][-1] += reward_clip[j]

        return rewards

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.cliprange, 1.0 + self.cliprange
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs["prompts"]
        log_probs = inputs["logprobs"]
        ref_log_probs = inputs["ref_logprobs"]
        reward_score = inputs["rewards"]
        values = inputs["value"]
        attention_mask = inputs["attention_mask"]
        seq = inputs["input_ids"]

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(
                prompts, log_probs, ref_log_probs, reward_score, action_mask
            )
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i] :] = 0
                old_values[i, ends[i] :] = 0
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start
            )

        ### process the new outputs
        batch = {"input_ids": seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(
            actor_log_prob[:, start:],
            log_probs[:, start:],
            advantages,
            action_mask[:, start:],
        )
        actor_loss.backward()

        self.actor_model.optimizer.step()
        self.actor_model.optimizer.zero_grad()

        value = self.critic_model.forward_value(
            **batch, return_value_only=True, use_cache=False
        )[:, :-1]
        critic_loss = self.critic_loss_fn(
            value[:, start:], old_values[:, start:], returns, action_mask[:, start:]
        )
        critic_loss.backward()
        self.critic_model.optimizer.step()
        self.critic_model.optimizer.zero_grad()

        return actor_loss, critic_loss


num_train_epochs = 1
max_answer_seq_len = 256
trainer = PPOTrainer(
    actor_model, ref_model, critic_model, reward_model, tokenizer, max_answer_seq_len
)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


for epoch in range(num_train_epochs):
    for step, batch_prompt in enumerate(train_dataloader):
        batch_prompt = to_device(batch_prompt, device)
        out = trainer.generate_experience(
            batch_prompt["prompt"], batch_prompt["prompt_att_mask"]
        )

        actor_loss, critic_loss = trainer.train_rlhf(out)
        average_reward = out["rewards"].mean()
        print(
            f"actor_loss: {actor_loss}, critic_loss: {critic_loss}, reward: {average_reward}"
        )
