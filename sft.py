from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import math
from data_utils import create_dataset
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import default_data_collator


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

train_phase = 1
device = "cuda"
train_dataset, eval_dataset = create_dataset(
    data_path,
    "2,4,4",
    train_phase,
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
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

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

# output_dir = "./output/model"
# if output_dir is not None:
#     print('saving the final model ...')

#     save_hf_format(model, tokenizer, output_dir)

