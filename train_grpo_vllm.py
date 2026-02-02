from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    GenerationConfig,
)
from trl import AutoModelForCausalLMWithValueHead
#from trl.experimental.ppo import PPOConfig, PPOTrainer
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from datetime import datetime
import wandb
import re
import torch
import torch.nn as nn
import os
import requests
import time
from tqdm import tqdm
import math
import argparse

# >> ------ meta_settings ----------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_name_default = f"run_{timestamp}"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--num_gens", type=int, default=8)
args = parser.parse_args()

run_name = args.run_name + "_" + run_name_default
wandb.init(project="ppo_training_grpo", name=run_name)

models_dir = os.getenv("MODELS_DIR", "/work01/koiwa/models")
OUTPUT_DIR = os.path.join(models_dir, run_name)

lr = args.lr
batch_size = args.batch_size
epoch = args.epochs
grad_accum = args.grad_accum
num_gens = args.num_gens
model_name = args.model_name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device = next(model.parameters()).device

# << ------ meta_settings ----------

# >> -------- dataset prepare ---------------
data_files = {
    "train": "dataset/gsm8k_json_no_bracket/train.jsonl",
    "test": "dataset/gsm8k_json_no_bracket/test.jsonl",
}

ds = load_dataset("json", data_files=data_files)
split = ds["train"].train_test_split(test_size=0.015, seed=42)
train_dataset = split["train"]
valid_dataset = split["test"]
test_dataset = ds["test"]

PROMPT_TEMPLATE = (
    "Answer the next question.\n"
    "Show your reasoning, and write the final answer in the last line as:\n"
    "#### <number>\n"
    "Question:\n"
    "{}"
)

def build_train_prompt(batch):
    prompts = [PROMPT_TEMPLATE.format(q) for q in batch["question"]]
    return {"prompt": prompts}


def build_eval_prompt(batch):
    prompts = [PROMPT_TEMPLATE.format(q) for q in batch["question"]]
    return {"prompt": prompts}


def tokenize(batch):
    out = tokenizer(batch["prompt"], truncation=True)
    return out

train_dataset = train_dataset.map(build_train_prompt, batched=True)
valid_dataset = valid_dataset.map(build_eval_prompt, batched=True)

train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)

print(f"train_dataset: {len(train_dataset)}")
print(f"valid_dataset: {len(valid_dataset)}")
# << ------- dataset prepare ---------------


# >> ------- reward func + callback ---------------
def extract_answer(text):
    m = re.search(r"####\s*([\-0-9,\.]+)", text)
    return m.group(1).replace(",", "") if m else "-10000"

answer_map = {
    PROMPT_TEMPLATE.format(q): a
    for q, a in zip(ds["train"]["question"], ds["train"]["answer"])
}

# def reward_func(prompts, completions, **kwargs):
#     rewards = []
#     for p, c in zip(prompts, completions):
#         q = p.replace("Answer the next question:\n", "").strip()
#         gold = answer_map.get(q, "")
#         rewards.append(1.0 if extract_answer(c) == extract_answer(gold) else 0.0)
#     return rewards

total_step = int(math.ceil(len(train_dataset) / batch_size) * epoch) * num_gens
LOG_EVERY = total_step // 10
LOG_N = 5

# eval時の一時バッファ
_eval_rows = []

def reward_func(prompts, completions, **kwargs):
    global _eval_rows

    #print(kwargs.keys())

    rewards = []
    trainer_state = kwargs.get("trainer_state", None)
    global_step = getattr(trainer_state, "global_step", None)
    # is_eval = bool(getattr(trainer_state, "is_evaluating", False))

    # rank0だけ
    is_main = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main = (torch.distributed.get_rank() == 0)

    do_log = (
        is_main
        #and is_eval
        #and (global_step is not None)
        and (global_step % LOG_EVERY == 0)
    )

    #print(f"global_step: {global_step},  is_main: {is_main}, do_log: {do_log}")

    for i, (p, c) in enumerate(zip(prompts, completions)):
        gold = answer_map.get(p, "")
        pred_ans = extract_answer(c)
        gold_ans = extract_answer(gold)
        r = 1.0 if pred_ans == gold_ans else 0.0
        rewards.append(r)

        if do_log and i < LOG_N:
            #print("log_append")
            _eval_rows.append(
                [global_step, i, p, c, pred_ans, gold_ans, r]
            )

    # stepの最後にまとめて送る
    if do_log and _eval_rows:
        #print("send_tables_to_wandb")
        table = wandb.Table(
            columns=["step", "idx", "prompt", "completion", "pred", "gold", "reward"]
        )
        for row in _eval_rows:
            table.add_data(*row)

        wandb.log({"eval_samples": table})
        _eval_rows = []  # クリア

    return rewards
# << ------- reward func + callback ---------------


# >> ------- make evaluate -------------
class SlackNotifyCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kargs):
        url = os.getenv("SLACK_WEBHOOK_URL")
        if not url:
            return
        run_url = wandb.run.get_url() if wandb.run else ""
        msg = {"text": f"train finished: step={state.global_step}\n{run_url}"}
        requests.post(url, json=msg, timeout=10)
# << ------- make evaluate and callbacks -------------


# >> ------- train_settings ----------

config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    do_train=True,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_model_impl="vllm",
    vllm_gpu_memory_utilization=0.20,   # OOMしたら 0.2〜0.4 で調整
    vllm_max_model_length=812,
    use_transformers_paged=False,     # HF paged attention を切る
    cache_implementation=None,        # torch.compile/inductor 経路を避ける
    torch_compile=False,              # ★重要：Inductorを切る
    torchdynamo=None,                 # ★重要：Dynamoを切る
    per_device_train_batch_size=batch_size // grad_accum,
    per_device_eval_batch_size=batch_size // grad_accum,
    num_generations=num_gens,
    eval_accumulation_steps=1,
    gradient_accumulation_steps=grad_accum,
    eval_strategy="steps",
    eval_steps=total_step // 100,
    save_steps=total_step // 5,
    learning_rate=lr,
    weight_decay=0,
    max_prompt_length=300,
    max_completion_length=512,
    adam_epsilon=1e-8,
    # logging_dir="loggings",
    logging_steps=10,
    report_to="wandb",
    bf16=True,
    num_train_epochs=epoch,
    generation_kwargs = {
        "temperature": 1.0,
        "top_p": 1.0,
        #"do_sample": True,
        #"pad_token_id": tokenizer.eos_token_id,
    }
)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_func,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,
    callbacks=[SlackNotifyCallback()],
)

# << ------- train_settings ----------

grpo_trainer.train()
grpo_trainer.save_model(OUTPUT_DIR)