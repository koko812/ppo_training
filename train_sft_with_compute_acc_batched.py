from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset
from datetime import datetime
import wandb
import re
import torch
import os
import requests
import time
from tqdm import tqdm
import math
import argparse

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_name_default = f"run_{timestamp}"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--run_name", type=str, default="")
args = parser.parse_args()

run_name = args.run_name + "_" + run_name_default
wandb.init(project="ppo_training_sft", name=run_name)
OUTPUT_DIR = f"/work01/koiwa/models/{run_name}"

lr = args.lr
batch_size = args.batch_size
epoch = args.epochs
grad_accum = args.grad_accum
model_name = args.model_name
model = AutoModelForCausalLM.from_pretrained(model_name)

data_files = {
    "train": "dataset/gsm8k_json_no_bracket/train.jsonl",
    "test": "dataset/gsm8k_json_no_bracket/test.jsonl",
}

ds = load_dataset("json", data_files=data_files)
split = ds["train"].train_test_split(test_size=0.015, seed=42)
train_dataset = split["train"]
valid_dataset = split["test"]
test_dataset = ds["test"]


def build_train_prompt(batch):
    prompt = "Answer the next question:\n{}\n\n{}"
    prompts = [prompt.format(q, a) for q, a in zip(batch["question"], batch["answer"])]
    return {"text": prompts}


def build_eval_prompt(batch):
    prompt = "Answer the next question:\n{}"
    prompts = [prompt.format(q) for q in batch["question"]]
    return {"text": prompts}


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=256)
    return out


train_dataset = train_dataset.map(build_train_prompt, batched=True)
valid_dataset = valid_dataset.map(build_eval_prompt, batched=True)
eval_dataset = valid_dataset.map(build_train_prompt, batched=True)

train_dataset = train_dataset.map(
    tokenize, batched=True, remove_columns=ds["train"].column_names
)
eval_dataset = valid_dataset.map(
    tokenize, batched=True, remove_columns=ds["train"].column_names
)
valid_dataset = valid_dataset.map(tokenize, batched=True)
# valid_dataset はこんな感じになるはず {"text", "question", "answer", "input_ids", "attention_mask"}

print(f"train_dataset: {len(train_dataset)}")
print(f"valid_dataset: {len(valid_dataset)}")
print(f"test_dataset: {len(test_dataset)}")
print(valid_dataset[0])


def extract_answer(text):
    m = re.search(r"####\s*([\-0-9,\.]+)", text)
    return m.group(1).replace(",", "") if m else "-10000"


class SampleGenCallback(TrainerCallback):
    def __init__(self, tokenizer, max_new_tokens=120):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
        t0 = time.perf_counter()
        model = kwargs["model"]
        model.eval()

        correct = 0
        sentence_cnt = 0
        rows = []
        b_size = batch_size//grad_accum
        print(f"evaluate_start, data_len={len(valid_dataset)}")
        print(f"valid_dataset is {type(valid_dataset)}")
        for start in range(0, len(valid_dataset), b_size):
            with torch.no_grad():
                batch = valid_dataset[start:start+b_size]
                print(batch.keys())
                inputs = tokenizer.pad(
                    {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    },
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                out = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
                for j, (text, gold_text) in enumerate(zip(texts, batch["answer"])):
                    ans = extract_answer(text)
                    gold = extract_answer(gold_text)
                    if ans == gold:
                        correct += 1
                    if sentence_cnt < 5:
                        prompt = batch["text"][j]
                        rows.append([state.global_step, sentence_cnt, prompt, text, ans, gold])
                    sentence_cnt+=1

        table = wandb.Table(columns=["step", "idx", "prompt", "output", "ans", "gold"])
        for r in rows:
            table.add_data(*r)
        end = time.perf_counter()

        print(f"duration: {end-t0}")
        wandb.log(
            {
                "eval/accuracy": correct/len(valid_dataset),
                "samples": table,
            },
            step=state.global_step,
        )

    def on_train_end(self, args, state, control, **kargs):
        url = os.getenv("SLACK_WEBHOOK_URL")
        if not url:
            return
        run_url = wandb.run.get_url() if wandb.run else ""
        msg = {"text": f"train finished: step={state.global_step}\n{run_url}"}
        requests.post(url, json=msg, timeout=10)

total_step = int(math.ceil(len(train_dataset)/batch_size) * epoch)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    per_device_train_batch_size=batch_size//grad_accum,
    per_device_eval_batch_size=batch_size//grad_accum,
    eval_accumulation_steps=1,
    # per_gpu_train_batch_size=8
    gradient_accumulation_steps=grad_accum,
    eval_strategy="steps",
    eval_steps=total_step//10,
    save_steps=total_step//5,
    learning_rate=lr,
    weight_decay=0,
    adam_epsilon=1e-8,
    # logging_dir="loggings",
    logging_steps=10,
    report_to="wandb",
    bf16=True,
    num_train_epochs=epoch
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    processing_class=tokenizer,
    callbacks=[SampleGenCallback(tokenizer)],
)

trainer.train()