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


timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_name = f"run_{timestamp}"
OUTPUT_DIR = f"/work01/koiwa/models/{run_name}"

wandb.init(project="ppo_training_sft", name=run_name)

model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)

data_files = {
    "train": "dataset/gsm8k_json/train.jsonl",
    "test": "dataset/gsm8k_json/test.jsonl",
}

ds = load_dataset("json", data_files=data_files)
split = ds["train"].train_test_split(test_size=0.015, seed=42)
train_dataset = split["train"]
valid_dataset = split["test"]
test_dataset = ds["test"]

prompt_tmpl = "Answer the next question:\n{}\n\n"


def build_train_prompt(batch):
    prompt = "Answer the next question:\n{}\n\n{}"
    prompts = [prompt.format(q, a) for q, a in zip(batch["question"], batch["answer"])]
    return {"text": prompts}


def build_eval_prompt(batch):
    prompt = "Answer the next question:\n{}"
    prompts = [prompt.format(q) for q in batch["question"]]
    return {"text": prompts}


tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=1024)
    return out


train_dataset = train_dataset.map(build_train_prompt, batched=True)
valid_dataset = valid_dataset.map(build_eval_prompt, batched=True)

train_dataset = train_dataset.map(
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
    return m.group(1).replace(",", "") if m else None


class SampleGenCallback(TrainerCallback):
    def __init__(self, tokenizer, max_new_tokens=120):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
        start = time.perf_counter()
        model = kwargs["model"]
        model.eval()

        correct = 0
        rows = []
        print(f"evaluate_start, data_len={len(valid_dataset)}")
        print(f"valid_dataset is {type(valid_dataset)}")
        for i, p in tqdm(enumerate(valid_dataset)):
            with torch.no_grad():
                inputs = {
                    "input_ids": torch.tensor([p["input_ids"]], device=model.device),
                    "attention_mask": torch.tensor([p["attention_mask"]], device=model.device),
                }
                out = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                ans = extract_answer(text)
                gold = extract_answer(p["answer"])
                if ans == gold:
                    correct += 1
            if i < 5:
                rows.append([state.global_step, i, p["text"], text, ans, gold])

        table = wandb.Table(columns=["step", "idx", "prompt", "output", "ans", "gold"])
        for r in rows:
            table.add_data(*r)
        end = time.perf_counter()

        print(f"duration: {end-start}")

        wandb.log({"samples": table})
        wandb.log({"eval/accuracy": correct/len(valid_dataset)}, step=state.global_step)

    def on_train_end(self, args, state, control, **kargs):
        url = os.getenv("SLACK_WEBHOOK_URL")
        if not url:
            return
        msg = {"text": f"train finished: step={state.global_step}, epoch={state.epoch}"}
        requests.post(url, json=msg, timeout=10)


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=1,
    # per_gpu_train_batch_size=8
    gradient_accumulation_steps=1,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=500,
    learning_rate=0.005,
    weight_decay=0,
    adam_epsilon=1e-8,
    # logging_dir="loggings",
    logging_steps=30,
    report_to="wandb",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    processing_class=tokenizer,
    callbacks=[SampleGenCallback(tokenizer)],
)

trainer.train()
