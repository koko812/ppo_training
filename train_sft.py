from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset
from datetime import datetime
import wandb
import re
import torch
import os
import requests

class SampleGenCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, max_new_tokens=120):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens

    # def on_evaluate(self, args, state, control, **kwargs):
    #     print("callback", state.global_step)
    #     #wandb.log("callback", state.global_step)
    #     model = kwargs["model"]
    #     model.eval()
    #     texts = []
    #     for p in self.prompts:
    #         inputs = self.tokenizer(p, return_tensors="pt").to(model.device)
    #         with torch.no_grad():
    #             # これはなんのデータセットに対して検証してる？？
    #             out = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
    #         texts.append(self.tokenizer.decode(out[0], skip_special_tokens=True))
    #     table = wandb.Table(columns=["idx", "text"])
    #     for i, t in enumerate(["foo", "bar"]):
    #         table.add_data(i, t)

    #     wandb.log({f"sample_{i}": t for i, t in enumerate(texts)})

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        rows = []
        for i,p in enumerate(self.prompts):
            inputs = self.tokenizer(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            rows.append([state.global_step, i, p, text])

        table = wandb.Table(columns=["step", "idx", "prompt", "output"])
        for r in rows:
            table.add_data(*r)

        wandb.log({"samples": table})

    def on_train_end(self, args, state, control, **kargs):
        url = os.getenv("SLACK_WEBHOOK_URL")
        if not url:
            return        
        msg = {"text": f"train finished: step={state.global_step}, epoch={state.epoch}"}
        requests.post(url, json=msg, timeout=10)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_name = f"run_{timestamp}"
OUTPUT_DIR = f"/work01/koiwa/models/{run_name}"

wandb.init(project="ppo_training_sft", name=run_name)

model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)

#dataset_name = "GSM8K"
#train_dataset = datasets.load_dataset(dataset_name)["train"]

data_files = {
    "train": "dataset/gsm8k_json/train.jsonl",
    "test": "dataset/gsm8k_json/test.jsonl"
}

ds = load_dataset("json", data_files=data_files)
train_dataset = ds["train"]
test_dataset = ds["test"]

prompt_tmpl = "Answer the next question:\n{}\n\n"
prompts = [prompt_tmpl.format(q) for q in train_dataset["question"][:5]]

print(prompts)

print(type(train_dataset))
print(train_dataset[1].keys())

def build_prompt(batch):
    #prompts = [prompt.format(q) for q in batch["question"]]
    prompt = "Answer the next question:\n{}\n\n{}"
    prompts = [prompt.format(q,a) for q,a in zip(batch["question"], batch["answer"])]
    return {"text": prompts}

ds = ds.map(build_prompt, batched=True, remove_columns=ds["train"].column_names)

print(ds["train"][0].keys())
print(ds["test"][0].keys())
print(str(ds["train"][0]["text"]))

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer.pad_token)
print(tokenizer.eos_token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    out = tokenizer(batch["text"], 
                    #padding="longest",
                    truncation=True,
                    max_length=1024)
    #out["labels"] = out["input_ids"].copy()
    return out

ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
split = ds["train"].train_test_split(test_size=0.015, seed=42)
train_dataset = split["train"]
valid_dataset = split["test"]
test_dataset = ds["test"]

print(ds["train"][0])
print(ds["train"][0].keys())
print(len(ds["train"][0]["input_ids"]))
print(len(ds["train"][0]["attention_mask"]))
#print(len(ds["train"][0]["labels"]))

def extract_answer(text):
    m = re.search(r"####\s*([\-0-9,\.]+)", text)
    return m.group(1).replace(",","") if m else None

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred_ids = logits.argmax(-1)

    labels = labels.copy()
    labels[labels == -100] = tokenizer.pad_token_id

    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_ans = [extract_answer(t) for t in pred_texts]
    gold_ans = [extract_answer(t) for t in label_texts]

    correct = sum(p == g and p is not None for p,g in zip(pred_ans, gold_ans))
    acc = correct / len(gold_ans)
    return {"accuracy": acc}


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
    #logging_dir="loggings",
    logging_steps=30,
    report_to="wandb",
)

print("train_start")
table = wandb.Table(columns=["idx", "text"])
for i, t in enumerate(["foo", "bar"]):
    table.add_data(i, t)

wandb.log({"samples_debug": table})

trainer = Trainer(model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                #compute_metrics=compute_metrics,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                processing_class=tokenizer,
                callbacks=[SampleGenCallback(tokenizer, prompts)]) 

trainer.train()