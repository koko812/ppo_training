# これまでの学習・実験メモまとめ（PPO/GRPO/SFT）

## 全体像（何をやってきたか）
- GSM8K を使って SFT → PPO（検討）→ GRPO（検討）までの流れを試行。
- `transformers` の `Trainer` と `trl` の `SFTTrainer`/`PPOTrainer`/`GRPOTrainer` の違いを整理。
- 生成評価（accuracy）を W&B に可視化しつつ、OOM 回避やログの出し方を調整。

---

## データ準備（GSM8K）
- HF からの正しい読み込み:
  - 公式名は `gsm8k`、サブセットは `main`。
- `load_dataset` → `save_to_disk` でローカル保存。
- parquet が嫌なら `to_json` で JSONL へ書き出し。

### JSONL 例
- 1行に `question` と `answer` があり、`answer` の末尾は `#### <数値>` 形式。

---

## SFT（transformers Trainer）での学習まとめ
### 重要ポイント
- `AutoModelForCausalLM.from_pretrained(...)` を使う。
- `TrainingArguments` には `output_dir` が必須。
- `datasets.load_dataset(...)` の戻りは `DatasetDict` なので `ds["train"]` を使う。

### tokenize の基本
- `tokenizer(batch["text"], truncation=True, max_length=...)` が基本。
- `labels` は `input_ids` のコピー（あるいは `DataCollatorForLanguageModeling` に任せる）。
- `padding="longest"` ならバッチ内最大長に合わせる。固定長なら `padding="max_length"`。

### よくあるバグ
- `remove_columns=ds["train"].column_names` にすると `answer` まで消える。
- `labels` が可変長のままだとテンソル化で落ちる。
- `DataCollatorWithPadding` は `labels` を作らないので、`labels` の扱いに注意。

---

## W&B / TensorBoard / ログのまとめ
- `report_to="wandb"` で W&B にログ。
- `wandb.log` で文字列を流す場合、グラフには出ないので `Table` が便利。
- `eval/accuracy` のように `eval/` プレフィックスをつけると eval セクションに並ぶ。

### W&B Table 例
```python
table = wandb.Table(columns=["step", "idx", "prompt", "output"])
for i, text in enumerate(texts):
    table.add_data(step, i, prompt, text)
wandb.log({"samples": table}, step=step)
```

---

## 評価（Accuracy）
### 最終解抽出
- 正規表現で `####` 以降の数値のみ抽出。
- 形式が崩れると `None` になるため精度が0になることがある。

```python
import re

def extract_answer(text):
    m = re.search(r"####\s*([\-0-9,\.]+)", text)
    return m.group(1).replace(",", "") if m else None
```

### 生成評価
- `logits.argmax` での評価は teacher-forcing なので実運用とズレる。
- 本番は `generate` で評価した方が正確。

---

## 生成ログ（on_evaluate callback）
- `TrainerCallback` の `on_evaluate` で生成とログを実施。
- `wandb.log` の `step` が重複すると上書きされるので注意。

```python
class SampleGenCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        texts = []
        for p in self.prompts:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=64)
            texts.append(tokenizer.decode(out[0], skip_special_tokens=True))
        table = wandb.Table(columns=["step", "idx", "prompt", "output"])
        for i, t in enumerate(texts):
            table.add_data(state.global_step, i, self.prompts[i], t)
        wandb.log({"samples": table})
```

---

## OOM 対策まとめ
- `bf16=True` でメモリを減らす（Ada GPU は対応）。
- `max_length` / `max_new_tokens` を小さくする。
- `per_device_eval_batch_size` を小さくする。
- `eval_accumulation_steps` や `prediction_loss_only=True` で保持を減らす。
- 断片化対策: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

## バッチ生成の評価（高速化）
- `tokenizer.pad(..., return_tensors="pt")` でまとめて pad。
- `model.generate` にバッチを渡せる。
- `batch_decode` でまとめて decode。

```python
batch = valid_dataset[start:start+batch_size]
inputs = tokenizer.pad(
    {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
    return_tensors="pt",
    padding=True,
).to(model.device)

out = model.generate(**inputs, max_new_tokens=64)
texts = tokenizer.batch_decode(out, skip_special_tokens=True)
```

---

## PPO 周辺の実験で詰まった点
- TRL 0.26〜0.27 の **PPO APIが揺れている**（`step` が無い版あり）。
- `AutoModelForCausalLMWithValueHead` に属性不足があり、以下の追加が必要だった:
  - `model.generation_config` / `model.pretrained_model.generation_config`
  - `model.base_model_prefix = "pretrained_model"`
  - `model.model = model.pretrained_model`
  - `model.is_gradient_checkpointing = False`
- `PPOTrainer` が `reward_model` / `value_model` 必須の版があり、
  ダミーの reward_model を用意して回避。

### ダミー reward_model 例
```python
class DummyRewardModel(nn.Module):
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            return torch.zeros(1)
        return torch.zeros(input_ids.size(0), device=input_ids.device)
```

---

## GRPO を検討する方向性
- TRL 0.27.1 では `GRPOTrainer` が推奨寄り。
- `prompt` 列が必須で、報酬関数は `reward_func(prompts, completions, ...)` 形式。
- 最終解一致で0/1報酬にするのが最初の実験として分かりやすい。

---

## tmux / シェル / 並列実行のメモ
- `CUDA_VISIBLE_DEVICES` で GPU を固定。
- バッチごとの実験を parallel に回す場合は `wait` を使う。
- tmux で pane を作り、各GPUにコマンドを投げる例。

```bash
SESSION=run4
GPUS=(0 1 2 3)
BATCHES=(1 2 4 8)

# セッション作成 & 分割
tmux new-session -d -s "$SESSION"
tmux split-window -h -t "$SESSION"
tmux split-window -v -t "$SESSION:0.0"
tmux split-window -v -t "$SESSION:0.1"

for i in "${!GPUS[@]}"; do
  CMD="CUDA_VISIBLE_DEVICES=${GPUS[$i]} uv run train.py --batch_size ${BATCHES[$i]}"
  tmux send-keys -t "$SESSION:0.$i" "$CMD" C-m
 done

tmux attach -t "$SESSION"
```

---

## 正規表現のメモ
- `$<<...>>` を削除したい場合:
```python
reg = r"\$?<<[^>]+>>"
text = re.sub(reg, "", text)
```
- `.` の貪欲さを避けたいときは `.+?`。

---

## よく使った確認系
- `ps -o user= -p <PID>` → プロセスのユーザー確認
- `watch -n 0.1 nvidia-smi -i 2` → GPU 監視

---

## 重要な気づき
- SFTは「質問+答え」を `text` にし、SFT用の学習が安定。
- 評価は「質問のみ」で生成させる方が自然。
- `compute_metrics` は logits を溜めるので OOM が起きやすい。
- 自前生成評価は callback でやると制御しやすい。

---

## いまの結論
- SFTは安定して回る。
- PPOは TRL の API 揺れで不安定になりがち。
- GRPOへ移行する方が現状は実装しやすい。

