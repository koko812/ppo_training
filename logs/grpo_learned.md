# GRPO これまでの学び（メモ）

## GRPO の基本方針
- PPO よりも **GRPO のほうが実装が素直**で、TRL 0.27.1 では GRPO 推奨寄り。
- `GRPOTrainer` は **`reward_funcs`（複数形）**を使う。`reward_func` ではない。
- `data_collator` は GRPOTrainer の引数に無い。`collate_fn` を直接渡すことはできない。
- `reward_func` のシグネチャは **`reward_func(prompts, completions, **kwargs)`** が安全。
  - `completion_ids` などが `kwargs` 経由で渡されるため `**kwargs` は必須。

## Dataset の作り方
- `prompt` 列が必須。
- `question` → `prompt` への変換が基本。
- `answer` は報酬計算に使うため残す。

### 例（prompt作成）
```python
def build_prompt(batch):
    prompt = "Answer the next question:\n{}"
    return {"prompt": [prompt.format(q) for q in batch["question"]]}
```

## Reward 設計
- GSM8K は `#### <num>` の抽出が必須。
- `answer` は `kwargs["batch"]["answer"]` から直接取れない場合がある。
  - TRL 0.27.1 の GRPOTrainer は `reward_func` に `batch` を渡さないことがある。
  - **prompt→answer の辞書**で参照するのが安定。

### 例（最終解一致 0/1）
```python
import re

def extract_answer(text):
    m = re.search(r"####\s*([\-0-9,\.]+)", text)
    return m.group(1).replace(",", "") if m else None

answer_map = {q: a for q, a in zip(ds["train"]["question"], ds["train"]["answer"])}

def reward_func(prompts, completions, **kwargs):
    rewards = []
    for p, c in zip(prompts, completions):
        q = p.replace("Answer the next question:\n", "").strip()
        gold = answer_map.get(q, "")
        rewards.append(1.0 if extract_answer(c) == extract_answer(gold) else 0.0)
    return rewards
```

## よく出たエラーと対処
- `reward_func() got an unexpected keyword argument 'completion_ids'`
  - `**kwargs` を受けるように変更。
- `GRPOTrainer.__init__() got an unexpected keyword argument 'reward_func'`
  - `reward_funcs=[reward_func]` に修正。

---

# GRPO 高速化メモ（現状コード向け）

### 症状
- GPU メモリは埋まっているが **GPU util が低い**
- profiling では **`transformers.generate` と `_prepare_inputs` が支配**
- たまに step が異常に遅い（スパイク）

---

## 原因
1. **生成が細切れ / 遅い**
2. **`_prepare_inputs` が毎 step 重い**
   → 事前 tokenize + trainer 内 tokenize の **二重処理**
3. **prompt + completion 長が不安定**
   → 計算量が step ごとにバラつく

---

## 最優先の対策（bs=64 を維持したまま）

### ① 長さを分離・固定
- `max_prompt_length = 128`
- `max_completion_length (or max_new_tokens) = 128`
  → スパイク削減・平均 step 短縮

### ② 二重 tokenize をやめる
**どちらか1つにする**
- A案（おすすめ）：dataset の `map(tokenize)` を削除
  → token 化は trainer に任せる
- B案：事前 tokenize を使うなら `data_collator=collate_fn` を必ず渡す

### ③ 生成を高速化
- **vLLM を使えるなら最優先で ON**
- 生成が支配のGRPOでは **1.5〜3×** の伸びが現実的

### ④ 生成の上限を明示
- GRPO 側に **`max_new_tokens` / `generation_config`** を必ず渡す
  （今の `generation_kwargs` は未使用）

---

## 期待効果
- まず **2×**（スパイク消滅 + prepare_inputs 軽量化）
- vLLM が刺されば **3×以上**
- その後 GPU 変更（H100 等）で **さらに 2〜4×**

---

## すぐ確認すること
1. `help(GRPOConfig)` に
   - `max_prompt_length`
   - `max_completion_length` or `max_new_tokens`
   - `use_vllm`
     があるか
2. `trl.__version__`

---

# vLLM で動かすまでに学んだこと（追記）

## 依存関係の落とし穴
- TRL が対応している vLLM の **バージョン範囲が狭い**。\n  例: 0.10.2 / 0.11.0 / 0.11.1 / 0.11.2 / 0.12.0 など。\n- vLLM のバージョンによっては **torch の要求バージョン**が噛み合わずインストールに失敗する。\n- `CUDA_HOME is not set` のようなエラーは **インストール時のビルド失敗**で、\n  「vLLM を起動しろ」という意味ではない。

## nvcc/Inductor まわりの回避策
- `vllm_mode=\"colocate\"` を使うと **torch.compile / inductor** 経由に入りやすい。\n- `nvcc` が使えない環境だと `PermissionError: nvcc` で落ちる。\n- 下記の設定で **Dynamo / Inductor を完全無効化**すると回避できた。

```bash
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
```

## 実際に有効だった vLLM 設定例
（環境依存だが、これで「動作 + 3倍程度の高速化」）

```python
use_vllm=True
vllm_mode="colocate"
vllm_model_impl="vllm"
vllm_gpu_memory_utilization=0.3  # OOMなら 0.2〜0.4 で調整
vllm_max_model_length=256
use_transformers_paged=False      # HF paged attention を切る
cache_implementation=None         # torch.compile/inductor 経路を避ける
torch_compile=False               # ★重要：Inductorを切る
torchdynamo=None                  # ★重要：Dynamoを切る
```

## 体感効果
- vLLM が安定すると **2〜3× 以上の高速化**が出やすい。\n- 速度が出ない場合は、\n  - `vllm_gpu_memory_utilization`\n  - `max_prompt_length` / `max_completion_length`\n  を調整すると改善しやすい。
