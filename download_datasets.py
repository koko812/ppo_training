from datasets import load_dataset, load_from_disk
import os

dataset_name = "gsm8k"
subset = "main"
cache_dir = "dataset/cache"
save_dir = "dataset/gsm8k"
jsondir = "dataset/gsm8k_json"

if os.path.isdir(save_dir):
    ds = load_from_disk(save_dir)
else:
    ds = load_dataset(dataset_name, subset, cache_dir=cache_dir)
    ds.save_to_disk(save_dir)

os.makedirs(jsondir, exist_ok=True)
print(type(ds), ds.keys())
print(type(ds["train"]))
print(len(ds["train"]))

for split, dset in ds.items():
    dset.to_json(f"{jsondir}/{split}.jsonl")