import json
import os
import re

os.makedirs("dataset/gsm8k_json_no_bracket", exist_ok=True)
reg = r"\$?<<[^>]+>>"

for split in ["train", "test"]:
    with open(f"dataset/gsm8k_json/{split}.jsonl", "r") as f:
        with open(f"dataset/gsm8k_json_no_bracket/{split}.jsonl", "w") as g:
            for line in f:
                line = re.sub(reg, "", line)
                g.write(line)