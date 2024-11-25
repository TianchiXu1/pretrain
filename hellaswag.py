"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import tiktoken
from transformers import GPT2LMHeadModel

# 模型和设备配置
MODEL_PATH = "/abies/ECMWF_Precipitation/code/other/LLM/gpt2/model"  # 本地模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 根据是否有GPU自动选择设备

# 数据缓存路径
DATA_CACHE_DIR = "/abies/ECMWF_Precipitation/code/other/LLM/gpt2/hellaswag"  # 已下载的数据文件夹

enc = tiktoken.get_encoding("gpt2")

def render_example(example):
    """
    将字典形式的例子渲染为三个张量：
    - tokens：上下文+完成的tokens (4xN)
    - mask：标识候选完成的区域
    - label：正确完成的索引
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # Prepend a space to match GPT-2 tokenization
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label

def iterate_examples(split):
    """迭代已下载的HellaSwag数据集"""
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    with open(data_filename, "r") as f:
        for line in f:
            yield json.loads(line)

@torch.no_grad()
def evaluate():
    """在HellaSwag数据集上评估GPT-2模型"""
    print(f"加载模型路径: {MODEL_PATH}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)  # 加载本地模型
    model.to(DEVICE)

    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)

        logits = model(tokens).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        shift_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_tokens.view(-1),
            reduction="none",
        ).view(tokens.size(0), -1)

        masked_losses = shift_losses * mask[..., 1:]
        avg_loss = masked_losses.sum(dim=1) / mask[..., 1:].sum(dim=1)
        pred = avg_loss.argmin().item()

        num_correct += int(pred == label)
        num_total += 1

        if num_total <= 10:  # Show details for the first 10 examples
            print(f"Example {num_total}: Prediction={pred}, Label={label}")

    accuracy = num_correct / num_total
    print(f"Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")

if __name__ == "__main__":
    evaluate()
