import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# 设置本地目录路径
local_dir = "/abies/ECMWF_Precipitation/code/other/LLM/gpt2/edu_fineweb10B_sample/data"
shard_size = int(1e8)  # 每个分片 100M tokens

# 创建缓存目录，如果目录不存在
DATA_CACHE_DIR = local_dir

def load_parquet_values(local_dir, prefix):
    """
    从本地目录加载指定前缀 (train/val) 的 Parquet 文件中的 values 列。
    """
    files = [f for f in os.listdir(local_dir) if f.endswith('.parquet') and f.startswith(prefix)]
    all_values = []
    for file in files:
        file_path = os.path.join(local_dir, file)
        df = pd.read_parquet(file_path)
        if 'values' in df.columns:
            all_values.extend(df['values'].tolist())  # 将 'values' 列的所有数据提取到列表
        else:
            print(f"Warning: 'values' column not found in {file_path}")
    return all_values

def write_datafile(filename, tokens_np):
    """
    将 token 数据写入文件。
    """
    np.save(filename, tokens_np)

def process_tokens(tokens):
    """
    将每个文档的 tokens 转换为 numpy 数组并确保它们符合 uint16 类型。
    """
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def process_shards(all_values, split_name):
    """
    处理 tokens 数据并分片保存。
    """
    nprocs = max(1, os.cpu_count() * 9 // 10)  # 设置并行任务数量
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        token_count = 0
        progress_bar = None
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # 用来存放当前分片的 token 数据

        for tokens in pool.imap(process_tokens, all_values, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"{split_name.capitalize()} Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                if progress_bar:
                    progress_bar.update(shard_size - token_count)
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split_name}_{shard_index:06d}")
                write_datafile(filename, all_tokens_np)
                shard_index += 1

                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"{split_name.capitalize()} Shard {shard_index}")
                all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # 用来存放下一次分片的 token 数据
                all_tokens_np[:len(tokens)] = tokens
                token_count = len(tokens)

        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split_name}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

# 分别加载并处理训练集和验证集
train_values = load_parquet_values(local_dir, "train")
process_shards(train_values, "train")

val_values = load_parquet_values(local_dir, "val")
process_shards(val_values, "val")
