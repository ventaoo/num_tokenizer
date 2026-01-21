import torch
import random
import datasets
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tool import process_single_svg_str


def encode_svg(example, max_len=2048, tokenizer=None):
    valid_keys = [
        k for k in ['raw_svg', 'stage1_svg', 'stage2_svg', 'stage2_augmented_svg'] 
        if example.get(k) is not None and str(example.get(k)) != "None" and isinstance(example.get(k), str)
    ]

    outputs = {
        "input_ids": [], "is_number": [], "num_values": [], "attention_mask": []
    }

    filename = example.get('filename', 'unknown')
    if not valid_keys:
        print(f"Skipping {filename}: No valid SVG keys found.")
        return outputs
    
    key = random.choice(valid_keys)
    try:
        processed = process_single_svg_str(example.get(key), max_len, tokenizer)
        if not processed or len(processed["input_ids"]) == 0:
            return outputs
            
        return processed
    except Exception as e:
        print(f"Error processing {key} in {filename}: {e}")

    return outputs

def encode_svg_batched(examples, max_len=2048, tokenizer=None):
    batch_outputs = {
        "input_ids": [],
        "is_number": [],
        "num_values": [],
        "attention_mask": []
    }
    
    batch_size = len(examples.get('filename', []))
    
    for i in range(batch_size):
        # 从批次字典中，为第 i 个样本构建一个单独的 example 字典
        single_example = {key: examples[key][i] for key in examples.keys()}
        
        single_output = encode_svg(single_example, max_len=max_len, tokenizer=tokenizer)
        
        batch_outputs['input_ids'].append(single_output.get('input_ids', []))
        batch_outputs['is_number'].append(single_output.get('is_number', []))
        batch_outputs['num_values'].append(single_output.get('num_values', []))
        batch_outputs['attention_mask'].append(single_output.get('attention_mask', []))
    
    return batch_outputs


def collate_fn(batch, tokenizer):
    input_ids_tensors = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    is_number_tensors = [torch.tensor(item["is_number"], dtype=torch.float) for item in batch]
    num_values_tensors = [torch.tensor(item["num_values"], dtype=torch.float) for item in batch]
    attention_mask_tensors = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]

    return {
        "input_ids": pad_sequence(input_ids_tensors, batch_first=True, padding_value=tokenizer.pad_token_id),
        "is_number": pad_sequence(is_number_tensors, batch_first=True, padding_value=0.0),
        "num_values": pad_sequence(num_values_tensors, batch_first=True, padding_value=0.0),
        "attention_mask": pad_sequence(attention_mask_tensors, batch_first=True, padding_value=0)
    }

def load_data_in_server(data_path, size=100):
    arr = []

    for i in tqdm(range(size)):
        split_number = ("00000" + str(i))[-5:]
        split = f"svg-corpus-train-{split_number}-of-00244.arrow"
        ds = datasets.Dataset.from_file(f"{data_path}/{split}")
        arr.append(ds)
    
    ds = datasets.concatenate_datasets(arr)

    return ds

def prepare_data(args, tokenizer):
    print("Loading Dataset...")
    ds = datasets.load_dataset('VectorGraphics/svg-corpus-private', 'svg_viewer_dataset', split='train')
    # ds = load_data_in_server(args.data_path, args.data_size) # ⚠️ 服务器上的训练

    split_ds = ds.train_test_split(test_size=min(10000, len(ds) // 10), seed=42, shuffle=True)
    print(f"Data Split: Train={len(split_ds['train'])}, Val={len(split_ds['test'])}")

    process_with_config = partial(
        encode_svg_batched,
        max_len=2048,
        tokenizer=tokenizer
    )

    # Lazy Loading
    split_ds['train'].set_transform(process_with_config)
    split_ds['test'].set_transform(process_with_config)

    print("Dataset is ready (Lazy Loading mode).")
    print(f"Expanded Dataset Size: Train={len(split_ds['train'])}, Val={len(split_ds['test'])}")

    collate_fn_w_tokenizer = partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(
        split_ds['train'], 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_w_tokenizer,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        split_ds['test'], 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_w_tokenizer,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # ROOT_DIR = "/data/svg-corpus/"
    # ds = load_data_in_server(ROOT_DIR, size=100)
    # ds.cleanup_cache_files() # clean cache
    # print(f"load dataset: {ds}")
    # print(ds.info, ds.features)
    # print(f"Total samples: {len(ds)}")

    import os
    from dotenv import load_dotenv
    from transformers import AutoTokenizer
    
    load_dotenv()
    if os.getenv('HF_TOKEN'):
        from huggingface_hub import login
        login(token=os.getenv('HF_TOKEN'))
    ds = datasets.load_dataset('VectorGraphics/svg-corpus-private', 'svg_viewer_dataset', split='train')
    print(f"load dataset: {ds}")

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[NUM]"]})
    print(f"Tokenizer initialized successfully.")

    processed_str = process_single_svg_str(ds[0]['raw_svg'][:200], max_len=300, tokenizer=tokenizer)
    print(ds[0]['raw_svg'][:200])
    for key, item in processed_str.items():
        print(f"{key}: {item}\n\n")
    
    print(tokenizer.convert_ids_to_tokens(processed_str['input_ids']))
