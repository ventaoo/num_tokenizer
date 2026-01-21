"""
train.py => train model [svgbert]
"""
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

from model import SvgBert
from tool import re_transform
from data_tool import prepare_data
from vis_tool import ModelCheckpointer, plot_compare_curves

def evaluate(args, model, loader, device, tokenizer, protected_ids_tensor):
    model.eval() # 切换到评估模式
    total_loss = 0
    total_ce = 0
    total_mse = 0
    total_ori_mse = 0
    steps = 0
    
    MASK_ID = tokenizer.mask_token_id
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            is_number = batch['is_number'].to(device)
            num_values = batch['num_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # --- MLM 逻辑与训练保持一致 ---
            labels = input_ids.clone()
            value_labels = num_values.clone()

            # 验证集我们也随机 Mask 吗？
            # 这里的标准做法通常是：
            # 1. 为了计算 Loss 对比，验证集也需要做 Mask。
            # 2. 为了结果稳定，最好固定 Seed，但在简单的 Loop 中随机也可以，因为样本量大。
            probability_matrix = torch.full(input_ids.shape, 0.2, device=device)
            current_protected_mask = torch.isin(input_ids, protected_ids_tensor)
            probability_matrix.masked_fill_(current_protected_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()

            input_ids[masked_indices] = MASK_ID
            num_values[masked_indices] = 0.0
            labels[~masked_indices] = -100

            token_logits, value_pred = model(input_ids, attention_mask, num_values, is_number)

            loss_ce = nn.functional.cross_entropy(
                token_logits.view(-1, len(tokenizer)),
                labels.view(-1),
                ignore_index=-100
            )

            mask_for_regression = masked_indices & (is_number > 0.5)
            if mask_for_regression.any():
                loss_mse = nn.functional.smooth_l1_loss(
                    value_pred[mask_for_regression],
                    value_labels[mask_for_regression]
                )

                # ORI MSE loss
                loss_ori_mse = nn.functional.smooth_l1_loss(
                    re_transform(value_pred[mask_for_regression]),
                    re_transform(value_labels[mask_for_regression])
                )

            else:
                loss_mse = torch.tensor(0.0, device=device)
                loss_ori_mse = torch.tensor(0.0, device=DEVICE)

            loss = loss_ce + args.mse_weight * loss_mse
            
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_mse += loss_mse.item()
            total_ori_mse += loss_ori_mse.item()
            steps += 1

    model.train()
    return total_loss / steps, total_ce / steps, total_mse / steps, total_ori_mse / steps

def get_args():
    parser = argparse.ArgumentParser(description="ModernBERT Training Script")

    # --- 训练超参数 ---
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率 (Learning Rate)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="物理 Batch Size (显存限制)")
    parser.add_argument("--target_batch_size", type=int, default=64, help="目标累积 Batch Size")
    
    # --- 模型与路径配置 ---
    parser.add_argument("--base_model", type=str, default="answerdotai/ModernBERT-base", help="预训练模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="模型保存路径")
    parser.add_argument("--model_prefix", type=str, default="svgbert", help="保存模型的文件名前缀")
    parser.add_argument("--max_saved", type=int, default=3, help="最多保留几个最新的 Checkpoint")
    parser.add_argument("--mse_weight", type=float, default=20.0, help="MSE loss 权重")
    
    # --- 数据路径 (建议添加，方便 Docker 挂载) ---
    parser.add_argument("--data_path", type=str, default="VectorGraphics/svg-corpus-private", help="数据集所在的文件夹路径")
    parser.add_argument("--data_size", type=int, default=10, help="抽取多少个数据块")

    # 2. 解析参数
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRAD_ACCUM_STEPS = max(1, args.target_batch_size // args.batch_size)
    checkpointer = ModelCheckpointer(output_dir=args.output_dir, prefix=args.model_prefix, max_saved=args.max_saved)

    print(f"Device: {DEVICE}")
    print(f"Physical Batch: {args.batch_size} | Accumulation Steps: {GRAD_ACCUM_STEPS} | Effective Batch: {args.target_batch_size}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[NUM]"]})
    print(f"Tokenizer initialized successfully.")

    # Dataloader
    train_loader, val_loader = prepare_data(args, tokenizer)
    print("DataLoader initialized successfully.")

    # Model
    bert_base = AutoModel.from_pretrained(args.base_model)
    bert_base.resize_token_embeddings(len(tokenizer))
    H = bert_base.config.hidden_size
    model = SvgBert(bert_base, H, len(tokenizer))
    model.to(DEVICE)
    print("Model initialized successfully.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 获取特殊 Token ID
    MASK_ID = tokenizer.mask_token_id
    PAD_ID = tokenizer.pad_token_id
    NUM_ID = tokenizer.convert_tokens_to_ids("[NUM]")

    print(f"Start Training... [NUM] ID: {NUM_ID} | [MASK] ID: {MASK_ID}")

    special_tokens = {
        tokenizer.cls_token_id, tokenizer.sep_token_id, 
        tokenizer.pad_token_id, tokenizer.mask_token_id, 
        tokenizer.unk_token_id
    }
    protected_ids_list = [tid for tid in special_tokens if tid is not None]
    protected_ids_tensor = torch.tensor(protected_ids_list, device=DEVICE)
    print(f"Protected Special Tokens (Never Mask): {protected_ids_list}")
    print(f"Maskable Target Token: [NUM] ({NUM_ID})")


    history = {
        "train_loss": [], "train_ce": [], "train_mse": [], "train_ori_mse": [], "train_steps": [], # 新增
        "val_loss": [],   "val_ce": [],   "val_mse": [], "val_ori_mse": [], "val_steps": []     # 新增
    }
    
    total_steps_per_epoch = len(train_loader)
    train_stats = {"loss": 0, "ce": 0, "mse": 0, "ori_mse": 0, "steps": 0}
    print("Start Training...")
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for step, batch in enumerate(loop):
            current_global_step = epoch * total_steps_per_epoch + (step + 1)

            input_ids = batch['input_ids'].to(DEVICE)           # [B, L]
            is_number = batch['is_number'].to(DEVICE)           # [B, L]
            num_values = batch['num_values'].to(DEVICE)         # [B, L]
            attention_mask = batch['attention_mask'].to(DEVICE) # [B, L]

            labels = input_ids.clone()
            value_labels = num_values.clone()

            probability_matrix = torch.full(input_ids.shape, 0.2, device=DEVICE) # 20% mask
            current_protected_mask = torch.isin(input_ids, protected_ids_tensor)
            probability_matrix.masked_fill_(current_protected_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()

            # mask
            input_ids[masked_indices] = MASK_ID
            num_values[masked_indices] = 0.0
            labels[~masked_indices] = -100

            # forawrd
            token_logits, value_pred = model(input_ids, attention_mask, num_values, is_number)

            # loss
            loss_ce = nn.functional.cross_entropy(
                token_logits.view(-1, len(tokenizer)),
                labels.view(-1),
                ignore_index=-100
            )

            mask_for_regression = masked_indices & (is_number > 0.5)            
            if mask_for_regression.any():
                loss_mse = nn.functional.smooth_l1_loss(
                    value_pred[mask_for_regression],
                    value_labels[mask_for_regression]
                )

                """To show how this model work in the ori space."""
                loss_ori_mse = nn.functional.smooth_l1_loss(
                    re_transform(value_pred[mask_for_regression]),
                    re_transform(value_labels[mask_for_regression])
                )
                
            else:
                loss_mse = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                loss_ori_mse = torch.tensor(0.0, device=DEVICE)

            loss = loss_ce + args.mse_weight * loss_mse
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            # Accumulate for stats (rescale back)
            train_stats["loss"] += loss.item() * GRAD_ACCUM_STEPS
            train_stats["ce"] += loss_ce.item()
            train_stats["mse"] += loss_mse.item()
            train_stats["ori_mse"] += loss_ori_mse.item()
            train_stats["steps"] += 1

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item() * GRAD_ACCUM_STEPS
            loop.set_postfix({
                'Loss': f"{current_loss:.3f}",
                'CE': f"{loss_ce.item():.3f}", 
                'MSE': f"{loss_mse.item():.3f}",
                'ORI_MSE': f"{loss_ori_mse.item():.3f}"
            })

            val_log_interval = len(loop) // 10
            if (step + 1) % val_log_interval == 0:
                train_avg_loss = train_stats["loss"] / train_stats["steps"]
                train_avg_ce = train_stats["ce"] / train_stats["steps"]
                train_avg_mse = train_stats["mse"] / train_stats["steps"]
                train_avg_ori_mse = train_stats["ori_mse"] / train_stats["steps"]

                history["train_loss"].append(train_avg_loss)
                history["train_ce"].append(train_avg_ce)
                history["train_mse"].append(train_avg_mse)
                history["train_ori_mse"].append(train_avg_ori_mse)
                history["train_steps"].append(current_global_step)

                val_avg_loss, val_avg_ce, val_avg_mse, val_avg_ori_mse = evaluate(
                    args, model, val_loader, DEVICE, tokenizer, protected_ids_tensor
                )
                
                history["val_loss"].append(val_avg_loss)
                history["val_ce"].append(val_avg_ce)
                history["val_mse"].append(val_avg_mse)
                history["val_ori_mse"].append(val_avg_ori_mse)
                history["val_steps"].append(current_global_step)

                print(f"\nEpoch {epoch+1} (Step {step+1}) Summary:")
                print(f"Train Loss: {train_avg_loss:.4f} | CE: {train_avg_ce:.4f} | MSE: {train_avg_mse:.4f} | ORIMSE: {train_avg_ori_mse:.4f}")
                print(f"Val   Loss: {val_avg_loss:.4f} | CE: {val_avg_ce:.4f} | MSE: {val_avg_mse:.4f} | ORIMSE: {val_avg_ori_mse:.4f}")
                
                train_stats = {"loss": 0, "ce": 0, "mse": 0, "ori_mse": 0, "steps": 0}

            save_log_interval = len(loop) // 5
            if (step + 1) % save_log_interval == 0:   
                checkpointer.save_checkpoint(
                    model=model,
                    step=current_global_step, 
                    epoch=epoch + 1, 
                    val_loss=val_avg_loss,
                    is_best=val_avg_loss < checkpointer.best_val_loss
                )
                plot_compare_curves(history, save_path=f"comparison_curves.png")

    print("Training Complete. History recorded.")