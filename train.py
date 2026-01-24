"""
train.py => train model [svgbert]
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertModel, ModernBertConfig, get_linear_schedule_with_warmup

from model import SvgBert
from data_tool import prepare_data
from vis_tool import ModelCheckpointer, plot_compare_curves

def get_sign_labels(mantissa_tensor):
    sign_labels = torch.ones_like(mantissa_tensor, dtype=torch.long) # 默认为 1 (Zero)
    sign_labels[mantissa_tensor < -1e-9] = 0 # Negative
    sign_labels[mantissa_tensor > 1e-9] = 2  # Positive
    return sign_labels


def evaluate(args, model, loader, device, tokenizer, protected_ids_tensor):
    model.eval()
    
    total_loss = 0
    total_ce = 0
    total_mant = 0 # [CHANGE] 改名 mse -> mant
    total_exp = 0  # [ADD] 指数 loss
    total_sign = 0 # [ADD] 符号 loss
    steps = 0
    
    MASK_ID = tokenizer.mask_token_id
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            is_number = batch['is_number'].to(device)
            num_values = batch['num_values'].to(device) # [CHANGE] 需要用到
            
            mantissa_labels = batch['mantissa'].to(device)
            exponent_labels = batch['exponent'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # --- 1. 准备 Target (Label) ---
            labels = input_ids.clone()
            
            # [ADD] 生成符号标签 (0:负, 1:零, 2:正)
            sign_targets = get_sign_labels(mantissa_labels)
            
            # [ADD] 底数回归的目标必须是绝对值
            mantissa_targets_abs = torch.abs(mantissa_labels) 
            
            exponent_targets = exponent_labels.clone()

            # --- 2. Masking Input ---
            prob_matrix = torch.full(input_ids.shape, 0.3, device=device)
            prob_matrix.masked_fill_(torch.isin(input_ids, protected_ids_tensor), 0.0)
            masked_indices = torch.bernoulli(prob_matrix).bool()

            input_ids[masked_indices] = MASK_ID
            
            # [IMPORTANT] 输入给模型的数值在 Mask 位置必须清零，否则模型就直接看到答案了
            num_values[masked_indices] = 0.0 
            
            labels[~masked_indices] = -100 # Token Loss 忽略非 Mask 区域

            # --- 3. Forward (接收4个返回值) ---
            # [CHANGE] 解包 token, mantissa, exponent, sign
            token_logits, pred_mantissa_abs, exp_logits, sign_logits = model(input_ids, attention_mask, num_values, is_number)

            # --- 4. Loss Calculation ---
            
            # A. Token Cross Entropy
            loss_ce = F.cross_entropy(token_logits.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)
            
            # 筛选出被 Mask 的数值区域
            mask_num = masked_indices & (is_number > 0.5)
            
            if mask_num.any():
                # B. Sign Classification Loss (符号分类) [ADD]
                # 展平以便计算 CE
                loss_sign = F.cross_entropy(sign_logits[mask_num], sign_targets[mask_num])
                
                # C. Numerical Losses (仅针对 非零 数字计算 Mantissa 和 Exponent) [ADD]
                # 零没有底数和指数的概念，或者由符号头处理
                non_zero_mask = mask_num & (sign_targets != 1) 
                
                if non_zero_mask.any():
                    # Mantissa Regression (Absolute Value)
                    loss_mant = F.smooth_l1_loss(pred_mantissa_abs[non_zero_mask], mantissa_targets_abs[non_zero_mask])
                    
                    # Exponent Distribution (KL Div)
                    soft_targets = model.value_head.get_soft_labels(exponent_targets[non_zero_mask])
                    pred_exp_log_probs = F.log_softmax(exp_logits[non_zero_mask], dim=-1)
                    loss_exp = F.kl_div(pred_exp_log_probs, soft_targets, reduction='batchmean')
                else:
                    loss_mant = torch.tensor(0.0, device=device)
                    loss_exp = torch.tensor(0.0, device=device)
            else:
                loss_sign = torch.tensor(0.0, device=device)
                loss_mant = torch.tensor(0.0, device=device)
                loss_exp = torch.tensor(0.0, device=device)

            # 总 Loss (权重可调整)
            loss = loss_ce + loss_sign + loss_mant + loss_exp

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_mant += loss_mant.item()
            total_exp += loss_exp.item()
            total_sign += loss_sign.item()
            steps += 1

    model.train()
    # [CHANGE] 返回更多指标
    return total_loss/steps, total_ce/steps, total_mant/steps, total_exp/steps, total_sign/steps

def get_args():
    parser = argparse.ArgumentParser(description="ModernBERT Training Script")

    # --- 训练超参数 ---
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率 (Learning Rate)")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="物理 Batch Size (显存限制)")
    parser.add_argument("--target_batch_size", type=int, default=64, help="目标累积 Batch Size")
    parser.add_argument("--val_freq", type=int, default=10, help="模型一个epoch验证的次数")
    
    # --- 模型与路径配置 ---
    parser.add_argument("--base_model", type=str, default="answerdotai/ModernBERT-base", help="预训练模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="模型保存路径")
    parser.add_argument("--model_prefix", type=str, default="svgbert", help="保存模型的文件名前缀")
    parser.add_argument("--max_saved", type=int, default=2, help="最多保留几个最新的 Checkpoint")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="MSE loss 权重")
    parser.add_argument("--resume_from", type=str, default=None, help="从指定的 checkpoint路径 恢复训练")
    
    # --- 数据路径 (建议添加，方便 Docker 挂载) ---
    parser.add_argument("--data_path", type=str, default="VectorGraphics/svg-bert-data", help="数据集所在的文件夹路径")
    parser.add_argument("--data_configs", 
                   type=str, 
                   nargs='+', 
                   default=['FIGR-8', 'OmniSVG__MMSVG-Icon', 'OmniSVG__MMSVG-Illustration', 'freesvg', 'nyuuzyou__clker-svg', 'nyuuzyou__svgfind', 'starvector__svg-fonts', 'starvector_svg-stack', 'svgicons', 'svgrepo'],
                   help="使用的数据集配置列表")

    # 2. 解析参数
    args = parser.parse_args()
    
    # 3. save args
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {config_path}")


    return args


if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 梯度累积步数
    GRAD_ACCUM_STEPS = max(1, args.target_batch_size // args.batch_size)
    
    checkpointer = ModelCheckpointer(output_dir=args.output_dir, prefix=args.model_prefix, max_saved=args.max_saved)

    print(f"Device: {DEVICE} | Batch: {args.batch_size} | Accum: {GRAD_ACCUM_STEPS}")

    # --- 1. 初始化 Tokenizer & Model ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[NUM]"]})

    bert_base_config = ModernBertConfig.from_pretrained(args.base_model)
    bert_base = ModernBertModel(bert_base_config) # 从头初始化 [FIXED]
    bert_base.resize_token_embeddings(len(tokenizer))
    
    model = SvgBert(bert_base, bert_base.config.hidden_size, len(tokenizer))
    model.to(DEVICE)
    print("Model initialized successfully.")
    
    # --- 2. 准备数据 ---
    train_loader, val_loader = prepare_data(args, tokenizer)
    total_steps_per_epoch = len(train_loader)

    # --- 3. 优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # --- 4. 学习率调度器 ---
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    total_training_steps = steps_per_epoch * args.epochs
    warmup_ratio = 0.1 # ⚠️ 当前任务似乎不太需要太多的预热，如果是从头训练可以考虑更大比例
    num_warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
    print(f"Scheduler set: {num_warmup_steps} warmup steps over {total_training_steps} total steps.")


    # 定义特殊 Token (用于 Mask 逻辑)
    MASK_ID = tokenizer.mask_token_id
    protected_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id, MASK_ID}
    protected_ids_tensor = torch.tensor([tid for tid in protected_ids if tid is not None], device=DEVICE)

    history = {
        "train_loss": [], "train_ce": [], "train_mant": [], "train_exp": [], "train_sign": [], "train_steps": [],
        "val_loss": [],   "val_ce": [],   "val_mant": [],   "val_exp": [],   "val_sign": [], "val_steps": []
    }
    
    # [CHANGE] 更新绘图配置
    plot_configs = [
        {'train_data': history["train_loss"], 'val_data': history["val_loss"], 'title': "Total Loss", 'ylabel': "Loss", 'train_color': 'blue', 'val_color': 'dodgerblue'},
        {'train_data': history["train_ce"],   'val_data': history["val_ce"],   'title': "Token CE",   'ylabel': "CE",   'train_color': 'green', 'val_color': 'limegreen'},
        {'train_data': history["train_mant"], 'val_data': history["val_mant"], 'title': "Mantissa L1", 'ylabel': "L1",  'train_color': 'red',   'val_color': 'tomato'},
        {'train_data': history["train_sign"], 'val_data': history["val_sign"], 'title': "Sign CE",     'ylabel': "CE",  'train_color': 'purple','val_color': 'violet'},
        {'train_data': history["train_exp"],  'val_data': history["val_exp"],  'title': "Exp KL",      'ylabel': "KL",  'train_color': 'orange','val_color': 'gold'},
    ]

    print("Start Training...")
    
    # 定义验证频率 (例如：每个 epoch 验证 10 次)
    eval_interval = max(1, total_steps_per_epoch // args.val_freq) 
    
    # --- [新增] 恢复训练逻辑 ---
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        # 加载权重、优化器状态，并更新开始的 epoch 和 step
        start_epoch, global_step = checkpointer.load_checkpoint(
            args.resume_from, 
            model, 
            optimizer,
            scheduler=scheduler
        )


    for epoch in range(start_epoch, args.epochs):
        model.train()
        # [CHANGE] 更新统计字典
        train_stats = {"loss": 0, "ce": 0, "mant": 0, "exp": 0, "sign": 0, "steps": 0} 
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for step, batch in enumerate(loop):
            global_step += 1
            
            input_ids = batch['input_ids'].to(DEVICE)
            is_number = batch['is_number'].to(DEVICE)
            num_values = batch['num_values'].to(DEVICE) # [CHANGE] 取出数值
            
            mantissa_labels = batch['mantissa'].to(DEVICE)
            exponent_labels = batch['exponent'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            if torch.isnan(num_values).any() or torch.isinf(num_values).any():
                print(f"!!! 检测到坏数据 (num_values) at Step {global_step} !!!")
                raise ValueError("Input data contains NaN or Inf!")

            if torch.isnan(mantissa_labels).any() or torch.isinf(mantissa_labels).any():
                print(f"!!! 检测到坏数据 (mantissa) at Step {global_step} !!!")
                raise ValueError("Mantissa labels contain NaN or Inf!")


            labels = input_ids.clone()
            exponent_targets = exponent_labels.clone()
            
            # [ADD] 生成符号 Label & 绝对值 Mantissa Target
            sign_targets = get_sign_labels(mantissa_labels)
            mantissa_targets_abs = torch.abs(mantissa_labels)

            # --- Masking ---
            prob_matrix = torch.full(input_ids.shape, 0.3, device=DEVICE)
            prob_matrix.masked_fill_(torch.isin(input_ids, protected_ids_tensor), 0.0)
            masked_indices = torch.bernoulli(prob_matrix).bool()

            input_ids[masked_indices] = MASK_ID
            num_values[masked_indices] = 0.0 # [IMPORTANT] 输入必须要 Mask 掉，否则泄露答案
            labels[~masked_indices] = -100 

            # --- Forward (4返回值) ---
            # [CHANGE]
            token_logits, pred_mantissa_abs, exp_logits, sign_logits = model(input_ids, attention_mask, num_values, is_number)

            # --- Loss Calculation ---
            # 1. Token Loss
            loss_ce = nn.functional.cross_entropy(token_logits.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)
            
            # 2. Numerical Losses
            mask_num = masked_indices & (is_number > 0.5)
            
            if mask_num.any():
                # [ADD] Sign Loss
                loss_sign = F.cross_entropy(sign_logits[mask_num], sign_targets[mask_num])
                
                # [ADD] 仅对非零数字计算数值回归
                non_zero_mask = mask_num & (sign_targets != 1)
                
                if non_zero_mask.any():
                    # Mantissa (Abs)
                    loss_mant = F.smooth_l1_loss(pred_mantissa_abs[non_zero_mask], mantissa_targets_abs[non_zero_mask])
                    
                    # Exponent (Soft Label KL)
                    soft_targets = model.value_head.get_soft_labels(exponent_targets[non_zero_mask])
                    log_probs = F.log_softmax(exp_logits[non_zero_mask], dim=-1)
                    loss_exp = F.kl_div(log_probs, soft_targets, reduction='batchmean')
                else:
                    loss_mant = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                    loss_exp = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            else:
                loss_sign = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                loss_mant = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                loss_exp = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            # [CHANGE] 加权求和 (可以给 loss_sign 等加权重参数)
            loss = (loss_ce + loss_sign + args.mse_weight * loss_mant + loss_exp) / GRAD_ACCUM_STEPS
            loss.backward()

            # --- 统计 ---
            train_stats["loss"] += loss.item() * GRAD_ACCUM_STEPS
            train_stats["ce"] += loss_ce.item()
            train_stats["mant"] += loss_mant.item() # [CHANGE]
            train_stats["exp"] += loss_exp.item()   # [CHANGE]
            train_stats["sign"] += loss_sign.item() # [CHANGE]
            train_stats["steps"] += 1
            
            if global_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loop.set_postfix({
                'Loss': f"{loss.item()*GRAD_ACCUM_STEPS:.3f}", 
                'CE': f"{loss_ce.item():.2f}", 
                'Sgn': f"{loss_sign.item():.2f}",
                'Mnt': f"{loss_mant.item():.2f}", 
                'Exp': f"{loss_exp.item():.2f}"
            })

            # --- Evaluation ---
            if (step + 1) % eval_interval == 0 or (step + 1) == total_steps_per_epoch:
                avg_t = {k: v / train_stats["steps"] for k, v in train_stats.items() if k != "steps"}
                
                history["train_loss"].append(avg_t["loss"])
                history["train_ce"].append(avg_t["ce"])
                history["train_mant"].append(avg_t["mant"]) # [CHANGE]
                history["train_exp"].append(avg_t["exp"])   # [CHANGE]
                history["train_sign"].append(avg_t["sign"]) # [CHANGE]
                history["train_steps"].append(global_step)

                # [CHANGE] 解包5个返回值
                val_metrics = evaluate(args, model, val_loader, DEVICE, tokenizer, protected_ids_tensor)
                v_loss, v_ce, v_mant, v_exp, v_sign = val_metrics 

                history["val_loss"].append(v_loss)
                history["val_ce"].append(v_ce)
                history["val_mant"].append(v_mant) # [CHANGE]
                history["val_exp"].append(v_exp)   # [CHANGE]
                history["val_sign"].append(v_sign) # [CHANGE]
                history["val_steps"].append(global_step)

                print(f"\n=== Step {global_step} Evaluation ===")
                print(f"Train | Loss: {avg_t['loss']:.4f} | CE: {avg_t['ce']:.4f} | Sgn: {avg_t['sign']:.4f} | Mnt: {avg_t['mant']:.4f} | Exp: {avg_t['exp']:.4f}")
                print(f"Val   | Loss: {v_loss:.4f} | CE: {v_ce:.4f} | Sgn: {v_sign:.4f} | Mnt: {v_mant:.4f} | Exp: {v_exp:.4f}")
                print("-" * 80)

                plot_compare_curves(history, plot_configs=plot_configs, save_path="comparison_curves.png", max_cols=5)

                checkpointer.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=global_step, 
                    epoch=epoch, 
                    val_loss=v_loss,
                    is_best=v_loss < checkpointer.best_val_loss,
                    scheduler=scheduler
                )
                
                train_stats = {"loss": 0, "ce": 0, "mant": 0, "exp": 0, "sign": 0, "steps": 0}
                model.train()

    print("Training Complete.")