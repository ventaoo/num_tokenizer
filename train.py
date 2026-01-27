"""
train.py => train model [svgbert]
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertModel, ModernBertConfig, get_linear_schedule_with_warmup

from model import SvgBert
from tool import get_sign_labels
from data_tool import prepare_data
from vis_tool import ModelCheckpointer, plot_compare_curves

def evaluate(args, model, loader, device, tokenizer, protected_ids_tensor):
    model.eval()
    
    keys = ["loss", "ce", "val_num"] if args.is_regression_only else ["loss", "ce", "mant", "exp", "sign"]
    total_stats = {k: 0.0 for k in keys}
    steps = 0
    MASK_ID = tokenizer.mask_token_id
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            # --- 1. 数据准备 ---
            input_ids = batch['input_ids'].to(device)
            is_number = batch['is_number'].to(device)
            
            # [CRITICAL] 区分输入值和目标值
            target_values = batch['num_values'].to(device) # 真值 (Label)
            input_values = target_values.clone()           # 输入值 (将被 Mask)
            
            mantissa_labels = batch['mantissa'].to(device)
            exponent_labels = batch['exponent'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # --- 2. Masking ---
            prob_matrix = torch.full(input_ids.shape, 0.3, device=device)
            prob_matrix.masked_fill_(torch.isin(input_ids, protected_ids_tensor), 0.0)
            masked_indices = torch.bernoulli(prob_matrix).bool()

            # A. Mask Token
            labels = input_ids.clone()
            input_ids[masked_indices] = MASK_ID
            labels[~masked_indices] = -100 # 忽略非 Mask 区域
            # B. Mask Number Inputs
            input_values[masked_indices] = 0.0 

            # --- 3. Forward & Loss Calculation ---
            outputs = model(input_ids, attention_mask, input_values, is_number)

            if args.is_regression_only:
                token_logits, pred_values = outputs
                
                # Token Loss
                loss_ce = F.cross_entropy(token_logits.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)
                
                # Regression Loss
                mask_num = masked_indices & (is_number > 0.5)
                if mask_num.any():
                    # 比较预测值与原始 Target Values
                    loss_val = F.smooth_l1_loss(pred_values[mask_num], target_values[mask_num])
                else:
                    loss_val = torch.tensor(0.0, device=device)
                
                # 汇总
                total_loss = loss_ce + args.mse_weight * loss_val
                
                # 记录
                total_stats["loss"] += total_loss.item()
                total_stats["ce"] += loss_ce.item()
                total_stats["val_num"] += loss_val.item()

            else:
                token_logits, pred_mantissa_abs, exp_logits, sign_logits = outputs
                
                # 准备 Labels
                sign_targets = get_sign_labels(mantissa_labels)
                mantissa_targets_abs = torch.abs(mantissa_labels)
                exponent_targets = exponent_labels # 只读，无需 clone

                # Token Loss
                loss_ce = F.cross_entropy(token_logits.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)
                
                # Numerical Losses
                mask_num = masked_indices & (is_number > 0.5)
                loss_sign = loss_mant = loss_exp = torch.tensor(0.0, device=device)

                if mask_num.any():
                    loss_sign = F.cross_entropy(sign_logits[mask_num], sign_targets[mask_num])
                    
                    non_zero_mask = mask_num & (sign_targets != 1)
                    if non_zero_mask.any():
                        loss_mant = F.smooth_l1_loss(pred_mantissa_abs[non_zero_mask], mantissa_targets_abs[non_zero_mask])
                        
                        soft_targets = model.value_head.get_soft_labels(exponent_targets[non_zero_mask])
                        pred_exp_log_probs = F.log_softmax(exp_logits[non_zero_mask], dim=-1)
                        loss_exp = F.kl_div(pred_exp_log_probs, soft_targets, reduction='batchmean')
                
                # 汇总
                total_loss = loss_ce + loss_sign + loss_mant + loss_exp
                
                # 记录
                total_stats["loss"] += total_loss.item()
                total_stats["ce"] += loss_ce.item()
                total_stats["mant"] += loss_mant.item()
                total_stats["exp"] += loss_exp.item()
                total_stats["sign"] += loss_sign.item()

            steps += 1

    model.train()
    return {k: v / steps for k, v in total_stats.items()}

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

    parser.add_argument("--is_multi_scale", action="store_true", help="是否使用多尺度输入嵌入")
    parser.add_argument("--is_regression_only", action="store_true", help="是否只进行数值回归任务")

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
    
    model = SvgBert(bert_base, bert_base.config.hidden_size, len(tokenizer), is_multi_scale=args.is_multi_scale, is_regression_only=args.is_regression_only)
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

    if args.is_regression_only:
        print("Training in Regression-Only Mode.")
        history = {
            "train_loss": [], "train_ce": [], "train_smooth_l1": [], "train_steps": [],
            "val_loss": [],   "val_ce": [],   "val_smooth_l1": [],   "val_steps": []
        }
        plot_configs = [
            {'train_data': history["train_loss"], 'val_data': history["val_loss"], 'title': "Total Loss", 'ylabel': "Loss", 'train_color': 'blue', 'val_color': 'dodgerblue'},
            {'train_data': history["train_ce"],   'val_data': history["val_ce"],   'title': "Token CE",   'ylabel': "CE",   'train_color': 'green', 'val_color': 'limegreen'},
            {'train_data': history["train_smooth_l1"], 'val_data': history["val_smooth_l1"], 'title': "Smooth L1", 'ylabel': "L1",  'train_color': 'red',   'val_color': 'tomato'},
        ]
    else:
        print("Training in Scientific Mode.")
        history = {
            "train_loss": [], "train_ce": [], "train_mant": [], "train_exp": [], "train_sign": [], "train_steps": [],
            "val_loss": [],   "val_ce": [],   "val_mant": [],   "val_exp": [],   "val_sign": [], "val_steps": []
        }
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
        
        if args.is_regression_only:
            train_stats = {"loss": 0, "ce": 0, "val_num": 0, "steps": 0} 
        else:
            train_stats = {"loss": 0, "ce": 0, "mant": 0, "exp": 0, "sign": 0, "steps": 0}
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for step, batch in enumerate(loop):
            global_step += 1
            
            input_ids = batch['input_ids'].to(DEVICE)
            is_number = batch['is_number'].to(DEVICE)
            
            target_values = batch['num_values'].to(DEVICE) 
            input_values = target_values.clone() # 先复制一份作为输入
            
            mantissa_labels = batch['mantissa'].to(DEVICE)
            exponent_labels = batch['exponent'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            if torch.isnan(target_values).any() or torch.isinf(target_values).any():
                print(f"!!! 检测到坏数据 (num_values) at Step {global_step} !!!")
                raise ValueError("Input data contains NaN or Inf!")

            labels = input_ids.clone()

            # --- Masking ---
            prob_matrix = torch.full(input_ids.shape, 0.3, device=DEVICE)
            prob_matrix.masked_fill_(torch.isin(input_ids, protected_ids_tensor), 0.0)
            masked_indices = torch.bernoulli(prob_matrix).bool()

            input_ids[masked_indices] = MASK_ID
            input_values[masked_indices] = 0.0 
            labels[~masked_indices] = -100
            
            # --- Forward & Loss Calculation (根据模式分支) ---
            outputs = model(input_ids, attention_mask, input_values, is_number)
            if args.is_regression_only:
                token_logits, pred_values = outputs
                
                # 1. Token Loss
                loss_ce = F.cross_entropy(token_logits.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)
                # 2. Regression Loss (比较 pred_values 和 target_values)
                mask_num = masked_indices & (is_number > 0.5)
                if mask_num.any():
                    loss_val = F.smooth_l1_loss(pred_values[mask_num], target_values[mask_num])
                else:
                    loss_val = torch.tensor(0.0, device=DEVICE, requires_grad=True)

                loss = (loss_ce + args.mse_weight * loss_val) / GRAD_ACCUM_STEPS
                
                # 记录
                train_stats["loss"] += loss.item() * GRAD_ACCUM_STEPS
                train_stats["ce"] += loss_ce.item()
                train_stats["val_num"] += loss_val.item() # Regression Error
                
                loop.set_postfix({'Loss': f"{loss.item()*GRAD_ACCUM_STEPS:.3f}", 'CE': f"{loss_ce.item():.2f}", 'L1': f"{loss_val.item():.2f}"})
            else:
                token_logits, pred_mantissa_abs, exp_logits, sign_logits = outputs
                
                sign_targets = get_sign_labels(mantissa_labels)
                mantissa_targets_abs = torch.abs(mantissa_labels)

                # 1. Token Loss
                loss_ce = F.cross_entropy(token_logits.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)
                
                # 2. Numerical Losses
                mask_num = masked_indices & (is_number > 0.5)
                
                if mask_num.any():
                    loss_sign = F.cross_entropy(sign_logits[mask_num], sign_targets[mask_num])
                    
                    non_zero_mask = mask_num & (sign_targets != 1) # 排除 0
                    if non_zero_mask.any():
                        loss_mant = F.smooth_l1_loss(pred_mantissa_abs[non_zero_mask], mantissa_targets_abs[non_zero_mask])
                        
                        soft_targets = model.value_head.get_soft_labels(exponent_labels[non_zero_mask])
                        loss_exp = F.kl_div(F.log_softmax(exp_logits[non_zero_mask], dim=-1), soft_targets, reduction='batchmean')
                    else:
                        loss_mant = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                        loss_exp = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                else:
                    loss_sign = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                    loss_mant = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                    loss_exp = torch.tensor(0.0, device=DEVICE, requires_grad=True)

                loss = (loss_ce + loss_sign + loss_mant + loss_exp) / GRAD_ACCUM_STEPS

                # 记录
                train_stats["loss"] += loss.item() * GRAD_ACCUM_STEPS
                train_stats["ce"] += loss_ce.item()
                train_stats["sign"] += loss_sign.item()
                train_stats["mant"] += loss_mant.item()
                train_stats["exp"] += loss_exp.item()
                
                loop.set_postfix({'Loss': f"{loss.item()*GRAD_ACCUM_STEPS:.3f}", 'CE': f"{loss_ce.item():.2f}", 'Mnt': f"{loss_mant.item():.2f}", 'Exp': f"{loss_exp.item():.2f}", 'Sgn': f"{loss_sign.item():.2f}"})


            loss.backward()
            train_stats["steps"] += 1
            
            if global_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # --- Evaluation ---
            if (step + 1) % eval_interval == 0 or (step + 1) == total_steps_per_epoch:
                avg_t = {k: v / train_stats["steps"] for k, v in train_stats.items() if k != "steps"}
                
               # [CHANGE 3] 统一 History 记录逻辑
                history["train_steps"].append(global_step)
                history["train_loss"].append(avg_t["loss"])
                history["train_ce"].append(avg_t["ce"])

                if args.is_regression_only:
                    history["train_smooth_l1"].append(avg_t["val_num"])
                else:
                    history["train_mant"].append(avg_t["mant"])
                    history["train_exp"].append(avg_t["exp"])
                    history["train_sign"].append(avg_t["sign"])

                # [CHANGE] 解包5个返回值
                val_metrics = evaluate(args, model, val_loader, DEVICE, tokenizer, protected_ids_tensor)
                history["val_steps"].append(global_step)
                history["val_loss"].append(val_metrics["loss"])
                history["val_ce"].append(val_metrics["ce"])

                if args.is_regression_only:
                    history["val_smooth_l1"].append(val_metrics["val_num"])
                    
                    print(f"\n=== Step {global_step} ===")
                    print(f"Train | Loss: {avg_t['loss']:.4f} | CE: {avg_t['ce']:.4f} | L1: {avg_t['val_num']:.4f}")
                    print(f"Val   | Loss: {val_metrics['loss']:.4f} | CE: {val_metrics['ce']:.4f} | L1: {val_metrics['val_num']:.4f}")
                else:
                    history["val_mant"].append(val_metrics["val_num"]) # 对应 evaluate 中的 val_num
                    history["val_sign"].append(val_metrics["sign"])
                    history["val_exp"].append(val_metrics["exp"])
                    
                    print(f"\n=== Step {global_step} ===")
                    print(f"Train | Loss: {avg_t['loss']:.4f} | CE: {avg_t['ce']:.4f} | Sgn: {avg_t['sign']:.4f} | Mnt: {avg_t['mant']:.4f}")
                    print(f"Val   | Loss: {val_metrics['loss']:.4f} | CE: {val_metrics['ce']:.4f} | Sgn: {val_metrics['sign']:.4f} | Mnt: {val_metrics['val_num']:.4f}")

                print("-" * 80)

                plot_compare_curves(history, plot_configs=plot_configs, save_path=os.path.join(args.output_dir, "loss_curves.png"), max_cols=len(train_stats.keys()))

                checkpointer.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=global_step, 
                    epoch=epoch, 
                    val_loss=val_metrics["loss"],
                    is_best=val_metrics["loss"] < checkpointer.best_val_loss,
                    scheduler=scheduler
                )

                 # 重置统计
                if args.is_regression_only:
                    train_stats = {"loss": 0, "ce": 0, "val_num": 0, "steps": 0}
                else:
                    train_stats = {"loss": 0, "ce": 0, "mant": 0, "exp": 0, "sign": 0, "steps": 0} 
                model.train()

    print("Training Complete.")