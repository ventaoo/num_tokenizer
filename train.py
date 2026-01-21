"""
train.py => train model [svgbert]
"""
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

from model import SvgBert, LogDistributionLoss
from data_tool import prepare_data
from vis_tool import ModelCheckpointer, plot_compare_curves

def evaluate(args, model, loader, device, tokenizer, protected_ids_tensor, dist_criterion):
    model.eval() # 切换到评估模式
    
    # 初始化统计器
    total_loss = 0
    total_ce = 0
    total_mse = 0
    steps = 0
    
    MASK_ID = tokenizer.mask_token_id
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            # 1. 数据加载
            input_ids = batch['input_ids'].to(device)
            is_number = batch['is_number'].to(device)
            num_values = batch['num_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 2. 准备标签
            labels = input_ids.clone()
            value_labels = num_values.clone()

            # 3. 生成 Mask (动态 Mask)
            # 注意：在验证集上做随机 Mask 是为了计算 MLM Loss。
            # 虽然这会导致每次验证的 Loss 有微小波动，但在样本量足够大时是可以接受的。
            # 如果追求严格一致性，可以在验证开始前设置固定的随机种子。
            probability_matrix = torch.full(input_ids.shape, 0.3, device=device)
            
            # 排除特殊 Token
            current_protected_mask = torch.isin(input_ids, protected_ids_tensor)
            probability_matrix.masked_fill_(current_protected_mask, value=0.0)
            
            # 生成 Mask 索引
            masked_indices = torch.bernoulli(probability_matrix).bool()

            # 应用 Mask
            input_ids[masked_indices] = MASK_ID
            num_values[masked_indices] = 0.0
            labels[~masked_indices] = -100 # CE Loss 忽略非 Mask 部分

            # 4. 模型前向传播
            token_logits, value_pred, value_logits = model(input_ids, attention_mask, num_values, is_number)

            # 5. 计算 Loss
            # --- Loss Calculation ---
            loss_ce = nn.functional.cross_entropy(token_logits.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)
            
            # --- MSE Loss for masked numbers ---
            mask_reg = masked_indices & (is_number > 0.5)
            loss_mse = nn.functional.smooth_l1_loss(value_pred[mask_reg], value_labels[mask_reg]) if mask_reg.any() else torch.tensor(0.0, device=DEVICE, requires_grad=True)

            # --- Log Distribution Loss for numbers ---
            mask_float = is_number.float() * masked_indices.float()
            loss_dist = dist_criterion(value_logits, num_values, mask_float)

            # 总 Loss
            loss = loss_ce + args.mse_weight * loss_mse + loss_dist
            
            # 6. 累积统计
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_mse += loss_mse.item()
            steps += 1

    # 切回训练模式 (可选，建议保留以防外层忘记切换)
    model.train()
    
    # 返回平均值 (只返回3个)
    return total_loss / steps, total_ce / steps, total_mse / steps

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
    parser.add_argument("--max_saved", type=int, default=2, help="最多保留几个最新的 Checkpoint")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="MSE loss 权重")
    
    # --- 数据路径 (建议添加，方便 Docker 挂载) ---
    parser.add_argument("--data_path", type=str, default="VectorGraphics/svg-corpus-private", help="数据集所在的文件夹路径")
    parser.add_argument("--data_size", type=int, default=10, help="抽取多少个数据块")

    # 2. 解析参数
    args = parser.parse_args()
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

    bert_base = AutoModel.from_pretrained(args.base_model)
    bert_base.resize_token_embeddings(len(tokenizer))
    
    model = SvgBert(bert_base, bert_base.config.hidden_size, len(tokenizer))
    model.to(DEVICE)
    
    dist_criterion = LogDistributionLoss(model.value_head, sigma=2.0) 
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # --- 2. 准备数据 ---
    train_loader, val_loader = prepare_data(args, tokenizer)
    total_steps_per_epoch = len(train_loader)
    
    # 定义特殊 Token (用于 Mask 逻辑)
    MASK_ID = tokenizer.mask_token_id
    protected_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id, MASK_ID}
    protected_ids_tensor = torch.tensor([tid for tid in protected_ids if tid is not None], device=DEVICE)

    # --- 3. 训练记录初始化 (已移除 ori_mse) ---
    history = {
        "train_loss": [], "train_ce": [], "train_mse": [], "train_steps": [],
        "val_loss": [],   "val_ce": [],   "val_mse": [],   "val_steps": []
    }
    
    # 绘图配置 (适配上一轮提供的 plot_compare_curves 函数)
    plot_configs = [
        {'train_data': history["train_loss"], 'val_data': history["val_loss"], 'title': "Total Loss", 'ylabel': "Loss", 'train_color': 'blue', 'val_color': 'dodgerblue', 'use_log': True},
        {'train_data': history["train_ce"],   'val_data': history["val_ce"],   'title': "CE Loss",    'ylabel': "CE",   'train_color': 'green', 'val_color': 'limegreen'},
        {'train_data': history["train_mse"],  'val_data': history["val_mse"],  'title': "MSE Loss",   'ylabel': "MSE",  'train_color': 'red',   'val_color': 'tomato', 'use_log': True}
    ]

    print("Start Training...")
    
    # 定义验证频率 (例如：每个 epoch 验证 5 次)
    eval_interval = max(1, total_steps_per_epoch // 5) 
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_stats = {"loss": 0, "ce": 0, "mse": 0, "steps": 0} # 临时统计容器
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for step, batch in enumerate(loop):
            global_step += 1
            
            # --- 数据准备 ---
            input_ids = batch['input_ids'].to(DEVICE)
            is_number = batch['is_number'].to(DEVICE)
            num_values = batch['num_values'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            labels = input_ids.clone()
            value_labels = num_values.clone()

            # --- Masking 逻辑 (简化写法) ---
            prob_matrix = torch.full(input_ids.shape, 0.3, device=DEVICE)
            prob_matrix.masked_fill_(torch.isin(input_ids, protected_ids_tensor), 0.0)
            masked_indices = torch.bernoulli(prob_matrix).bool()

            input_ids[masked_indices] = MASK_ID
            num_values[masked_indices] = 0.0
            labels[~masked_indices] = -100 # Ignore non-masked tokens in Loss

            # --- Forward ---
            token_logits, value_pred, value_logits = model(input_ids, attention_mask, num_values, is_number)

            # --- Loss Calculation ---
            loss_ce = nn.functional.cross_entropy(token_logits.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)
            
            # --- MSE Loss for masked numbers ---
            mask_reg = masked_indices & (is_number > 0.5)
            loss_mse = nn.functional.smooth_l1_loss(value_pred[mask_reg], value_labels[mask_reg]) if mask_reg.any() else torch.tensor(0.0, device=DEVICE, requires_grad=True)

            # --- Log Distribution Loss for numbers ---
            mask_float = is_number.float() * masked_indices.float()
            loss_dist = dist_criterion(value_logits, num_values, mask_float)

            loss = (loss_ce + args.mse_weight * loss_mse + loss_dist) / GRAD_ACCUM_STEPS
            loss.backward()

            # --- 统计累积 (还原 loss 大小用于显示) ---
            train_stats["loss"] += loss.item() * GRAD_ACCUM_STEPS
            train_stats["ce"] += loss_ce.item()
            train_stats["mse"] += loss_mse.item()
            train_stats["steps"] += 1
            
            # --- Optimizer Step ---
            if global_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            loop.set_postfix({'Loss': f"{loss.item()*GRAD_ACCUM_STEPS:.3f}", 'CE': f"{loss_ce.item():.3f}", 'MSE': f"{loss_mse.item():.3f}", "Dist": f"{loss_dist.item():.3f}"})

            # --- Evaluation & Checkpoint Block (统一处理) ---
            if (step + 1) % eval_interval == 0 or (step + 1) == total_steps_per_epoch:
                # 1. 计算当前阶段的平均训练 Loss
                avg_t = {k: v / train_stats["steps"] for k, v in train_stats.items() if k != "steps"}
                
                # 2. 更新训练历史
                history["train_loss"].append(avg_t["loss"])
                history["train_ce"].append(avg_t["ce"])
                history["train_mse"].append(avg_t["mse"])
                history["train_steps"].append(global_step)

                # 3. 运行验证集
                val_metrics = evaluate(args, model, val_loader, DEVICE, tokenizer, protected_ids_tensor, dist_criterion)
                # 假设 evaluate 返回 (loss, ce, mse) 元组，已去除 ori_mse
                val_avg_loss, val_avg_ce, val_avg_mse = val_metrics 

                # 4. 更新验证历史
                history["val_loss"].append(val_avg_loss)
                history["val_ce"].append(val_avg_ce)
                history["val_mse"].append(val_avg_mse)
                history["val_steps"].append(global_step)

                print(f"\nStep {global_step} | Train: {avg_t['loss']:.4f} | Val: {val_avg_loss:.4f}")

                # 5. 绘图 (传入上一个问题优化的函数)
                # 注意：这里我们传入 plot_configs，函数会自动根据 config 数量绘制 3 张图
                plot_compare_curves(history, plot_configs=plot_configs, save_path="comparison_curves.png")

                # 6. 保存模型 (合并在一起，逻辑更顺)
                checkpointer.save_checkpoint(
                    model=model,
                    step=global_step, 
                    epoch=epoch + 1, 
                    val_loss=val_avg_loss,
                    is_best=val_avg_loss < checkpointer.best_val_loss
                )
                
                # 7. 重置统计
                train_stats = {"loss": 0, "ce": 0, "mse": 0, "steps": 0}
                model.train() # 确保 evaluate 后切回 train 模式

    print("Training Complete.")