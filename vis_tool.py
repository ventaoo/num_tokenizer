"""
vis_tool.py => 可视化，保存相关的工具函数
"""
import os
import math
import glob
import torch
import matplotlib.pyplot as plt

class ModelCheckpointer:
    def __init__(self, output_dir, prefix="model", max_saved=5):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_saved = max_saved
        self.best_val_loss = float('inf')
        
    def save_checkpoint(self, model, optimizer, step, epoch, val_loss=None, is_best=False, scheduler=None):
        """
        [修改] 增加了 optimizer 和 scheduler 参数
        """
        checkpoint_path = os.path.join(self.output_dir, f"{self.prefix}_step_{step}.pth")
        
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        save_dict = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), # [新增] 保存优化器状态
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss, # [新增] 记录截止目前的历史最佳 Loss (用于恢复)
        }
        
        # [新增] 如果有学习率调度器，也保存
        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(save_dict, checkpoint_path)
        
        print(f"检查点已保存: {checkpoint_path}" + 
              (f" (val_loss: {val_loss:.4f})" if val_loss is not None else ""))
        
        if is_best and val_loss is not None:
            best_path = os.path.join(self.output_dir, f"{self.prefix}_best.pth")
            torch.save(save_dict, best_path)
            print(f"🎉 New best model updated => val_loss: {val_loss:.4f}")
        
        self._cleanup_old_checkpoints()

    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """
        [新增] 加载检查点用于恢复训练
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        print(f"正在加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 1. 恢复模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. 恢复优化器状态 (如果是恢复训练，这步很关键)
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # 3. 恢复调度器 (如果有)
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 4. best loss
        saved_best = checkpoint.get('best_val_loss', float('inf'))
        if saved_best < self.best_val_loss:
            self.best_val_loss = saved_best
            
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('step', 0)
        
        print(f"恢复成功! 从 Epoch {start_epoch}, Step {global_step} 开始")
        return start_epoch, global_step
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点"""
        checkpoint_files = glob.glob(os.path.join(self.output_dir, f"{self.prefix}_step_*.pth"))
        
        if len(checkpoint_files) <= self.max_saved:
            return
        
        checkpoint_files.sort(key=os.path.getmtime) 
        
        files_to_keep = checkpoint_files[-self.max_saved+1:]  # 最新的 N-1 个
        files_to_keep.append(checkpoint_files[0])             # 保留最老的一个 (Start point)
        
        for f in checkpoint_files:
            if f not in files_to_keep:
                try:
                    os.remove(f)
                    print(f"清理旧检查点: {os.path.basename(f)}")
                except OSError as e:
                    print(f"删除文件失败 {f}: {e}")

def draw_metric_subplot(ax, train_x, val_x, config):
    train_data = config.get('train_data', [])
    val_data = config.get('val_data', [])
    
    # 颜色默认值
    t_color = config.get('train_color', 'blue')
    v_color = config.get('val_color', 'orange')
    
    # 绘制曲线
    if train_data:
        ax.plot(train_x, train_data, '.-', label='Train', color=t_color, 
                markersize=5, linewidth=1.5)
    if val_data:
        ax.plot(val_x, val_data, '.-', label='Val', color=v_color, alpha=0.7,
                markersize=5, linewidth=1.5)

    # 设置标题和标签
    ax.set_title(config.get('title', ''), fontsize=12, fontweight='bold')
    ax.set_ylabel(config.get('ylabel', ''), fontsize=10)
    ax.set_xlabel("Global Steps", fontsize=10)
    
    # 图例与网格
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)

    # 处理对数坐标
    if config.get('use_log', False):
        ax.set_yscale('log')
    else:
        all_data = []
        if train_data: all_data.extend(train_data)
        if val_data: all_data.extend(val_data)
        
        if all_data:
            y_min, y_max = min(all_data), max(all_data)
            if y_min != y_max:
                margin = (y_max - y_min) * 0.05
                ax.set_ylim(y_min - margin, y_max + margin)

def plot_compare_curves(history, plot_configs, save_path=None, max_cols=4):
    """
    主绘图函数
    :param history: 包含训练数据的字典
    :param plot_configs: 配置列表，如果为 None 则使用默认配置
    :param save_path: 保存路径
    :param max_cols: 网格布局的最大列数
    """
    train_x = history.get("train_steps", [])
    val_x = history.get("val_steps", [])

    # 1. 如果没有传入配置，则定义默认配置 (兼容旧代码逻辑)

    num_plots = len(plot_configs)
    if num_plots == 0:
        print("没有可绘制的数据配置。")
        return

    # 2. 自动计算行数和列数
    # 如果只有1张图，则 1x1；如果有3张且 max_cols=2，则 2x2
    ncols = min(num_plots, max_cols)
    nrows = math.ceil(num_plots / ncols)

    # 3. 初始化画布
    # 动态调整高度：每行大约给 4-5 inches
    fig_width = 6 * ncols
    fig_height = 5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=False)
    
    # 确保 axes 总是可迭代的扁平数组（即使只有1个子图）
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 4. 循环绘制
    for idx, config in enumerate(plot_configs):
        ax = axes[idx]
        draw_metric_subplot(ax, train_x, val_x, config)

    # 5. 隐藏多余的空坐标轴 (例如 2x2 网格只有 3 张图时)
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    # 6. 保存逻辑
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close(fig)