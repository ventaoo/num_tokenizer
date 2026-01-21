"""
vis_tool.py => 可视化，保存相关的工具函数
"""
import os
import glob
import torch
import matplotlib.pyplot as plt

class ModelCheckpointer:
    def __init__(self, output_dir, prefix="model", max_saved=5):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_saved = max_saved
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
    def save_checkpoint(self, model, step, epoch, val_loss=None, is_best=False):
        """
        保存检查点 (按 Step)
        :param step: 当前的全局步数 (global_step)
        :param epoch: 当前的 epoch (可选，用于记录元数据)
        """
        checkpoint_path = os.path.join(self.output_dir, f"{self.prefix}_step_{step}.pth")
        
        save_dict = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }
        
        torch.save(save_dict, checkpoint_path)
        
        print(f"检查点已保存: {checkpoint_path}" + 
              (f" (val_loss: {val_loss:.4f})" if val_loss is not None else ""))
        
        if is_best and val_loss is not None:
            best_path = f"{self.prefix}_best.pth"
            torch.save(save_dict, best_path)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = best_path
                print(f"🎉 New best model => val_loss: {val_loss:.4f}")
        
        self._cleanup_old_checkpoints()
    
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


def plot_compare_curves(history, save_path=None):
    train_x = history["train_steps"]
    val_x = history["val_steps"]
    
    plot_configs = [
        {
            'ax_idx': 0,
            'train_data': history["train_loss"],
            'val_data': history["val_loss"],
            'title': "Total Loss",
            'ylabel': "Loss",
            'train_color': 'blue',
            'val_color': 'dodgerblue'
        },
        {
            'ax_idx': 1,
            'train_data': history["train_ce"],
            'val_data': history["val_ce"],
            'title': "Cross Entropy Loss",
            'ylabel': "CE Loss",
            'train_color': 'green',
            'val_color': 'limegreen'
        },
        {
            'ax_idx': 2,
            'train_data': history["train_mse"],
            'val_data': history["val_mse"],
            'title': "MSE Loss (Transformed Space)",
            'ylabel': "MSE Loss",
            'train_color': 'red',
            'val_color': 'tomato'
        },
        {
            'ax_idx': 3,
            'train_data': history["train_ori_mse"],
            'val_data': history["val_ori_mse"],
            'title': "ORI MSE Loss (Original Space)",
            'ylabel': "MSE Loss",
            'train_color': 'purple',
            'val_color': 'violet'
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)
    axes = axes.flatten()
    
    def plot_metric(ax, train_data, val_data, title, ylabel, train_color='blue', val_color='pink'):
        ax.plot(train_x, train_data, '.-', label='Train', color=train_color, 
                markersize=5, linewidth=1.5)
        ax.plot(val_x, val_data, '.-', label='Val', color=val_color, alpha=0.5,
                markersize=5, linewidth=1.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if ax in [axes[2], axes[3]]:
            ax.set_xlabel("Global Steps", fontsize=10)
        
        all_data = train_data + val_data
        if all_data:
            y_min = min(all_data)
            y_max = max(all_data)
            if y_min != y_max:
                margin = (y_max - y_min) * 0.05
                ax.set_ylim(y_min - margin, y_max + margin)
    
    for config in plot_configs:
        ax = axes[config['ax_idx']]
        plot_metric(ax, 
                   config['train_data'], 
                   config['val_data'], 
                   config['title'], 
                   config['ylabel'],
                   config['train_color'],
                   config['val_color'])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.close(fig)