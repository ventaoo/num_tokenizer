import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogDistributionLoss(nn.Module):
    def __init__(self, head_module, sigma=1.0):
        """
        head_module: 传入你实例化好的 LogIntegralHead，我们需要它的 bins 信息
        sigma: 高斯分布的标准差 (在 Log 空间的宽度)
        """
        super().__init__()
        self.num_bins = head_module.num_bins
        # 获取 Head 里存好的 log_indices [num_bins]
        # 形状变换为 [1, 1, num_bins] 以便广播
        self.register_buffer('bin_log_centers', head_module.log_indices.view(1, 1, -1))
        self.sigma = sigma
        
    def forward(self, pred_logits, target_values, mask):
        """
        pred_logits: [B, L, num_bins]
        target_values: [B, L] 真实的物理数值
        mask: [B, L] 0/1 mask
        """
        # 1. 预处理 Target (防止 log(0))
        # 确保 target 至少比 min_val 大一点点
        safe_targets = torch.clamp(target_values, min=1e-5)
        log_targets = torch.log(safe_targets).unsqueeze(-1) # [B, L, 1]
        
        # 2. 动态生成对数空间的高斯分布 (Soft Labels)
        # 公式: exp( - (x - mu)^2 / 2sigma^2 )
        # x 是 bin的log中心，mu 是 target的log值
        dist = torch.exp( - (self.bin_log_centers - log_targets)**2 / (2 * self.sigma**2) )
        
        # 归一化，变成概率分布
        target_dist = dist / (dist.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 3. 计算 KL 散度 (简化版：CrossEntropy)
        # pred_logits 未经过 softmax，所以用 log_softmax
        log_probs = F.log_softmax(pred_logits, dim=-1)
        
        # Loss = - sum( Target * log(Pred) )
        loss_per_token = - (target_dist * log_probs).sum(dim=-1) # [B, L]
        
        # 4. Masking
        # 只计算 is_number == 1 的位置
        active_loss = (loss_per_token * mask).sum()
        num_active = mask.sum() + 1e-6
        
        return active_loss / num_active

class SinusoidalNumberEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        self.proj = nn.Linear(hidden_dim, hidden_dim) 
        
    def forward(self, x):
        sinusoid_inp = torch.ger(x.view(-1), self.inv_freq.to(x.device))
        sinusoid_inp = sinusoid_inp.view(x.shape[0], x.shape[1], -1)
        
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return self.proj(pos_emb)
    
class LogIntegralHead(nn.Module):
    def __init__(self, hidden_dim, num_bins=2000, min_val=1e-3, max_safe_val=1e8):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_bins)
        
        self.log_min = math.log(min_val)
        self.log_max = math.log(max_safe_val)
        self.num_bins = num_bins
        
        # 生成 Log 空间的均匀坐标
        log_indices = torch.linspace(self.log_min, self.log_max, num_bins)
        self.register_buffer('pos_indices', torch.exp(log_indices)) # 映射回物理空间用于积分预测
        self.register_buffer('log_indices', log_indices) # 同时也存一份 Log 坐标用于 Loss 计算
        
    def forward(self, x):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        pred_values = torch.sum(probs * self.pos_indices, dim=-1)
        
        return logits, pred_values

class SvgBert(nn.Module):
    def __init__(self, base_model, hidden, vocab_size):
        super().__init__()
        self.bert = base_model
        self.token_head = nn.Linear(hidden, vocab_size)  # CE по токенам
        
        self.value_head = LogIntegralHead(hidden)  # Регрессия по числам
        self.num_input_proj = SinusoidalNumberEmbedding(hidden)

    def forward(self, input_ids, attention_mask, num_values, is_number):
        inputs_embeds = self.bert.get_input_embeddings()(input_ids)
        num_embeds = self.num_input_proj(num_values.unsqueeze(-1))
        
        final_embeds = inputs_embeds + (is_number.unsqueeze(-1) * num_embeds)

        out = self.bert(inputs_embeds=final_embeds, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B,L,H]

        token_logits = self.token_head(h)               # [B,L,V]
        value_logits, value_pred = self.value_head(h)  # [B,L,num_bins], [B,L]

        return token_logits, value_pred, value_logits