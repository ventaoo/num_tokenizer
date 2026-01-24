import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalNumberEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_abs=1e6):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(max_abs / 10.0))
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, L]
        scaled_x = x / self.scale.clamp(min=1e-6)  # 防止除零
        sinusoid_inp = torch.ger(scaled_x.view(-1), self.inv_freq.to(x.device))
        sinusoid_inp = sinusoid_inp.view(x.shape[0], x.shape[1], -1)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return self.proj(pos_emb)

class SoftExponentScientificHead(nn.Module):
    def __init__(self, hidden_dim, min_exp=-10, max_exp=10, sigma=1.0):
        super().__init__()
        self.min_exp = min_exp
        self.max_exp = max_exp
        self.num_bins = max_exp - min_exp + 1
        self.sigma = sigma

        # 将区间中心注册为 Buffer，会自动随模型 .to(device)
        # shape: [num_bins]
        self.register_buffer('bin_centers', torch.arange(min_exp, max_exp + 1, dtype=torch.float32))

        # 1. 指数分类头 (Exponent) -> 预测离散的指数分布
        self.exp_head = nn.Linear(hidden_dim, self.num_bins)
        
        # 2. 底数回归头 (Mantissa) -> 预测绝对值 [1, 10)
        # 逻辑: Sigmoid [0, 1] -> *9 -> [0, 9] -> +1 -> [1, 10]
        self.mantissa_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        ) 

        # 3. 符号分类头 (Sign) -> 3类: [负数(0), 零(1), 正数(2)]
        # 专门处理 0 的问题，解耦符号与数值大小
        self.sign_head = nn.Linear(hidden_dim, 3) 

    def forward(self, hidden):
        """
        hidden: [Batch, Len, Hidden_Dim]
        """
        # --- Exponent ---
        exp_logits = self.exp_head(hidden) # [B, L, num_bins]

        # --- Mantissa (Absolute Value) ---
        raw_mant = self.mantissa_head(hidden).squeeze(-1) # [B, L] within [0, 1]
        pred_mantissa_abs = 0.5 + 10.0 * raw_mant # [确保整数1, 10好学]

        # --- Sign / Zero ---
        sign_logits = self.sign_head(hidden) # [B, L, 3]

        return pred_mantissa_abs, exp_logits, sign_logits

    def get_soft_labels(self, target_exp):
        """
        生成高斯分布的 Soft Labels (完全向量化，无循环)
        
        Args:
            target_exp: 真实的指数张量, shape [N] (例如 flatten 后的 tensor)
        Returns:
            probs: 概率分布, shape [N, num_bins]
        """
        # 1. 准备数据形状进行广播
        # target: [N, 1]
        target = target_exp.float().unsqueeze(-1)
        
        # centers: [1, num_bins] (使用已注册的 buffer)
        centers = self.bin_centers.unsqueeze(0)
        
        # 2. 向量化计算高斯 Logits
        # [N, 1] - [1, K] -> [N, K] (自动广播)
        dist_sq = (centers - target) ** 2
        logits = -0.5 * (dist_sq / (self.sigma ** 2))
        
        # 3. 归一化为概率
        probs = F.softmax(logits, dim=-1)
        
        return probs

class SvgBert(nn.Module):
    def __init__(self, base_model, hidden, vocab_size, min_exp=-10, max_exp=10):
        super().__init__()
        self.bert = base_model
        self.num_input_proj = SinusoidalNumberEmbedding(hidden)
        
        self.token_head = nn.Linear(hidden, vocab_size)
        self.value_head = SoftExponentScientificHead(
            hidden, 
            min_exp=min_exp, 
            max_exp=max_exp
        )

    def forward(self, input_ids, attention_mask, num_values, is_number):
        inputs_embeds = self.bert.get_input_embeddings()(input_ids)
        num_embeds = self.num_input_proj(num_values.unsqueeze(-1))
        
        final_embeds = inputs_embeds + (is_number.unsqueeze(-1) * num_embeds)

        out = self.bert(inputs_embeds=final_embeds, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, L, H]
        
        token_logits = self.token_head(h)  # [B, L, V]
        pred_mantissa_abs, exp_logits, sign_logits = self.value_head(h)

        # 返回 4 个部分：Token预测, 底数绝对值, 指数分布, 符号分布
        return token_logits, pred_mantissa_abs, exp_logits, sign_logits