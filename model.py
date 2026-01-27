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
    
class MultiScaleInputEmbedding(nn.Module):
    def __init__(self, hidden_dim, scales=[10.0, 1.0, 0.1, 0.01]):
        """
        SVG 数值多尺度编码模块
        :param hidden_dim: 映射到的隐藏层维度 (BERT 的 hidden_size)
        :param scales: 缩放因子列表。
                       10.0 负责捕捉小数细节 (0.1 * 10 = 1)
                       0.01 负责捕捉大数全局 (100 * 0.01 = 1)
        """
        super().__init__()
        # 将 scales 注册为 buffer，这样它会随模型移动到 GPU，且不会被视作训练参数
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        
        self.input_dim = len(scales)
        self.hidden_dim = hidden_dim
        
        # 线性投影层：将多尺度特征融合
        self.proj = nn.Linear(self.input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        :param x: 输入张量，形状可以是 [Batch, SeqLen] 或 [Batch, SeqLen, 1]
        :return: 嵌入向量 [Batch, SeqLen, hidden_dim]
        """
        # 1. 确保输入是数值张量且维度正确 [B, L, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.float() # 确保数据类型为 float

        # 2. 多尺度变换
        # x: [B, L, 1] * scales: [S] -> [B, L, S]
        multi_scaled_x = x * self.scales

        # 3. 投影到隐藏层维度
        h = self.proj(multi_scaled_x)

        # 4. 规范化处理
        return self.norm(h)
    
class ValueRegressionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, h):
        """
        h: [B, L, hidden_dim] Transformer 的输出
        return: [B, L] 预测的数值（绝对值）
        """
        return self.net(h).squeeze(-1)

class SvgBert(nn.Module):
    def __init__(self, 
        base_model, 
        hidden, 
        vocab_size, 
        min_exp=-10, 
        max_exp=10, 
        is_multi_scale=False,
        is_regression_only=False
    ):
        super().__init__()
        self.bert = base_model
        self.num_input_proj = MultiScaleInputEmbedding(hidden) if is_multi_scale else SinusoidalNumberEmbedding(hidden)
        self.value_head = ValueRegressionHead(hidden) if is_regression_only else SoftExponentScientificHead(
            hidden, 
            min_exp=min_exp, 
            max_exp=max_exp
        )

        self.token_head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask, num_values, is_number):
        inputs_embeds = self.bert.get_input_embeddings()(input_ids)
        num_embeds = self.num_input_proj(num_values.unsqueeze(-1))
        
        final_embeds = inputs_embeds + (is_number.unsqueeze(-1) * num_embeds)

        out = self.bert(inputs_embeds=final_embeds, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, L, H]
        
        token_logits = self.token_head(h)  # [B, L, V]

        if isinstance(self.value_head, ValueRegressionHead):
            pred_values = self.value_head(h)  # [B, L]
            return token_logits, pred_values
        
        pred_mantissa_abs, exp_logits, sign_logits = self.value_head(h)
        return token_logits, pred_mantissa_abs, exp_logits, sign_logits