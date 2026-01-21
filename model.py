import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        log_min = math.log(min_val)
        log_max = math.log(max_safe_val)
        
        log_indices = torch.linspace(log_min, log_max, num_bins)
        self.register_buffer('pos_indices', torch.exp(log_indices))
        
    def forward(self, x):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        
        # 2. 积分求期望
        pred_values = torch.sum(probs * self.pos_indices, dim=-1)
        return pred_values

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
        value_pred = self.value_head(h).squeeze(-1)     # [B,L]

        return token_logits, value_pred