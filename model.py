import torch.nn as nn

class SvgBert(nn.Module):
    def __init__(self, base_model, hidden, vocab_size):
        super().__init__()
        self.bert = base_model
        self.token_head = nn.Linear(hidden, vocab_size)  # CE по токенам
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, 1)
        )

        self.num_input_proj = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask, num_values, is_number):
        inputs_embeds = self.bert.get_input_embeddings()(input_ids)
        num_embeds = self.num_input_proj(num_values.unsqueeze(-1))
        
        final_embeds = inputs_embeds + (is_number.unsqueeze(-1) * num_embeds)

        out = self.bert(inputs_embeds=final_embeds, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B,L,H]

        token_logits = self.token_head(h)               # [B,L,V]
        value_pred = self.value_head(h).squeeze(-1)     # [B,L]

        return token_logits, value_pred