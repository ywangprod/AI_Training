import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, attn_mask=None):
        # Multi-Head Self-Attention with masking (causal)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + attn_out)  # 残差连接 + LayerNorm
        
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)    # 残差连接 + LayerNorm
        return x

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)    # token embedding
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)     # 位置编码
        self.layers = nn.ModuleList([SimpleGPTBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)  # 输出预测
        
        self.max_seq_len = max_seq_len
        
    def forward(self, idx):
        batch_size, seq_len = idx.size()
        
        # 词向量 + 位置向量
        token_embeddings = self.token_emb(idx)          # (batch, seq_len, embed_dim)
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)  # (1, seq_len)
        pos_embeddings = self.pos_emb(positions)        # (1, seq_len, embed_dim)
        x = token_embeddings + pos_embeddings           
        
        # 生成 causal mask 保证当前 token 只能关注前面的 token
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).bool()
        attn_mask = ~mask  # PyTorch MultiheadAttention 里，True 表示**屏蔽**
        
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
            
        x = self.ln_f(x)        
        logits = self.head(x)  # (batch, seq_len, vocab_size)
        return logits

# --------------------
# 测试模型
# --------------------

vocab_size = 1000      # 假设词表大小 1000
embed_dim = 64         # 词向量维度
num_heads = 4          # 多头数量
ff_hidden_dim = 256    # FeedForward 隐藏层维度
num_layers = 2         # Transformer 层数（简化）
max_seq_len = 16       # 最大序列长度

model = SimpleGPT(vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len)

# 模拟输入：batch=2，序列长度=10，随机token id
input_ids = torch.randint(0, vocab_size, (2, 10))

# 前向计算
logits = model(input_ids)

print("输入形状:", input_ids.shape)       # torch.Size([2, 10])
print("输出形状:", logits.shape)          # torch.Size([2, 10, 1000])

