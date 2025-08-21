import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# === 简化的 GPT 模型，跟之前代码相同 ===
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
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([SimpleGPTBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        
    def forward(self, idx):
        batch_size, seq_len = idx.size()
        token_embeddings = self.token_emb(idx)
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        pos_embeddings = self.pos_emb(positions)
        x = token_embeddings + pos_embeddings
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).bool()
        attn_mask = ~mask
        
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# === 训练参数 ===
vocab_size = 50        # 小词表，用于演示
embed_dim = 32
num_heads = 4
ff_hidden_dim = 128
num_layers = 2
max_seq_len = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleGPT(vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# === 构造简单数据集 ===
# 用数字表示词汇，比如句子：[1,2,3,4,5]
# 任务：根据前n个预测第n+1个
def generate_batch(batch_size, seq_len):
    # 随机生成 batch_size 个序列，每个长度 seq_len，数字范围在 [1, vocab_size-1]
    data = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    return data

# === 训练函数 ===
def train(model, optimizer, criterion, epochs=1000, batch_size=16, seq_len=10):
    model.train()
    for epoch in range(epochs):
        data = generate_batch(batch_size, seq_len)
        inputs = data[:, :-1]      # 输入序列，去掉最后一个 token
        targets = data[:, 1:]      # 目标序列，去掉第一个 token
        
        optimizer.zero_grad()
        logits = model(inputs)     # (batch, seq_len-1, vocab_size)
        loss = criterion(logits.view(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# === 简单生成函数 ===
@torch.no_grad()
def generate(model, start_token, max_len=20):
    model.eval()
    generated = [start_token]
    for _ in range(max_len-1):
        idx = torch.tensor(generated, device=device).unsqueeze(0)  # (1, seq_len)
        logits = model(idx)      # (1, seq_len, vocab_size)
        logits = logits[:, -1, :]  # 取最后一个时间步的预测
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        if next_token == 0:  # 0 可以当做结束标志
            break
    return generated

# === 主流程 ===
if __name__ == "__main__":
    print("开始训练...")
    train(model, optimizer, criterion, epochs=1000, batch_size=32, seq_len=10)
    
    print("训练完成，开始生成...")
    start_token = random.randint(1, vocab_size-1)
    generated_seq = generate(model, start_token)
    print("生成序列:", generated_seq)

