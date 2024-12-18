import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# 因果自注意力机制的实现

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 确保嵌入维度可以被头数整除
        assert config.n_embd % config.n_head == 0
        # 为所有头生成键（K）、查询（Q）、值（V）的线性投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出的线性投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 注意力头的数量
        self.n_head = config.n_head
        # 嵌入维度
        self.n_embd = config.n_embd
        # 注册一个下三角遮罩，用于因果性（防止未来信息泄露）
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # 批次大小，序列长度，嵌入维度
        # 计算所有头的查询、键、值，并将头部维度前移
        # nh为头数，hs为每个头的维度，C = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 重塑并转置以适应多头注意力的计算
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # 计算缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 应用因果遮罩，确保只关注当前及之前的位置
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # 计算注意力权重
        att = F.softmax(att, dim=-1)
        # 计算注意力输出
        y = att @ v  # (B, nh, T, hs)
        # 重组所有头的输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        # 输出线性投影
        y = self.c_proj(y)
        return y

# -----------------------------------------------------------------------------
# 前馈神经网络（MLP）的实现

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 第一层线性变换，将嵌入维度扩展到4倍
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU激活函数
        self.gelu = nn.GELU(approximate='tanh')
        # 第二层线性变换，将维度缩回原始大小
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# -----------------------------------------------------------------------------
# Transformer块的实现，将自注意力和MLP结合

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 第一层归一化
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # 自注意力层
        self.attn = CausalSelfAttention(config)
        # 第二层归一化
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # 前馈神经网络
        self.mlp = MLP(config)

    def forward(self, x):
        # 残差连接与自注意力
        x = x + self.attn(self.ln_1(x))
        # 残差连接与前馈神经网络
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# GPT模型的配置类，使用dataclass定义

@dataclass
class GPTConfig:
    block_size: int = 1024  # 最大序列长度
    vocab_size: int = 50257  # 词汇表大小
    n_layer: int = 12  # Transformer层数
    n_head: int = 12  # 注意力头数
    n_embd: int = 768  # 嵌入维度

# -----------------------------------------------------------------------------
# GPT模型主体的实现

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 定义Transformer模块
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入
            wpe = nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 多层Transformer块
            ln_f = nn.LayerNorm(config.n_embd),  # 最终层归一化
        ))
        # 语言模型头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx的形状为 (B, T)
        B, T = idx.size()
        # 确保序列长度不超过块大小
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # 前向传播词嵌入和位置嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # 形状 (T)
        pos_emb = self.transformer.wpe(pos)  # 位置嵌入，形状 (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # 词嵌入，形状 (B, T, n_embd)
        x = tok_emb + pos_emb  # 将词嵌入与位置嵌入相加
        # 通过所有Transformer块
        for block in self.transformer.h:
            x = block(x)
        # 最终的层归一化
        x = self.transformer.ln_f(x)
        # 通过语言模型头获取logits
        logits = self.lm_head(x)  # 形状 (B, T, vocab_size)
        loss = None
        if targets is not None:
            # 计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """从Hugging Face加载预训练的GPT-2模型权重"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 根据模型类型确定层数、头数和嵌入维度
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),    # 124M 参数
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),   # 350M 参数
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M 参数
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M 参数
        }[model_type]
        config_args['vocab_size'] = 50257  # GPT模型检查点的词汇表大小
        config_args['block_size'] = 1024   # GPT模型检查点的块大小
        # 创建一个从头初始化的GPT模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # 过滤掉不需要的参数
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 初始化一个Hugging Face的GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 过滤Hugging Face模型中的缓冲区参数
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        # 需要转置的权重名称
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 确保参数键数量匹配
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的权重进行转置操作
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 直接复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------
# 设备检测与模型加载

# 尝试自动检测设备
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"使用设备: {device}")

# -----------------------------------------------------------------------------
# 数据加载与训练循环

# 获取一个数据批次
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]  # 截取前1000个字符
tokens = enc.encode(text)  # 对文本进行编码
B, T = 4, 32  # 批次大小和序列长度
buf = torch.tensor(tokens[:B*T + 1])  # 创建缓冲区，包含B*T+1个token
buf = buf.to(device)
x = buf[:-1].view(B, T)  # 输入序列，形状 (B, T)
y = buf[1:].view(B, T)   # 目标序列，形状 (B, T)

# 初始化模型并移动到设备
model = GPT(GPTConfig())
model.to(device)

# 优化器的设置
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 训练循环，进行50步优化
for i in range(50):
    optimizer.zero_grad()          # 清空梯度
    logits, loss = model(x, y)     # 前向传播，获取logits和损失
    loss.backward()                # 反向传播，计算梯度
    optimizer.step()               # 更新模型参数
    print(f"step {i}, loss: {loss.item()}")  # 打印当前步数和损失

# 训练完成后退出程序，防止后续生成代码执行
import sys; sys.exit(0)

# -----------------------------------------------------------------------------
# 文本生成部分（当前不可达，由于sys.exit(0)）

# 前缀token的处理
model.eval()
num_return_sequences = 5  # 生成的序列数量
max_length = 30          # 生成的最大长度
tokens = enc.encode("Hello, I'm a language model,")  # 编码前缀文本
tokens = torch.tensor(tokens, dtype=torch.long)  # 转换为张量，形状 (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # 扩展为 (5, 8)
x = tokens.to(device)

# 生成文本
torch.manual_seed(42)          # 设置随机种子
torch.cuda.manual_seed(42)     # 设置CUDA随机种子
while x.size(1) < max_length:
    # 前向传播获取logits
    with torch.no_grad():
        logits = model(x)[0]   # 获取logits，形状 (B, T, vocab_size)
        # 获取最后一个时间步的logits
        logits = logits[:, -1, :]  # 形状 (B, vocab_size)
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        # 进行top-k采样，k=50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # 从top-k中采样一个token
        ix = torch.multinomial(topk_probs, 1)  # 形状 (B, 1)
        # 获取对应的token索引
        xcol = torch.gather(topk_indices, -1, ix)  # 形状 (B, 1)
        # 将新token追加到序列中
        x = torch.cat((x, xcol), dim=1)

# 打印生成的文本
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
