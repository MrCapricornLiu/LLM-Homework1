import os
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制模块，实现了GPT-2中的多头自注意力机制。
    """

    def __init__(self, config):
        super().__init__()
        # 确保嵌入维度能被头数整除
        assert config.n_embd % config.n_head == 0
        # 为所有头的键、查询、值进行线性变换，输出维度为3倍嵌入维度
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出线性变换
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 头数和嵌入维度
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 注册下三角掩码，用于因果注意力（防止信息泄露）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # B: batch size, T: 序列长度, C: 嵌入维度
        # 计算查询、键、值的线性变换
        qkv = self.c_attn(x)
        # 将qkv按嵌入维度拆分为查询、键、值
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 重塑并转置以适应多头注意力的计算
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # 计算注意力得分
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 应用下三角掩码，防止未来信息泄露
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # 归一化注意力得分
        att = F.softmax(att, dim=-1)
        # 加权求和得到注意力输出
        y = att @ v  # (B, nh, T, hs)
        # 重塑并合并所有头的输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        # 输出线性变换
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """
    多层感知机模块，作为Transformer块中的前馈网络。
    """

    def __init__(self, config):
        super().__init__()
        # 第一层线性变换，将嵌入维度扩展到4倍
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU激活函数
        self.gelu = nn.GELU(approximate="tanh")
        # 第二层线性变换，将维度恢复到原始嵌入维度
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """
    Transformer块，由层归一化、自注意力和前馈网络组成。
    """

    def __init__(self, config):
        super().__init__()
        # 第一层归一化
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # 自注意力模块
        self.attn = CausalSelfAttention(config)
        # 第二层归一化
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # 前馈网络模块
        self.mlp = MLP(config)

    def forward(self, x):
        # 自注意力子层与残差连接
        x = x + self.attn(self.ln_1(x))
        # 前馈网络子层与残差连接
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """
    GPT模型的配置参数。
    """
    block_size: int = 1024  # 最大序列长度
    vocab_size: int = 50257  # 词汇表大小
    n_layer: int = 12  # Transformer块的数量
    n_head: int = 12  # 注意力头的数量
    n_embd: int = 768  # 嵌入维度

class GPT(nn.Module):
    """
    GPT模型的主体，实现了GPT-2的结构。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 定义Transformer的各个组件
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入
                wpe=nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer块列表
                ln_f=nn.LayerNorm(config.n_embd),  # 最后的层归一化
            )
        )
        # 语言模型头，用于生成词汇表大小的输出
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        """
        前向传播过程。
        :param idx: 输入的token索引，形状为 (B, T)
        :return: logits，形状为 (B, T, vocab_size)
        """
        B, T = idx.size()
        # 确保序列长度不超过最大块大小
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # 生成位置索引
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # 形状 (T,)
        pos_emb = self.transformer.wpe(pos)  # 位置嵌入，形状 (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # 词嵌入，形状 (B, T, n_embd)
        # 将词嵌入与位置嵌入相加
        x = tok_emb + pos_emb
        # 通过所有Transformer块
        for block in self.transformer.h:
            x = block(x)
        # 最后的层归一化
        x = self.transformer.ln_f(x)
        # 通过语言模型头得到logits
        logits = self.lm_head(x)  # 形状 (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """
        从预训练的GPT-2模型加载权重。
        :param model_type: 模型类型，如 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        :return: 加载了预训练权重的GPT模型
        """
        assert model_type in {
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
        }, "Unsupported model type"
        from transformers import GPT2LMHeadModel

        print("加载预训练的GPT-2模型权重: %s" % model_type)

        # 根据模型类型设置配置参数
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M参数
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M参数
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M参数
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M参数
        }[model_type]
        config_args["vocab_size"] = 50257  # GPT模型的词汇表大小固定为50257
        config_args["block_size"] = 1024  # GPT模型的块大小固定为1024
        # 创建一个从头初始化的GPT模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # 排除掩码相关的键
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # 加载HuggingFace的GPT-2模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 过滤HuggingFace模型的状态字典，排除掩码相关的键
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k
            for k in sd_keys_hf
            if not k.endswith(".attn.masked_bias")
            and not k.endswith(".attn.bias")
        ]
        # 需要转置的权重名称
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # 确保两个状态字典的键数量一致
        assert (
            len(sd_keys_hf) == len(sd_keys)
        ), f"状态字典键数量不匹配: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的权重进行转置操作
                assert (
                    sd_hf[k].shape[::-1] == sd[k].shape
                ), f"形状不匹配: {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 直接复制其他权重
                assert (
                    sd_hf[k].shape == sd[k].shape
                ), f"形状不匹配: {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------
# 自动检测设备（CPU、CUDA、MPS）
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"使用设备: {device}")

num_return_sequences = 5  # 生成序列的数量
max_length = 30  # 生成的最大长度

# 初始化GPT模型，可以选择加载预训练权重
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()  # 设置为评估模式
model.to(device)  # 将模型移动到指定设备

# 词嵌入编码
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")  # 编码前缀文本
tokens = torch.tensor(tokens, dtype=torch.long)  # 转换为张量，形状为 (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # 复制为 (5, 8)
x = tokens.to(device)  # 将输入移动到指定设备

# 生成文本
torch.manual_seed(42)  # 设置随机种子以确保结果可复现
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # 前向传播获取logits
    with torch.no_grad():
        logits = model(x)  # 形状为 (B, T, vocab_size)
        # 获取最后一个时间步的logits
        logits = logits[:, -1, :]  # 形状为 (B, vocab_size)
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        # 进行top-k采样，这里k=50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # 从top-k概率中进行多项式采样
        ix = torch.multinomial(topk_probs, 1)  # 形状为 (B, 1)
        # 获取对应的token索引
        xcol = torch.gather(topk_indices, -1, ix)  # 形状为 (B, 1)
        # 将新生成的token添加到输入序列中
        x = torch.cat((x, xcol), dim=1)

# 打印生成的文本
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
