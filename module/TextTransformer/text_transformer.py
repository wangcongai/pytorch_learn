import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from contextlib import contextmanager


@contextmanager
def null_context():
    yield


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


class LayerNorm(nn.Module):
    # layernorm是对每个样本的所有位置特征做归一化，而batchnorm是对一个批次内的样本的同一位置特征做归一化。
    # 这样，layernorm可以更好地适应变长的数据，比如不同长度的句子，而batchnorm可能会受到padding的影响。
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        # 对embedding维度进行归一化
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        # mult为dim的扩增倍数 512*4=2048
        inner_dim = int(dim * mult)
        # feedforward的内部结构，一种可能的解释是，这样做可以增加模型的非线性能力和复杂度，从而提高模型的泛化能力和性能。
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, causal=False, dropout=0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h, device, scale = self.heads, x.device, self.scale
        # q, k, v: [4, 256, 512]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v: [4, 8, 256, 64]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        # sim [4, 8, 256, 256]
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max
        if exists(mask):
            # 这一行是对sim进行掩码填充的操作，使用了pytorch自带的函数masked_fill，
            # 它可以根据一个布尔型的掩码张量，将原张量中对应位置的值替换为指定的值。
            # 这里的掩码张量是~mask，表示对mask取反，也就是将True变为False，将False变为True。
            # 这样，原张量sim中mask为False的位置的值，就会被替换为mask_value，也就是一个很小的负数。
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        # transformer里使用causal_mask的条件是什么呢？一般来说，当模型需要进行自回归的生成任务时，就需要使用causal_mask，也就是因果掩码
        # 因果掩码的作用是防止模型在预测当前位置的输出时看到未来位置的输入，从而保证模型的因果性，即只能根据过去和当前的信息进行预测。
        if self.causal:
            i, j = sim.shape[-2:]
            # 掩码是上三角矩阵
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            # openAI的next token prediction，是不是肯定使用了causal_mask呢？
            # 是的，openAI的next token prediction，也就是下一个令牌预测，是一种典型的自回归的生成任务，
            # 它的目标是根据给定的文本序列，预测下一个最可能的令牌，比如单词或者字符。
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            dim_head=64,
            heads=8,
            causal=False,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            checkpoint_during_training=False
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        # 总共包括depth个residual blocks
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 在做Attention之前，先做Layer norm
                PreNorm(dim, Attention(dim=dim, dim_head=dim_head, heads=heads, causal=causal, dropout=attn_dropout)),
                # 在做FeedForward之前，先做Layer norm
                PreNorm(dim, FeedForward(dim=dim, mult=ff_mult)),
            ]))

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(
            self,
            x,
            rotary_pos_emb=None,
            mask=None
    ):

        checkpoint_fn = identity
        # layer norm
        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))
            # attention 多头自注意力层的输入是上一层的输出，或者是初始的嵌入向量。它的作用是计算输入序列中每个位置的特征与其他位置的特征的相关性，并输出一个新的序列特征
            x = attn(x, mask) + x
            # feedforward 前馈网络层的输入是多头自注意力层的输出，经过残差连接和层归一化后。它的作用是对每个位置的特征进行非线性变换，提高模型的表达能力
            # attention和feedforward是串行结构
            x = ff(x) + x

        return self.norm_out(x)


class TextTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            num_tokens,
            max_seq_len,
            dim_head,
            rotary_pos_emb=None,
            causal=False,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        self.rotary_pos_emb = None

        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        self.transformer = Transformer(dim, dim_head=dim_head, causal=causal, **kwargs)

    def forward(self, x, mask=None):
        b, n, device = *x.shape, x.device
        # token embedding: [4, 256] -> [4, 256, 512]
        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            # pos_emb: [256, 512]
            pos_emb = self.abs_pos_emb(torch.arange(n, device=device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

            if exists(mask):
                # mask [4, 256] -> [4, 257]
                mask = F.pad(mask, (1, 0), value=True)

        out = self.transformer(x, mask=mask, rotary_pos_emb=None)
        return out


def model_forward_with_context(*, fn, args, freeze, ):
    encoding_context = null_context if not freeze else torch.no_grad

    with encoding_context():
        enc = fn(*args)

        if freeze:
            # enc.detach_()是一个pytorch中的函数，它的作用是将enc从当前的计算图中分离出来，并且将enc的requires_grad属性设置为False，
            # 这样enc就不会再参与梯度的计算和反向传播，也不会有grad属性enc.detach_()是一个原地操作，它会直接修改enc本身
            enc.detach_()

    return enc


if __name__ == '__main__':
    # mock data bs=4, seq_len=256
    text = torch.randint(0, 10000, (4, 256))
    b, device = text.shape[0], text.device
    text_mask = text != 0
    text_args = (text,)
    text_args = (*text_args, text_mask)
    text_transformer = TextTransformer(dim=512,
                                       num_tokens=10000,
                                       max_seq_len=256,
                                       depth=6,
                                       heads=8,
                                       causal=False,
                                       dim_head=64,
                                       rotary_pos_emb=False,
                                       checkpoint_during_training=False
                                       )
    # 如果causal=False，enc_text [4, 257, 512]; 如果causal=True，enc_text [4, 256, 512]
    enc_text = model_forward_with_context(fn=text_transformer,
                                          args=text_args,
                                          freeze=False)
    print('finish')
