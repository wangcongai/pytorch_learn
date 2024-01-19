import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from torch_models.module.TextTransformer.text_transformer import Transformer, model_forward_with_context


class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x, force_keep_all=False):
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        b, n, _, device = x.shape[0], x.shape[1], x.shape[2], x.device
        # 从x的第2个维度n里，随机挑选一部分出来
        batch_indices = torch.arange(b, device=device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device=device).topk(num_patches_keep, dim=-1).indices
        # 第二个维度patch_indices_keep会改变原来x的第二个维度上的顺序
        return x[batch_indices, patch_indices_keep]


class VisionTransformer(nn.Module):
    def __init__(self,
                 dim,
                 *,
                 image_size,
                 patch_size,
                 channels,
                 patch_dropout=0.5,
                 **kwargs
                 ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        # 图片token化
        # 将输入x [4, 3, 256, 256] -> [4, 8*8, 32*32*3] -> [4, 64, 512]
        self.to_tokens = nn.Sequential(
            # Rearrange 返回的是nn.module
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim))

        self.pos_emb = nn.Embedding(num_patches, dim)
        self.patch_dropout = PatchDropout(patch_dropout)

        self.transformer = Transformer(dim, **kwargs)

        self.to_cls_tokens = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim, bias=False),
            Rearrange('b d -> b 1 d')
        )

    def forward(self, x, keep_all_patches=False):
        device = x.device

        x = self.to_tokens(x)
        b, n, _ = x.shape
        # pos_emb [64, 512]
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        # rearrange返回的是张量
        x = x + rearrange(pos_emb, 'n d -> 1 n d')
        # patch drop out这一步是在position embedding之后做的，位置信息已经编码进去了
        x = self.patch_dropout(x, force_keep_all=keep_all_patches)
        # x [4, 32, 512] -> out [4, 32, 512]
        out = self.transformer(x)
        # cls_tokens [4, 1, 512]
        cls_tokens = self.to_cls_tokens(out)
        return torch.cat((cls_tokens, out), dim=1)


if __name__ == '__main__':
    images = torch.randn(4, 3, 256, 256)
    visual_transformer = VisionTransformer(dim=512,
                                           image_size=256,
                                           patch_size=32,
                                           channels=3,
                                           depth=6,
                                           heads=8,
                                           dim_head=64,
                                           patch_dropout=0.5,
                                           checkpoint_during_training=False
                                           )
    # enc_image [4, 33, 512]
    enc_image = model_forward_with_context(fn=visual_transformer,
                                           args=(images,),
                                           freeze=False
                                           )
    print('finish')
