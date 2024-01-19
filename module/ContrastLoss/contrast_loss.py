import torch
from torch import einsum
from einops import rearrange, reduce


def log(t, eps=1e-20):
    return torch.log(t + eps)


def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d=num_diag_el)


def max_neg_value(dtype):
    return -torch.finfo(dtype).max


def masked_mean(t, mask, dim=1, eps=1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim=dim)
    denom = mask.sum(dim=dim).clamp(min=eps)
    return numer / denom


if __name__ == '__main__':
    use_all_token_embeds = True
    # 定义超参数
    m = 4  # batches of text
    n = 4  # batches of images
    t = 32  # sequence dimension along text tokens
    i = 64  # sequence dimension along image tokens
    d = 512  # token的维度
    temp = 0.07  # 温度参数

    # 构造随机数据
    text = torch.randint(0, 10000, (m, t))
    text_mask = text != 0
    text_latents = torch.randn(1, m, t, d)
    image_latents = torch.randn(1, n, i, d)

    if use_all_token_embeds:
        # fine-grained CLIP logic
        sim_text_to_image = einsum('m x t d, n y i d -> m n x y t i', text_latents, image_latents) * temp

        sim_image_to_text = sim_text_to_image

        text_to_image = reduce(sim_text_to_image, '... t i -> ... t', 'max')
        text_to_image_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t', m=1)
        text_to_image = masked_mean(text_to_image, text_to_image_mask, dim=-1)

        image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m=1)
        masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
        image_to_text = reduce(reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')
    else:
        text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp
        image_to_text = rearrange(text_to_image, '... t i -> ... i t')

    # calculate loss
    text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
    image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

    # exponentiate
    text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

    # numerators
    text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

    # denominator
    text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim=-1), (text_to_image_exp, image_to_text_exp))

    # loss
    text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim=-1)
    image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(dim=-1)

    # calculate CL loss
    cl_losses = (text_to_image_loss + image_to_text_loss) / 2
