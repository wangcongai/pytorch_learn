import torch
from x_clip import CLIP

if __name__ == '__main__':
    clip = CLIP(
        dim_text=512,
        dim_image=512,
        dim_latent=512,
        num_text_tokens=10000,
        text_enc_depth=6,
        text_seq_len=256,
        text_heads=8,
        visual_enc_depth=6,
        visual_image_size=256,
        visual_patch_size=32,
        visual_heads=8,
        visual_patch_dropout=0.5,
        # patch dropout probability, used in Kaiming He's FLIP to save compute and improve end results - 0.5 is good value, 0.75 on high end is tolerable
        use_all_token_embeds=True,  # whether to use fine-grained contrastive learning (FILIP)
        decoupled_contrastive_learning=True,
        # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
        extra_latent_projection=True,
        # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_visual_ssl=True,  # whether to do self supervised learning on images
        use_mlm=False,  # use masked language learning (MLM) on text (DeCLIP)
        text_ssl_loss_weight=0.05,  # weight for text MLM loss
        image_ssl_loss_weight=0.05  # weight for image self-supervised learning loss
    )

    # mock data
    text = torch.randint(0, 10000, (4, 256))
    images = torch.randn(4, 3, 256, 256)

    # train
    loss = clip(
        text,
        images,
        freeze_image_encoder=False,
        # whether to freeze image encoder if using a pretrained image net, proposed by LiT paper
        return_loss=True  # needs to be set to True to return contrastive loss
    )

    loss.backward()
