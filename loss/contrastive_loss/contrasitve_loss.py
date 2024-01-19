import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__ == '__main__':
    # 定义超参数
    N = 16  # batch_size
    T1 = 64  # 图片的token数量
    T2 = 128  # 文字的token数量
    D = 512  # token的维度
    T = 0.07  # 温度参数

    # 构造随机数据
    image_tokens = torch.randn(N, T1, D)  # [N, T1, 512]
    text_tokens = torch.randn(N, T2, D)  # [N, T2, 512]

    # 计算图片和文字的token level相似度矩阵
    sim_matrix = torch.einsum('nsd,ntd->nst', image_tokens, text_tokens)  # [N, T1, T2]

    # 逐行按列求Max，再对行求Mean，得到图片和文字的相似度
    sim_scores = sim_matrix.max(dim=-1)[0].mean(dim=-1)  # [N]

    # 组成相似度矩阵
    sim_matrix = torch.einsum('i,j->ij', sim_scores, sim_scores)  # [N, N]

    # 计算对比学习loss
    labels = torch.arange(N)  # [N]
    criterion = nn.CrossEntropyLoss()
    loss = criterion(sim_matrix / T, labels)
    print(loss)
