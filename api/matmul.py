import torch


if __name__ == '__main__':
    # 两个一维向量的点积
    a = torch.tensor([1, 2])
    b = torch.tensor([3, 4])
    # 等价于 torch.dot(a, b)
    c = torch.matmul(a, b)
    print(c)  # tensor(11)
    d = torch.dot(a, b)
    print(d)

    # 两个二维矩阵的矩阵乘积
    a = torch.rand(2, 3)
    b = torch.rand(3, 4)
    # 等价于 torch.mm(a, b)
    c = torch.matmul(a, b)
    print(c.shape)  # torch.Size([2, 4])
    d = torch.mm(a, b)
    print(d.shape)

    # 一个一维向量和一个二维矩阵的矩阵乘积
    a = torch.rand(3)
    b = torch.rand(3, 4)
    # 等价于 torch.mv(b.T, a)
    c = torch.matmul(a, b)
    print(c.shape)  # torch.Size([4])
    d = torch.mv(b.T, a)
    print(d.shape)

    # 一个二维矩阵和一个一维向量的矩阵乘积
    a = torch.rand(2, 3)
    b = torch.rand(3)
    # 等价于 torch.mv(a, b)
    c = torch.matmul(a, b)
    print(c.shape)  # torch.Size([2])
    d = torch.mv(a, b)
    print(d.shape)

    # 两个三维张量的矩阵乘积
    a = torch.rand(2, 3, 4)
    b = torch.rand(2, 4, 5)
    # 等价于 torch.bmm(a, b)
    c = torch.matmul(a, b)
    print(c.shape)  # torch.Size([2, 3, 5])
    d = torch.bmm(a, b)
    print(d.shape)
