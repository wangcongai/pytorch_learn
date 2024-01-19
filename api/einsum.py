import torch

if __name__ == '__main__':
    # 矩阵乘法
    a = torch.rand(2, 3)
    b = torch.rand(3, 4)
    # 等价于 torch.mm(a, b)
    c = torch.einsum('ik,kj->ij', [a, b])
    print(c.shape)  # torch.Size([2, 4])

    # 矩阵转置
    a = torch.rand(2, 3)
    # 等价于 torch.transpose(a, 0, 1)
    b = torch.einsum('ij->ji', [a])
    print(b.shape)  # torch.Size([3, 2])

    # 矩阵对角线元素
    a = torch.rand(3, 3)
    # 等价于 torch.diagonal(a, 0)
    b = torch.einsum('ii->i', [a])
    print(b.shape)  # torch.Size([3])

    # 矩阵行求和
    a = torch.rand(2, 3)
    # 等价于 torch.sum(a, dim=1)
    b = torch.einsum('ij->i', [a])
    print(b.shape)  # torch.Size([2])

    # 矩阵列求和
    a = torch.rand(2, 3)
    # 等价于 torch.sum(a, dim=0)
    b = torch.einsum('ij->j', [a])
    print(b.shape)  # torch.Size([3])

    # 矩阵点积求和
    a = torch.rand(2, 3)
    # 等价于 torch.sum(a * a)
    b = torch.einsum('ij,ij->', [a, a])
    print(b.shape)  # torch.Size([])

    # 多维矩阵乘法
    a = torch.rand(3, 4)
    b = torch.rand(3, 4, 5)
    c = torch.rand(4, 5)
    # 等价于 torch.matmul(a, torch.matmul(b, c))
    d = torch.einsum('ij,ijk,jk->ik', [a, b, c])
    print(d.shape)  # torch.Size([3, 5])
    # tmp = torch.matmul(b, c)
    # print(tmp.shape)
