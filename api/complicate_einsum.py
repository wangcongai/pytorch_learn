import torch

if __name__ == '__main__':
    # 创建三个输入张量，形状分别为(3, 4)，(3, 4, 5)，(4, 5)
    a = torch.rand(3, 4)
    b = torch.rand(3, 4, 5)
    c = torch.rand(4, 5)
    # 使用torch.einsum计算a和b与c的矩阵乘积，得到一个形状为(3, 5)的输出张量d
    d = torch.einsum('ij,ijk,jk->ik', [a, b, c])
    print(d.shape)
    # 输出torch.Size([3, 5])

    # 使用2个einsum算子得到相同的输出张量e
    # 首先，对b和c进行矩阵乘法，得到一个(3, 4, 5)的张量
    bc = torch.einsum('ijk,jk->ijk', [b, c])
    # 然后，对a和bc进行矩阵乘法，得到一个(3, 5)的张量
    e = torch.einsum('ij,ijk->ik', [a, bc])
    # 验证d和e是否相等
    print(torch.allclose(d, e))  # 输出True

    # 不使用torch.einsum，用torch.mul或torch.*得到相同的输出张量e
    # 首先，对c进行扩展，使其与b的形状一致 c_new=(3, 4, 5)
    c_new = c.unsqueeze(0).expand_as(b)
    # 然后，对b和c进行逐元素的乘法
    bc_new = torch.mul(b, c_new)  # 或者 e = b * c
    # 验证d和e是否相等
    print(torch.allclose(bc, bc_new))  # 输出True

    # 已知a=(3, 4), bc_new=(3, 4, 5), 实现e=torch.einsum('ij,ijk->ik', [a, bc_new])=(3, 5)
    a_new = a.unsqueeze(-1)
    # a_new = (3, 1, 4)
    a_new = torch.permute(a_new, (0, 2, 1))
    # e_new1 = (3, 1, 5)
    e_new1 = torch.matmul(a_new, bc_new)
    # e_new2 = (3, 1, 5)
    e_new2 = torch.bmm(a_new, bc_new)
    print(torch.allclose(e_new1.squeeze(1), e))
    print(torch.allclose(e_new2.squeeze(1), e))


