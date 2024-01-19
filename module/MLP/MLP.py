# 导入pytorch库
import torch
import torch.nn as nn


# 定义一个MLP层的类，继承自nn.Module
class MLPLayer(nn.Module):
    # 初始化方法，接收输入维度，输出维度，激活函数和随机种子
    def __init__(self, input_dim, output_dim, activation, seed=0):
        # 调用父类的初始化方法
        super(MLPLayer, self).__init__()
        # 设置随机种子，保证每次运行结果一致
        torch.manual_seed(seed)
        # 创建一个线性层，设置输入维度，输出维度
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)
        # 创建一个激活层，设置激活函数
        self.activation = activation

    # 前向传播方法，接收输入向量，返回输出向量
    def forward(self, x):
        # 计算线性层的输出
        z = self.linear(x)
        # 计算激活层的输出
        y = self.activation(z)
        # 返回输出向量
        return y


if __name__ == '__main__':
    # 创建一个MLP层的实例，输入维度为2，输出维度为3，激活函数为ReLU
    mlp_layer = MLPLayer(2, 3, nn.ReLU())
    # 创建一个输入向量，值为[1, 2]
    x = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
    # 调用MLP层的前向传播方法，得到输出向量
    y = mlp_layer(x)
    # 打印输出向量
    print(y)

