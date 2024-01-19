import torch
import numpy as np
import matplotlib.pyplot as plt

# 创建一个线性模型
# model = torch.nn.Linear(in_features=1, out_features=1)

# 创建一个复杂的神经网络模型
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=1, out_features=10),  # 输入层到隐藏层的线性变换，有10个神经元
    torch.nn.ReLU(),  # 隐藏层的激活函数，使用ReLU函数
    torch.nn.Linear(10, 100),  # 隐藏层到隐藏层的线性变换，有100个神经元
    torch.nn.ReLU(),  # 隐藏层的激活函数，使用ReLU函数
    torch.nn.Linear(100, 10),  # 隐藏层到隐藏层的线性变换，有10个神经元
    torch.nn.ReLU(),  # 隐藏层的激活函数，使用ReLU函数
    torch.nn.Linear(10, 1)  # 隐藏层到输出层的线性变换，有1个神经元
)

# 生成训练数据
# 生成100个随机的x，每个x是一个一维的向量
x = np.random.rand(1000, 1)
# 根据y=sin(x)的公式计算y的值
y = np.sin(2 * np.pi * x)

# 将x和y转换成pytorch的张量，并调整形状
# 将x转换成float类型的张量，并将形状从(100, 1)变成(-1, 1)，表示每个样本都是一个一维的向量
x = torch.from_numpy(x).float().view(-1, 1)
# 同理，将y转换成float类型的张量，并将形状从(100, 1)变成(-1, 1)
y = torch.from_numpy(y).float().view(-1, 1)

# 创建一个损失函数
loss_fn = torch.nn.MSELoss()

# 创建一个优化器
# 指定模型的参数和学习率为0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# 设置训练的轮数
epoch = 200

if __name__ == '__main__':
    # 开始训练的循环
    for i in range(epoch):
        # 使用模型对输入的x进行预测，得到输出的y_pred
        y_pred = model(x)
        # 使用损失函数计算y_pred和y之间的损失值
        loss = loss_fn(y_pred, y)
        # 使用优化器将模型的参数的梯度清零
        optimizer.zero_grad()
        # 使用损失值的反向传播方法计算损失函数对模型参数的梯度
        loss.backward()
        # 使用优化器的步进方法更新模型的参数
        optimizer.step()
        # 打印损失值和模型的参数
        print("Epoch: {}, Loss: {}".format(i+1, loss.item()))
        # print("w: {}, b: {}".format(model.weight.data.item(), model.bias.data.item()))

    x_show = np.linspace(0, 1, 100)
    y_show = np.sin(2 * np.pi * x_show)
    x_show = torch.from_numpy(x_show).float().view(-1, 1)
    y_show = torch.from_numpy(y_show).float().view(-1, 1)

    y_pred_show = model(x_show)
    # 使用matplotlib库来绘制真实的y值和预测的y_pred值的曲线图
    # 绘制真实的y值，用蓝色的线条表示
    plt.plot(x_show.numpy(), y_show.numpy(), label="true")
    # 绘制预测的y_pred值，用橙色的线条表示
    plt.plot(x_show.numpy(), y_pred_show.detach().numpy(), label="predict")
    # 设置图的标题
    plt.title("y=sin(x)")
    # 设置x轴的标签
    plt.xlabel("x")
    # 设置y轴的标签
    plt.ylabel("y")
    # 显示图例
    plt.legend()
    # 显示图像
    plt.show()
