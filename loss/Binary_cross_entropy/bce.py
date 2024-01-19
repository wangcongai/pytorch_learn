# 导入pytorch库
import torch
import torch.nn as nn

# 创建一个逻辑回归模型
model = nn.Linear(2, 1)  # 线性变换层，相当于w^Tx + b
sigmoid = nn.Sigmoid()  # sigmoid激活函数，相当于y = \sigma(w^Tx + b)

# 创建一个二元交叉熵损失函数
criterion = nn.BCELoss()

if __name__ == '__main__':
    # 创建一些模拟数据
    x = torch.tensor([[0.5, 0.3], [0.2, 0.4], [0.7, 0.6]])  # 特征向量，shape=(3, 2)
    y = torch.tensor([[0], [1], [1]], dtype=torch.float)  # 类别标签，shape=(3, 1)

    # 计算预测值
    output = model(x)  # 线性变换，shape=(3, 1)
    output = sigmoid(output)  # sigmoid激活，shape=(3, 1)

    # 计算损失值
    loss = criterion(output, y)  # 二元交叉熵损失，shape=(1,)
    print(loss)  # 打印损失值
