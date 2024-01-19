# 导入Pytorch库
import torch


# 定义一个函数f(x) = x^2 + 2x + 1
def f(x):
    return x**2 + 2*x + 1


if __name__ == '__main__':
    # 创建一个张量x，值为3，设置requires_grad为True，表示需要计算梯度
    x = torch.tensor(3.0, requires_grad=True)

    # 计算y = f(x)的值
    y = f(x)

    # 调用backward()方法，根据链式法则计算x的梯度
    y.backward()

    # 打印x的梯度，应该等于f'(x) = 2x + 2，在x=3时，值为8
    print(x.grad)
