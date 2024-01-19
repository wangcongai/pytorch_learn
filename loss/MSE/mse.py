import torch
import torch.nn as nn


def mse_loss(y_pred, y_true):
    # Calculate the mean square error
    loss = (y_true - y_pred) ** 2
    # Return the mean loss over the batch
    return loss.mean()


if __name__ == '__main__':
    predict = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    l2_loss = nn.MSELoss()
    l2_loss1 = l2_loss(predict, target)
    l2_loss1.backward()
    print(predict.grad)
    print('l2_loss1: ', l2_loss1)

    l2_loss2 = mse_loss(predict, target)
    # 将predict变量梯度清零
    predict.grad.data.zero_()
    l2_loss2.backward()
    print(predict.grad)
    print('l2_loss2: ', l2_loss2)