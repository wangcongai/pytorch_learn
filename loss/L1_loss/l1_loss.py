import torch
import torch.nn as nn


def mae_loss(y_pred, y_true):
    # Calculate the absolute error
    loss = abs(y_true - y_pred)
    # Return the mean loss over the batch
    return loss.mean()


if __name__ == '__main__':
    predict = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)

    l1_loss = nn.L1Loss()
    l1_loss1 = l1_loss(predict, target)
    l1_loss1.backward()
    print(predict.grad)
    print('l1_loss1: ', l1_loss1)

    l1_loss2 = mae_loss(predict, target)
    # 将predict变量梯度清零
    predict.grad.data.zero_()
    l1_loss2.backward()
    print(predict.grad)
    print('l1_loss2: ', l1_loss2)
