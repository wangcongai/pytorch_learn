import torch
import torch.nn as nn


# Define a function to calculate the Huber loss between two tensors
def huber_loss(y_true, y_pred, delta=1.0):
    # Calculate the absolute error
    error = abs(y_true - y_pred)
    # Use a torch.where function to choose the loss function for each element
    loss = torch.where(error < delta, (error ** 2) / 2, delta * (error - delta / 2))
    # Return the mean loss over the batch
    return loss.mean()


if __name__ == '__main__':
    # Create some random input and target tensors
    predict = torch.randn(4, 5)
    target = torch.randn(4, 5)

    # Define the delta parameter
    Delta = 1.0

    # Use the HuberLoss class
    loss_fn = nn.HuberLoss(delta=Delta)
    huber_loss1 = loss_fn(predict, target)
    print(huber_loss1)

    # Use self-define huber_loss function
    huber_loss2 = huber_loss(predict, target, delta=Delta)
    print(huber_loss2)
