import torch
import torch.nn as nn

if __name__ == '__main__':
    # size of input (N x C) is = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    # every element in target should have 0 <= value < C
    target = torch.tensor([1, 0, 4])

    m = nn.LogSoftmax(dim=1)
    log_softmax = m(input)
    nll_loss = nn.NLLLoss()
    # log_softmax中每一行取正样本对应的元素，取负值，然后逐行相加，最后求平均
    output1 = nll_loss(log_softmax, target)
    output1.backward()

    # nn.CrossEntropyLoss()相当于nn.NLLLoss()加上一个log_softmax层
    cross_entropy = nn.CrossEntropyLoss()
    output2 = cross_entropy(input, target)

    print('input: ', input)
    print('target: ', target)
    print('output1: ', output1)
    print('output2: ', output2)
