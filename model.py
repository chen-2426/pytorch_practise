import torch
from torch import nn


class Net(nn.module):
    def __init__(self):
        super(Net, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)

        )

    def forword(self, x):
        x = self.modle(x)
        return x


if __name__ == '__main__':
    net = Net()
    input = torch.one((64, 3, 32, 32))
    out = net(input)
    print(out.shape)
