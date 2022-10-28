import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../newDataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class aanet(nn.Module):
    def __init__(self):
        super(aanet, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1),
            MaxPool2d(kernel_size=3, ceil_mode=True),
            ReLU()
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
Net = aanet()
optim = torch.optim.SGD(Net.parameters(), lr=0.01)
for epoch in range(20):
    for data in dataloader:
        imgs, targets = data
        output = Net(imgs)
        result_loss = loss(output, targets)
        result_loss.backward()
        optim.step()
# 保存网络模型及路径，需要访问到模型的情况
torch.save(Net, "Net.pth")
# 加载模型
model = torch.load("Net.pth")
# 保存方式2 保存成字典 推荐使用
torch.save(Net.state_dict(), "Net.pth")
# 加载
model.load_state_dict(torch.load("Net.pth"))
