import time

import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 指定设备
device = torch.device("cpu")

# 引入数据
traindata = torchvision.datasets.CIFAR10(root="newDataset", train=True, transform=torchvision.transforms.ToTensor,
                                         download=True)
testdata = torchvision.datasets.CIFAR10(root="newDataset", train=False, transform=torchvision.transforms.ToTensor,
                                        download=True)

traindata_size = len(traindata)
testdata_size = len(testdata)
print("train数据集长度：{}".format(traindata_size))
print("test数据集长度：{}".format(testdata_size))

traindataloader = DataLoader(traindata, batch_size=64)
testdataloader = DataLoader(testdata, batch_size=64)
# 创建网络模型
net = Net()
net = net.to(device)
# if torch.cuda.is_available():
#     net = net.cuda() #使用GPU
# 定义损失函数和优化器
Loss_fn = nn.CrossEntropyLoss()
Loss_fn = Loss_fn.to(device)

# Loss_fn = Loss_fn.cuda() #使用GPU
lr = 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr)
# 训练
total_train_step = 0
total_test_step = 0
epoch = 10
writer = SummaryWriter("../logs")
start_time = time.time()
for i in range(epoch):

    net.train()
    for data in traindataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # imgs = imgs.cuda() #使用GPU
        # targets = targets.cuda() #使用GPU
        output = net(imgs)
        loss = Loss_fn(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            time = end_time - start_time
            # 此处可用于输出训练次数和训练时间；
    # 测试部分
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in testdataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            # imgs = imgs.cuda() #使用GPU
            # targets = targets.cuda() #使用GPU
            output = net(imgs)
            loss = Loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    writer.add_scalar("test loss", total_test_loss, total_test_step)
    writer.add_scalar("Precision", total_accuracy / testdata_size, i)
    total_test_step += 1

writer.close()
