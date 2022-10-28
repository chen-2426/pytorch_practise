import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

testdata = torchvision.datasets.CIFAR10("./newDataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
# testdata = torchvision.datasets.CIFAR10("./newDataset",train=False,transform =torchvision.transforms.ToTensor(),
#                                           download=True)
test_loader = DataLoader(dataset=testdata, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
img, target = testdata[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        img, target = data
        writer.add_images("Epoch:{}".format(epoch), img, step)
        step += 1
writer.close()
