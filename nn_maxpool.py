import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''最大池化MaxPool2d处理'''
dataset = torchvision.datasets.CIFAR10('../dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)
dataloader = DataLoader()


class Tian(nn.Module):
    def __init__(self):
        super(Tian, self).__init__()
        self.maxpool = MaxPool2d(3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output


tian = Tian()

writer = SummaryWriter("../lo")
step = 0
for data in dataloader:
    imgs, targets = data
    output = tian(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
