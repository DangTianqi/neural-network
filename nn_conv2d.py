import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''卷积Conv2d处理'''
dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=True)


class Tian(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3, 1, 0)

    def forward(self, input):
        input = self.conv1(input)
        return input


tian = Tian()

writer = SummaryWriter("../lo")
step = 0
for data in dataloader:
    imgs, targets = data
    output = tian(imgs)
    writer.add_images('input', imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('output', output, step)

    step = step + 1