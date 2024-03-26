import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''非线性Sigmoid处理'''
input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Tian(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


tian = Tian()
writer = SummaryWriter('../lo')
step = 0
for data in dataloader:
    imgs, targets = data
    output = tian(imgs)
    writer.add_images("input", imgs, global_step=step,)
    writer.add_images("output", output, global_step=step)
    step = step + 1

writer.close()