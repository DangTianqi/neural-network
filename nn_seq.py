import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

'''Sequential集合处理'''
class Tian(nn.Module):
    def __init__(self):
        super().__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


tian = Tian()

# 测试网络正确性
input = torch.ones(64, 3, 32, 32)
output = tian(input)
print(output.shape)

writer = SummaryWriter("../lo")
writer.add_graph(tian, input)
writer.close()