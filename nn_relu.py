import torch
from torch import nn
from torch.nn import ReLU

'''非线性ReLU处理'''
input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))


class Tian(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output


tian = Tian()
output = tian(input)
print(output)
