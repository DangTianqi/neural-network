import torch
from torch import nn
'''神经网络的基础模型架构'''

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()

    def forward(self, input):
        optput = input + 1
        return optput


tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)