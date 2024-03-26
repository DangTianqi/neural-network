import torch
import torch.nn.functional as f
import torchvision
from torch.utils.data import DataLoader

'''基本卷积网络的模型'''
input = torch.tensor([[1, 2, 0, 3, 1],
                     [4, 5, 6, 7, 8],
                     [9, 10, 11, 12, 13],
                     [14, 15, 16, 17, 18],
                     [19, 20, 21, 22, 23]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = f.conv2d(input, kernel, stride=1, padding=0)
print(output)
