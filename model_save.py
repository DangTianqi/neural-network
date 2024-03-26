import torch
import torchvision
from torch import nn
'''模板保存'''
vgg16 = torchvision.models.vgg16(weights=None)
# 保存方式1 保存模型结构及参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2 只保存模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# 陷阱1
class Tian(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x


tian = Tian()
torch.save(tian, "tian.pth")