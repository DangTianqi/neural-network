import torch
import torchvision
'''模板加载'''
# 方式1 -> 保存方式1的加载模型
model = torch.load("vgg16_method1.pth")
print(model)

# 方式2 -> 保存方式2的加载模型
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(model)
print(vgg16)

# 陷阱1
# 需要先将保存的模型的类给导入后，才可以再使用load方法
