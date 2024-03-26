import torch
import torchvision
from PIL import Image
from torch import nn
from model import *
image_path = "E:\\CodesProjects\\Python\\pythonProject\\images\\dog.png"
image = Image.open(image_path)
# image = Image.convert("RGB")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
cifar10_dataset = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True)

image = transform(image)


# class Tian(nn.Module):
#     def __init__(self):
#         super(Tian, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(1024, 64),
#             nn.Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x


model = torch.load("E:\\CodesProjects\\Python\\pythonProject\\neural_network\\tian_3.pth")
image = torch.reshape(image, (1, 3, 32, 32))

if torch.cuda.is_available():
    device = torch.device('cuda')
    model = model.to(device)
    image = image.to(device)

with torch.no_grad():
    output = model(image)
print(output)
print("CIFAR10 数据集的种类包括：", cifar10_dataset.classes)
print(output.argmax(1))
