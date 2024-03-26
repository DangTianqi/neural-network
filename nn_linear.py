import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''线性Linear处理'''
transform = transforms.Compose([transforms.ToTensor()])
# 加载CIFAR10数据集
dataset = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True,
                                       transform=transform)
# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Tian(nn.Module):
    """
    定义一个名为Tian的神经网络模型。

    该模型继承自nn.Module, 包含一个线性层。
    """

    def __init__(self):
        """
        初始化Tian模型。
        """
        super().__init__()
        # 创建一个输入维度为196608，输出维度为10的线性层
        self.linear1 = Linear(3 * 32 * 32, 10)

    def forward(self, input):
        """
        定义模型的前向传播路径。

        参数:
        - input: 输入的张量

        返回:
        - output: 经过模型处理后的输出张量
        """

        output = self.linear1(input.view(-1, 3 * 32 * 32))
        return output


# 实例化Tian模型
tian = Tian()
# 创建一个SummaryWriter对象，用于记录TensorBoard上的数据
writer = SummaryWriter("../lo")
step = 0
# 遍历数据加载器中的每个batch
for data in dataloader:
    imgs, targets = data
    # 在TensorBoard上添加输入图像
    writer.add_images("input", imgs, step)
    # 将图像数据展平，并在TensorBoard上添加
    # output = torch.flatten(imgs)
    output = tian(imgs.view(-1, 3 * 32 * 32))
    writer.add_histogram("output", output, step)

# 关闭SummaryWriter
writer.close()
