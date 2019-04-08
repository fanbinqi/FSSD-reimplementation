# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""Pytorch中神经网络模块化接口nn的了解"""
"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
定义自已的网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中，
    不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)

    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
    在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
    if,for,print,log等python语法.

    注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
    比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：

    input_image = torch.FloatTensor(1, 28, 28)
    input_image = Variable(input_image)
    input_image = input_image.unsqueeze(0)   # 1 x 1 x 28 x 28

"""


class LeNet(nn.Module):
    def __init__(self):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(LeNet, self).__init__()  # 等价与nn.Module.__init__()

        # nn.Conv2d返回的是一个Conv2d class的一个对象，该类中包含forward函数的实现
        # 当调用self.conv1(input)的时候，就会调用该类的forward函数
        self.conv1 = nn.Conv2d(1, 6, (5, 5))  # output (N, C_{out}, H_{out}, W_{out})`
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # F.max_pool2d的返回值是一个Variable
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # 返回值也是一个Variable对象
        return x


def output_name_and_params(net):
    for name, parameters in net.named_parameters():
        print('name: {}, param: {}'.format(name, parameters))


if __name__ == '__main__':
    net = LeNet()
    print('net: {}'.format(net))
    params = net.parameters()  # generator object
    print('params: {}'.format(params))
    output_name_and_params(net)

    input_image = torch.FloatTensor(10, 1, 28, 28)

    # 和tensorflow不一样，pytorch中模型的输入是一个Variable，而且是Variable在图中流动，不是Tensor。
    # 这可以从forward中每一步的执行结果可以看出
    input_image = Variable(input_image)

    output = net(input_image)
    print('output: {}'.format(output))
    print('output.size: {}'.format(output.size()))
