# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：nltk-practise -> ConvNet
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/12 10:44
@Desc   ：
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个cnn网络。包含两个卷积层conv1和conv2，紧接着两个线性层作为输出，最后输出10个维度。
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积层。（输入通道数，输出通道数，卷积核大小）
        self.conv1 = nn.Conv2d(1, 10, 5)  # 输入通道数1，输出通道数10，核大小5
        self.conv2 = nn.Conv2d(10, 20, 3)  # 输入通道数10，输出通道数20，核大小3
        # 全连接层Linear,（输入通道数，输出通道数）
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 输入通道2000，输出通道500
        self.fc2 = nn.Linear(500, 10)  # 输入通道500，输出通道10，即10种分类

    def forward(self, x):
        # 本例中，in_size即BATCH_SIZE的值，输入的x可看成512*1*28*28的张量
        in_size = x.size(0)
        # 第一次卷积
        # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = self.conv1(x)
        # batch*10*24*24  激活函数relu不改变形状
        out = F.relu(out)
        # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = F.max_pool2d(out, 2, 2)

        # 第二次卷积
        # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = self.conv2(x)
        # batch*20*10*10
        out = F.relu(out)
        # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = out.view(in_size, -1)

        # 全连接层
        # batch*2000 -> batch*500
        out = self.fc1(out)
        # batch*500
        out = F.relu(out)
        # batch*500 -> batch*10
        out = self.fc2(out)
        # 计算log(softmax(x))
        out = F.log_softmax(out, dim=1)
        return out
