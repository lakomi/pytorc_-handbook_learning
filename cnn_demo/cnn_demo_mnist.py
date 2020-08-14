# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：nltk-practise -> cnn_demo_mnist
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/12 10:17
@Desc   ： 找到数据集，对数据预处理，定义模型，调整超参数，测试训练，在通过训练结果对超参数进行调整或对模型进行调整
=================================================='''
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import cnn_demo.ConvNet
import cnn_demo.TrainAndTest

# 2g显存
BATCH_SIZE = 512
# 总共训练批次
EPOCHS = 20
# 判断能否有GPU，且能否使用
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用dataloader对数据进行读取
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST('data', train=False, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 实例化一个网络。并使用to方法，指定CPU还是GPU
model = ConvNet.ConvNet().to(DEVICE)
# 优化器。使用简单的Adam
optimizer = optim.Adam(model.parameters())

# 开始训练
for epoch in range(1, EPOCHS + 1):
    TrainAndTest.train(model, DEVICE, train_loader, optimizer, epoch)
    TrainAndTest.test(model, DEVICE, test_loader)
