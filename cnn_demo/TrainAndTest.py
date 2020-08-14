# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：nltk-practise -> TrainAndTest
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/12 16:27
@Desc   ：
=================================================='''
import torch
import torch.nn.functional as F


# 封装一个训练函数
def train(model, device, train_loader, optimizer, epoch):
    # 设置为训练模式
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 清零
        optimizer.zero_grad()

        output = model(data)
        # 损失
        loss = F.nll_loss(output, target)
        # 反向传播
        loss.backward()
        # 优化
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# 封装一个测试函数
def test(model, device, test_loader):
    # 设置为计算模式
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # 将一批的损失相加
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # 找到概率最大的下标
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
