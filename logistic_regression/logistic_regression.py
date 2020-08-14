# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：nltk-practise -> logistic
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/11 9:48
@Desc   ：
=================================================='''

import torch
import torch.nn as nn
import numpy as np
import logistic_regression.logistic_regression_module as lrm

# german.data-numeric是numpy处理好的数值化数据。使用load方法读取
data = np.loadtxt("german.data-numeric")
# 归一化。将数据变成（0,1）之间的小数
n, l = data.shape
# 按列遍历
for j in range(l - 1):
    meanVal = np.mean(data[:, j])
    # 标准差。方差的算术平方根
    stdVal = np.std(data[:, j])
    data[:, j] = (data[:, j] - meanVal) / stdVal
# 打乱数据
np.random.shuffle(data)

# 数据集格式，前24列为24个维度，第25列为标签
train_data = data[:900, :l - 1]  # 特征数据
train_lab = data[:900, l - 1] - 1  # 标签
test_data = data[900:, :l - 1]
test_lab = data[900:, l - 1] - 1


def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


# 以下进行一些设置
net = lrm.LR()
# 使用CrossEntropyLoss损失
criterion = nn.CrossEntropyLoss()
# Adam优化
optm = torch.optim.Adam(net.parameters())
# 训练1000次
epochs = 1000

# 开始训练
for i in range(epochs):
    # 指定模式为训练模式，计算梯度
    net.train()
    # 输入值都需要转化成torch的Tensor
    x = torch.from_numpy(train_data).float()
    y = torch.from_numpy(train_lab).long()
    y_hat = net(x)
    # 计算损失
    loss = criterion(y_hat, y)

    # 前一步的损失清零
    optm.zero_grad()
    # 反向传播
    loss.backward()
    # 优化
    optm.step()

    # 每100次，输出相关信息
    if (i + 1) % 100 == 0:
        # 指定模型为计算模式
        net.eval()
        test_in = torch.from_numpy(test_data).float()
        test_l = torch.from_numpy(test_lab).long()
        test_out = net(test_in)
        # 使用测试集，计算准确率
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i + 1, loss.item(), accu))
