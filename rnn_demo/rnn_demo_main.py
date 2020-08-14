# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：rnn_demo -> rnn_demo_main
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/13 10:25
@Desc   ：通过sin预测cos
=================================================='''
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
import rnn_demo.rnn_module as rnn_module

# rnn 时序步长数
TIME_STEP = 10
# GPU or CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 训练次数
EPOCHS = 300
# 隐藏层状态
h_state = None

# 使用sin和cos函数，不需要dataloader
steps = np.linspace(0, np.pi * 2, 256, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)

# 测试。将sin，cos可视化
plt.figure(1)
plt.suptitle('Sin and Cos', fontsize='18')
plt.plot(steps, y_np, 'r-', label='target(cos)')
plt.plot(steps, x_np, 'b-', label='input(sin)')
plt.legend(loc='best')
plt.show()

# 定义网络
rnn = rnn_module.RNN().to(DEVICE)
# Adam优化
optimizer = optim.Adam(rnn.parameters())
# 由于结果是一个数值，因此损失函数使用均方误差
criterion = nn.MSELoss()

# 训练模式
rnn.train()
plt.figure(2)
# 训练测试一起
for step in range(EPOCHS):
    # 一个时间周期
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    x = x.to(DEVICE)
    # rnn输出
    prediction, h_state = rnn(x, h_state)

    # 重置隐藏层的状态。切断和前一次迭代的链接
    h_state = h_state.data

    loss = criterion(prediction.cpu(), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每训练20个批次，就可视化一下。输出loss
    if (step + 1) % 20 == 0:
        # 控制台输出loss信息
        print("EPOCHS:{},LOSS:{:4f}".format(step, loss))
        # 可视化出两条曲线。红线-函数，蓝线-预测
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.cpu().data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.01)
