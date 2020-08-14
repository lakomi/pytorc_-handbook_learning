# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：rnn_demo -> rnn_module
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/13 17:51
@Desc   ：自定义的rnn网络结构
=================================================='''

import torch
import torch.nn as nn

# rnn 输入维度
INPUT_SIZE = 1
# of rnn 隐藏单元个数
H_SIZE = 64


# 定义rnn网络结构
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=INPUT_SIZE,
                          hidden_size=H_SIZE,
                          num_layers=1,
                          batch_first=True)
        self.out = nn.Linear(H_SIZE, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        # 保存所有的预测值
        outs = []
        for time_step in range(r_out.size(1)):  # 计算每一步长的预测值
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
