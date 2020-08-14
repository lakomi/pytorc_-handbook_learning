# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：nltk-practise -> logistic_regression_module
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/11 10:55
@Desc   ：
=================================================='''
import torch
import torch.nn as nn


# 定义模型
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        # 24个维度
        self.fc = nn.Linear(24, 2)

    def forward(self, x):
        out = self.fc(x)
        # Sigmod函数是最常见的logistic函数
        out = torch.sigmoid(out)
        return out
