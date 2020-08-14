# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：pytorch_handbook_learning -> TabularDataSet
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/14 17:24
@Desc   ：定义一个简单的数据集
=================================================='''
from torch.utils.data import Dataset, DataLoader


# 定义一个简单的数据集
class tabularDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
