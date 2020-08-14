# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：pytorch_handbook_learning -> process_struct_data
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/14 12:20
@Desc   ：处理结构化数据
=================================================='''
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
import structured_data.TabularDataSet as TabularDataSet

# 读取。df为DataFrame结构
df = pd.read_csv('../data/adult.csv')
# pandas中unique() 函数返回每个特征的唯一值
# salary是最后要分类的结果
salary_items = df['salary'].unique()
# print(salary_items)  # ['>=50k' '<50k']

# 预览一小块数据  head()方法 从开头开始预览。不带参数默认显示5条数据。tail()方法 从结尾预览，默认5条。
data_pre_look = df.head()
# print(data_pre_look)

# describe()对数据集进行概览，输出该数据集的计数、最大值、最小值等。
csv_des = df.describe()
# print(csv_des)

# ------------------------------------------------------------------------------------------------------
# 将数据分成以下三个类别
# 训练结果
result_var = 'salary'
# 分类型数据。csv中，不是数值型数据的表头
cata_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
# 数值型数据。CSV中，数值型数据的表头
cont_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# 查看分类型数据的数量和分布情况
# for col in df.columns:
#     if col in cata_names:
#         ccol = Counter(df[col])
#         print(col, len(ccol), ccol)
#         print("\r\n")

# ------------------------------------------------------------------------------------------------------
# 将分类型数据转成数字型数据。并对缺失的数据做填充
# fillna()函数做空值填充。标识成一个与其他值不一样的值即可。
for col in df.columns:
    if col in cata_names:
        df[col].fillna('---')
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    if col in cont_names:
        # 数值型做0填充。
        df[col] = df[col].fillna(0)

# ------------------------------------------------------------------------------------------------------
# 分割训练数据和标签
Y = df['salary']
Y_label = LabelEncoder()
Y = Y_label.fit_transform(Y)
X = df.drop(columns=result_var)
# ------------------------------------------------------------------------------------------------------


train_ds = TabularDataSet.tabularDataset(X, Y)
# 可以直接索引
print(train_ds[0])

# 训练的过程中，使用DataLoader加载数据
train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)
