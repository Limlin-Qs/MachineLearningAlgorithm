# -*- coding: UTF-8 -*-
# 可以顺利分析，并绘制出方格图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import time

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/5/30 18:46
  Software: PyCharm
  Profile:
"""
# 定义下载数据的函数
def ReadAndSaveDataByPandas(target_url=None, file_save_path=None, save=False):
    if target_url != None:
        target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
    if file_save_path != None:
        file_save_path = "file_result.txt"

    wine = pd.read_csv(target_url, header=0, sep=";")
    if save == True:
        wine.to_csv(file_save_path, index=False)

    return wine


# 从硬盘读取数据进入内存
wine = pd.read_csv("D:\\Analysis_Data\\分类算法实验数据\\white_wine\\winequality-red.csv", header=0, sep=";")

print(wine.head())

""" ============================================================ """

start  = time.time()

def GRA_ONE(DataFrame, m=0):
    gray = DataFrame
    # 读取为df格式
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    # 标准化
    std = gray.iloc[:, m]  # 为标准要素
    ce = gray.iloc[:, 0:]  # 为比较要素
    n = ce.shape[0]
    m = ce.shape[1]  # 计算行列

    # 与标准要素比较，相减
    a = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            a[i, j] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中最大值与最小值
    c = np.max(a)
    d = np.min(a)

    # 计算值
    result = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            result[i, j] = (d + 0.5 * c) / (a[i, j] + 0.5 * c)

    # 求均值，得到灰色关联值
    result2 = np.zeros(m)
    for i in range(m):
        result2[i] = np.mean(result[i, :])
    RT = pd.DataFrame(result2)
    return RT


def GRA(DataFrame):
    list_columns = [str(s) for s in range(len(DataFrame.columns)) if s not in [None]]
    df_local = pd.DataFrame(columns=list_columns)
    for i in range(len(DataFrame.columns)):
        df_local.iloc[:, i] = GRA_ONE(DataFrame, m=i)[0]
    return df_local


data_wine_gra = GRA(wine)
end = time.time()
time = end - start
print("算法运行时间：", time)

# data_wine_gra.to_csv(path+"GRA.csv") 存储结果到硬盘

# 灰色关联结果矩阵可视化
def ShowGRAHeatMap(DataFrame):
    colormap = cm.get_cmap('RdYlBu_r')
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(DataFrame.astype(float), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
                annot=True)
    plt.show()


ShowGRAHeatMap(data_wine_gra)
