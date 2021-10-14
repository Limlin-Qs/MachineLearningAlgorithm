# -*- coding: UTF-8 -*-
import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/10/13 11:00
  Software: PyCharm
  Profile: https://blog.csdn.net/myhome908/article/details/88428062
"""

# 读入数据
data = genfromtxt('lingRegression.csv', delimiter=',')
print(data)
# 切分数据
x_data = data[1:, 2:]  # 第2行到最后一行，第第三列到最后一列
y_data = data[1:, 1]  # 第2行到最后一行，第二列
print(x_data)
print(y_data)
# 常见模型
# 生成50个值
alphas_to_test = np.linspace(0.001, 1)  # 作为勒木达的值默认是50个，可以自行设置
# RidgeCV岭回归交叉验证，用交叉验证测试alphas_to_test的50个值找到合适的值
# store_cv_values存储交叉验证得到的结果、

model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
model.fit(x_data, y_data)

# 岭系数，测试50个领系数后，得到较好的那个
print(model.alpha_)
# loss值
print(model.cv_values_.shape)  # (16, 50)16个loss值，因为进行交叉验证，一共有16行数据，其中一行作为测试集。其余为训练集
# 这样可以得到16组loss值，50指的是领系数有50个[[],[],[],[],[],[]]二维数组，16行，每一行50个数
# 画图
# 领系数跟loss值的关系
plt.plot(alphas_to_test, model.cv_values_.mean(axis=0))
# 选取领系数的位置
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)), 'ro')
plt.show()

model.predict(x_data[2,np.newaxis])#x_data的带三行数据，是一维的，np.newaxis将其变为二维的，得到预测结果