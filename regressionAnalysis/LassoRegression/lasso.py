# -*- coding: UTF-8 -*-
from sklearn.linear_model import Lasso
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.datasets import load_boston
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/10/13 11:00
  Software: PyCharm
  Profile: https://blog.csdn.net/weixin_44700798/article/details/110690015
  两种方式来求Lasso的参数：坐标轴下降法、用最小角回归法
"""

#生成100个一元回归数据集
x,y = make_regression(n_features=1,noise=5,random_state=2020)
plt.scatter(x,y)
plt.show()
#加5个异常数据,为什么这么加，大家自己看一下生成的x,y的样子
a = np.linspace(1,2,5).reshape(-1,1)
b = np.array([350,380,410,430,480])

#生成新的数据集
x_1 = np.r_[x,a]
y_1 = np.r_[y,b]

plt.scatter(x_1,y_1)
plt.show()

""" 线性回归展示
class normal():
    def __init__(self):
        pass

    def fit(self, x, y):
        m = x.shape[0]
        X = np.concatenate((np.ones((m, 1)), x), axis=1)
        xMat = np.mat(X)
        yMat = np.mat(y.reshape(-1, 1))

        xTx = xMat.T * xMat
        # xTx.I为xTx的逆矩阵
        ws = xTx.I * xMat.T * yMat

        # 返回参数
        return ws


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
clf1 = normal()
# 拟合原始数据
w1 = clf1.fit(x, y)
# 预测数据
y_pred = x * w1[1] + w1[0]

# 拟合新数据
w2 = clf1.fit(x_1, y_1)
# 预测数据
y_1_pred = x_1 * w2[1] + w2[0]

print('原始样本拟合参数：\n', w1)
print('\n')
print('新样本拟合参数：\n', w2)

ax1 = plt.subplot()
ax1.scatter(x_1, y_1, label='样本分布')
ax1.plot(x, y_pred, c='y', label='原始样本拟合')
ax1.plot(x_1, y_1_pred, c='r', label='新样本拟合')
ax1.legend(prop={'size': 15})  # 此参数改变标签字号的大小
plt.show()
"""
# 调用sklearn的Lasso回归
lr=Lasso(alpha=5)
lr.fit(x_1,y_1)
print('alpha=0时',lr.coef_,'\n')

#波士顿房价回归数据集
data = load_boston()
x_b =data['data']
y_b = data['target']

from sklearn.linear_model import Lasso
lr=Lasso(alpha= 1)
lr.fit(x_b, y_b)
print('当alpha=1时：\n',lr.coef_)