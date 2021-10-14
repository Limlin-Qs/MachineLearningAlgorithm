# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/10/13 11:00
  Software: PyCharm
  Profile: https://blog.csdn.net/weixin_43864473/article/details/86600466
  非参数回归算法，非参数方法拟合得到的曲线可以更好地描述变量之间的关系，不管是多么复杂的曲线关系都能拟合得到。
  loess（locally weighted regression）是一种用于局部回归分析的非参数方法，它主要是把样本划分成一个个小区间，
  对区间中的样本进行多项式拟合，不断重复这个过程得到在不同区间的加权回归曲线，最后再把这些回归曲线的中心连在
  一起合成完整的回归曲线，
"""

#定义一个正态分布，参数分别为均值，方差以及X的行向量
def guassianDistribution(mean,var,x):
    return 1/np.sqrt( 2 * np.pi * var )*np.exp( - (x[1]-mean) ** 2 / (2*var) )
#定义权值计算函数，带宽参数默认为0.1
def weight(xi,x,bd=0.1):
    diff=xi-x
    return np.exp(np.dot(diff,diff.T)/(-2 * bd**2 ))
#这里为了方便可视化，我们依旧就用二维特征(x0恒为1)
m=40
n=2
X=np.array([[1,x1] for x1 in np.random.rand(m)])
Y=np.array([guassianDistribution(0.5,0.5,x) for x in X])
#输入变量x及默认参数，分别设置0.2，0.6，0.9
input_x=np.array([1,0.2])
theta=np.array([0.0,0.0])
#学习率
a=0.1
#迭代1000次
for k in range(1000):
    for i in range(m):
        theta+=a * weight(X[i],input_x) * (Y[i]-np.dot(theta,X[i].T)) * X[i]
#画图
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],np.dot(X,theta))
plt.show()