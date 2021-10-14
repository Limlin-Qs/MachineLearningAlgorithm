# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/10/13 11:00
  Software: PyCharm
  Profile:
"""
# 一元线性回归模型
def model(a, b, x):
    return a*x + b

# 损失函数
def cost_function(a, b, x, y):
    n = 5
    return 0.5/n * (np.square(y-a*x-b)).sum()

# 优化函数
def optimize(a,b,x,y):
    n = 5
    alpha = 1e-1
    y_hat = model(a,b,x)
    da = (1.0/n) * ((y_hat-y)*x).sum()
    db = (1.0/n) * ((y_hat-y).sum())
    a = a - alpha*da
    b = b - alpha*db
    return a, b

if __name__ == '__main__':
    x = [13854,12213,11009,10655,9503] #程序员工资，顺序为北京，上海，杭州，深圳，广州
    x = np.reshape(x,newshape=(5,1)) / 10000.0
    y =  [21332, 20162, 19138, 18621, 18016] #算法工程师，顺序和上面一致
    y = np.reshape(y,newshape=(5,1)) / 10000.0
    # 调用模型
    lr = LinearRegression()
    # 训练模型
    lr.fit(x,y)
    # 计算R平方
    print(lr.score(x,y))
    # 计算y_hat
    y_hat = lr.predict(x)
    # 打印出图
    plt.scatter(x,y)
    plt.plot(x, y_hat)
    plt.show()

