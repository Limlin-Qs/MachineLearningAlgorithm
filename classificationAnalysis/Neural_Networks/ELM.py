# -*- coding: UTF-8 -*-
import numpy as np
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/9/8 22:40
  Software: PyCharm
  Profile: 极限学习机是一种单隐含层的前馈神经网络，其效率高，正确率高，泛化性能强，从初学该算法至今一直有在使用。
"""
#加载数据
def loadData():
    train_x = []
    train_y = []
    fileIn = open('testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))

    return np.mat(train_x), np.mat(train_y)

train_x, train_y = loadData()

[N, n] = train_x.shape
L = 20

W = np.random.rand(n, L)

b = np.random.rand(1, L)
b = np.repeat(b, N, axis=0)

tempH = train_x * W + b
H = g(tempH)
T = train_y.T

# m = 2
# T = np.zeros((N, m))
# for i in xrange(N):
#     if train_y[0, i] == 0:
#         T[i, 0] = 1
#     else:
#         T[i, 1] = 1
# 最小二乘求解
outputWeight = H.I * T

output = H * outputWeight

# print np.sign(output)

print(sum((output > 0.5) == train_y.T) * 1.0 / N)

# print np.sum(np.argmax(output, axis=1) == train_y.T) * 1.0 / N
