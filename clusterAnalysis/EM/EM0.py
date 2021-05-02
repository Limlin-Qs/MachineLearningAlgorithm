# -*- coding: UTF-8 -*-
# 模拟两个正态分布的均值估计

from numpy import *
import numpy as np
import random
import copy
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/2/25 11:38
  Software: PyCharm
  Profile: 运用了GMM高斯混合模型
  https://blog.csdn.net/u010866505/article/details/77877345?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&dist_request_id=6bca44d3-a1f5-485e-9019-e5be8422bb00&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
"""


SIGMA = 6
EPS = 0.0001


# 生成方差相同,均值不同的样本
def generate_data():
    Miu1 = 20
    Miu2 = 40
    N = 1000
    X = mat(zeros((N, 1)))
    for i in range(N):
        temp = random.uniform(0, 1)
        if (temp > 0.5):
            X[i] = temp * SIGMA + Miu1
        else:
            X[i] = temp * SIGMA + Miu2
    return X


# EM算法
def my_EM(X):
    k = 2
    N = len(X)
    Miu = np.random.rand(k, 1)
    Posterior = mat(zeros((N, 2)))
    dominator = 0
    numerator = 0
    # 先求后验概率
    for iter in range(1000):
        for i in range(N):
            dominator = 0
            for j in range(k):
                dominator = dominator + np.exp(-1.0 / (2.0 * SIGMA ** 2) * (X[i] - Miu[j]) ** 2)
            # print dominator,-1/(2*SIGMA**2) * (X[i] - Miu[j])**2,2*SIGMA**2,(X[i] - Miu[j])**2
            # return
            for j in range(k):
                numerator = np.exp(-1.0 / (2.0 * SIGMA ** 2) * (X[i] - Miu[j]) ** 2)
                Posterior[i, j] = numerator / dominator
        oldMiu = copy.deepcopy(Miu)
        # 最大化
        for j in range(k):
            numerator = 0
            dominator = 0
            for i in range(N):
                numerator = numerator + Posterior[i, j] * X[i]
                dominator = dominator + Posterior[i, j]
            Miu[j] = numerator / dominator
        print((abs(Miu - oldMiu)).sum())
        # print '\n'
        if (abs(Miu - oldMiu)).sum() < EPS:
            print
            Miu, iter
            break


if __name__ == '__main__':
    X = generate_data()
    my_EM(X)