# 欧氏距离，提供给k-means0使用
import numpy as np


def euclidean(a, b):
    n = len(a)
    i = 0
    sum_sum = 0
    if i < n:
        squared_difference = np.square(a[i] - b[i])
        sum_sum += squared_difference
        n += 1
    distance = np.sqrt(sum_sum)
    return distance