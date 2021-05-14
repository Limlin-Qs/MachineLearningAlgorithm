# -*- coding: UTF-8 -*-
from sklearn.naive_bayes import GaussianNB
import random
import matplotlib.pyplot as plt
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/4/11 17:01
  Software: PyCharm
  Profile: 如果条件独立假设成立的话，NB将比鉴别模型（如Logistic回归）收敛的更快，所以你只需要少量的训练数据。即使条件独立假设不成
  立，NB在实际中仍然表现出惊人的好。如果你想做类似半监督学习，或者是既要模型简单又要性能好，NB值得尝试。
  https://blog.csdn.net/qq_40985471/article/details/84349507
"""

k = 5
# 颜色标签
colors = ['green', 'red', 'blue', 'yellow', 'pink']
# 先随机出中心点
centers = []
for i in range(k):
    x = 10 + 100 * random.random()
    y = 10 + 100 * random.random()
    centers.append((x, y))

points = []
# 然后在每个中心点的周围随机100个点
for ci, (x, y) in enumerate(centers):
    ps = []
    for i in range(100):
        px = x + random.random() * 20 - 10
        py = y + random.random() * 20 - 10
        ps.append((px, py))
        points.append(((px, py), ci))
    # 显示数据点
    plt.scatter(
        [x for x, y in ps],
        [y for x, y in ps],
        c=colors[ci], marker='.')
# plt.show()

model = GaussianNB()
model.fit([p for p,ci in points],[ci for p,ci in points])
pcolor=[]
grids=[(x,y) for x in range(0,120,5) for y in range(0,120,5)]
for i,ci in enumerate(model.predict(grids)):
    pcolor.append((grids[i],ci))


plt.scatter(
    [x for (x,y),ci in pcolor],
    [y for (x,y),ci in pcolor],
    c=[colors[ci] for (x,y) , ci in pcolor],marker='x'
)

plt.show()

