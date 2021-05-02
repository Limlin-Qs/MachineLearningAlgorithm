# -*- coding: UTF-8 -*-
# 导入画图工具
import matplotlib.pyplot as plt
# 导入数组工具
import numpy as np
# 导入数据集生成器
from sklearn.datasets import make_blobs
# 导入KNN 分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/4/11 17:06
  Software: PyCharm
  Profile: 读取数据集; 处理数据集数据 清洗，采用留出法hold-out拆分数据集：训练集、测试集
"""

# 生成样本数为200，分类数为2的数据集
data = make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.0, random_state=8)
X, Y = data

# 将生成的数据集进行可视化
# plt.scatter(X[:,0], X[:,1],s=80, c=Y,  cmap=plt.cm.spring, edgecolors='k')
# plt.show()

clf = KNeighborsClassifier()
clf.fit(X, Y)

# 绘制图形
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# meshgrid函数用两个坐标轴上的点在平面上画网格。
# meshgrid的作用是根据传入的两个一维数组参数生成两个数组元素的列表。
# 如果第一个参数是xarray，维度是xdimesion，第二个参数是yarray，维度是ydimesion。
# 那么生成的第一个二维数组是以xarray为行，ydimesion行的向量；
# 而第二个二维数组是以yarray的转置为列，xdimesion列的向量。

# np.arange()函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是6，步长为1。
# 参数个数情况： np.arange()函数分为一个参数，两个参数，三个参数三种情况
# 1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
# 2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。
# 3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Pastel1)
plt.scatter(X[:, 0], X[:, 1], s=80, c=Y, cmap=plt.cm.spring, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classifier:KNN")

# 把待分类的数据点用五星表示出来
plt.scatter(6.75, 4.82, marker='*', c='red', s=200)

# 对待分类的数据点的分类进行判断
res = clf.predict([[6.75, 4.82]])
plt.text(6.9, 4.5, 'Classification flag: ' + str(res))

plt.show()