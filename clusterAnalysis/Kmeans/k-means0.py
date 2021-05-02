import random
from Kmeans import Calculating_the_mean
from Kmeans import Euclidean_distance
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from timeit import default_timer as timer

tic = timer()
# 待测试的代码


# 循环了两次，每次都算一次聚类中心，第一次随机，第二次与均值比较。
# 实验结果将数据分成了三组，三个聚堆，共三个py程序
data_sets = [[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
             [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
             [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
             [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
             [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
             [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]]

# 选择k=3
# k-means是初值敏感的，即当选取不同的初始值时分类结果可能不同
u = random.sample(data_sets, 3)

# u = [[0.774, 0.376], [0.639, 0.161], [0.714, 0.346]]
C1 = []
C2 = []
C3 = []
for data in data_sets:
    L1 = Euclidean_distance.euclidean(data, u[0])
    L2 = Euclidean_distance.euclidean(data, u[1])
    L3 = Euclidean_distance.euclidean(data, u[2])
    L = min(L1, L2, L3)
    if L1 == L:
        C1.append(data)
    elif L2 == L:
        C2.append(data)
    else:
        C3.append(data)

u1 = Calculating_the_mean.calculating(C1)
u2 = Calculating_the_mean.calculating(C2)
u3 = Calculating_the_mean.calculating(C3)

while (round(u[0][0], 6) != round(u1[0], 6)) | (round(u[0][1], 6) != round(u1[1], 6)) \
        | (round(u[1][0], 6) != round(u2[0], 6)) | (round(u[1][1], 6) != round(u2[1], 6)) \
        | (round(u[2][0], 6) != round(u3[0], 6)) | (round(u[2][1], 6) != round(u3[1], 6)):
    if u[0] != u1:
        u[0] = u1
    if u[1] != u2:
        u[1] = u2
    if u[2] != u3:
        u[2] = u3
    C1.clear()
    C2.clear()
    C3.clear()
    for data in data_sets:
        L1 = Euclidean_distance.euclidean(data, u[0])
        L2 = Euclidean_distance.euclidean(data, u[1])
        L3 = Euclidean_distance.euclidean(data, u[2])
        L = min(L1, L2, L3)
        if L1 == L:
            C1.append(data)
        elif L2 == L:
            C2.append(data)
        else:
            C3.append(data)

        u1 = Calculating_the_mean.calculating(C1)
        u2 = Calculating_the_mean.calculating(C2)
        u3 = Calculating_the_mean.calculating(C3)

toc = timer()
print("算法执行时间", toc - tic)  # 输出的时间，秒为单位
# print(C1)
# print(C2)
# print(C3)
x_axis = []
y_axis = []
plt.figure(figsize=(8, 8), dpi=80)
fig = plt.figure(1)
for i in C1:
    x_axis.append(i[0])
    y_axis.append(i[1])
plt.title('c1')
fig.add_subplot(221).scatter(x_axis, y_axis, alpha=0.6, cmap=plt.cm.Blues)
# fig = plt.subplot(2, facecolor='white')
for i in C2:
    x_axis.append(i[0])
    y_axis.append(i[1])
plt.title('c2')
fig.add_subplot(222).scatter(x_axis, y_axis, alpha=0.6, cmap=plt.cm.Reds)
# fig = plt.subplot(3, facecolor='white')
for i in C3:
    x_axis.append(i[0])
    y_axis.append(i[1])
fig.add_subplot(223).scatter(x_axis, y_axis, alpha=0.6, cmap=plt.cm.Greens)
plt.title('c3')
plt.show()
