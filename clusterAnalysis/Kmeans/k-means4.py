# -*- coding:utf-8 -*-
# 会根据给出的数据分析，并生成聚类，输出图像到文件
import math
import pylab as pl
import codecs
import re
import datetime

pl.rcParams['axes.unicode_minus'] = False


# 计算欧式距离,a,b代表两个元组
def calcudistance(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


# 求出最小距离
def dist_min(Ci, Cj):
    return min(calcudistance(i, j) for i in Ci for j in Cj)


# 求出最大距离
def dist_max(Ci, Cj):
    return max(calcudistance(i, j) for i in Ci for j in Cj)


# 求出平均距离
def dist_avg(Ci, Cj):
    return sum(calcudistance(i, j) for i in Ci for j in Cj) / (len(Ci) * len(Cj))


# 找到距离最小的下标
def find_min(M):
    min = 1000
    x = 0;
    y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j];
                x = i;
                y = j
    return (x, y, min)


# 算法核心
def AGNES(dataset, dist, k):
    # 初始化C和M
    C = [];
    M = []
    for i in dataset:
        Ci = []
        Ci.append(i)
        C.append(Ci)
    for i in C:
        Mi = []
        for j in C:
            Mi.append(dist(i, j))
        M.append(Mi)
    q = len(dataset)
    # 合并更新
    while q > k:
        x, y, min = find_min(M)
        C[x].extend(C[y])
        C.remove(C[y])
        M = []
        for i in C:
            Mi = []
            for j in C:
                Mi.append(dist(i, j))
            M.append(Mi)
        q -= 1
    return C


# 画出结果图
def drawfig(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']  # 颜色数组
    for i in range(len(C)):
        coo_X = []  # x坐标
        coo_Y = []  # y坐标
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='o', color=colValue[i % len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.title("聚类结果图")
    pl.savefig(savepath + '2.png')
    pl.show()


def draworigian(dataset):
    x_list = list()
    y_list = list()
    for i in range(len(dataSet)):
        temp = dataSet[i]
        x_list.append(temp[0])
        y_list.append(temp[1])
    pl.scatter(x_list, y_list, marker='o', color="b")
    pl.legend(loc='upper right')
    pl.title("数据原始分布")
    pl.savefig(savepath + '1.png')
    pl.show()


def loadtxt(Filepath):
    # 读取文本 保存为二维点集
    inDate = codecs.open(Filepath, 'r', 'utf-8').readlines()
    dataSet = list()
    for line in inDate:  # 段落的处理
        line = line.strip()
        strList = re.split('[ ]+', line)
        numList = list()
        for item in strList:
            num = float(item)
            numList.append(num)
            # print numList
        dataSet.append(numList)
    return dataSet  # dataSet = [[], [], [], ...]


savepath = 'D:/研2/模式识别/'
Filepath = "D:/研2/模式识别/testSet2.txt"  # 数据集文件
dataSet = loadtxt(Filepath)  # 载入数据集
draworigian(dataSet)

start = datetime.datetime.now()
result = AGNES(dataSet, dist_avg, 4)
end = datetime.datetime.now()
timeused = end - start
print(timeused)

drawfig(result)

# 100   1.203133, 01.140652, 1.156260, 1.203152, 1.453138
# 200点 9.359476, 09.367193, 09.312600, 09.325362, 09.356845
# 500点 147.946446, 147:351248, 147.153595,147.946446, 145.493638

# 500 无需 145.429797 146.016936  147.240645  146.563253 147.534587
