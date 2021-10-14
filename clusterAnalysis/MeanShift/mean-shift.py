# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_iris
from scipy.spatial.distance import euclidean

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/10/14 17:07
  Software: PyCharm
  Profile: https://blog.csdn.net/u013894072/article/details/84553750
"""
# 欧式距离函数
def eculide(A, B):
    if len(A) != len(B):
        # A是一个点，B是点集
        print('a\n',A,'\nb\n',type(B))
        # return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
        dis = list(euclidean(A, b) for b in B)
        print(len(dis), np.array(dis))
        return np.array(dis)
    else:
        return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

def load_data():
    """
    导入iris标准数据集
    :return:
    """
    iris = load_iris()
    data = iris.data
    target = iris.target
    target_names = iris.target_names
    return data,target,target_names

class Group(object):
    """
    定义类簇的类 -- 后续也会使用
    """
    def __init__(self):
        self._name = ""
        self._no = None
        self._members = []
        self._center = None
    @property
    def no(self):
        return self._no
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,no):
        self._no = no
        self._name = "G"+str(self._no)
    @property
    def members(self):
        return self._members
    @members.setter
    def members(self,member):
        if member is None:
            raise TypeError("member is None,please set value")
        if isinstance(member,list):
            self.members.extend(member)
            return
        self._members.append(member)
    def clear_members(self):
        self._members = []
    @property
    def center(self):
        return self._center
    @center.setter
    def center(self,c):
        self._center = c

class MeanShift(object):
    """
    均值漂移聚类-基于密度
    """

    def __init__(self, radius=0.5, distance_between_groups=2.5, bandwidth=1, use_gk=True):
        self._radius = radius
        self._groups = []
        self._bandwidth = bandwidth
        self._distance_between_groups = distance_between_groups
        self._use_gk = use_gk  # 是否启用高斯核函数

    def _find_nearst_indexes(self, xi, XX):
        if XX.shape[0] == 0:
            return []
        distances = eculide(xi, XX)
        nearst_indexes = np.where(distances <= self._distance_between_groups)[0].tolist()
        print(len(nearst_indexes), len(XX))
        return nearst_indexes

    def _compute_mean_vector(self, xi, datas):
        distances = datas - xi
        if self._use_gk:
            sum1 = self.gaussian_kernel(distances)
            sum2 = sum1 * (distances)
            mean_vector = np.sum(sum2, axis=0) / np.sum(sum1, axis=0)
        else:
            mean_vector = np.sum(datas - xi, axis=0) / datas.shape[0]
        return mean_vector

    def fit(self, X):
        XX = X
        while (XX.shape[0] != 0):
            # 1.从原始数据选取一个中心点及其半径周边的点 进行漂移运算
            index = np.random.randint(0, XX.shape[0], 1).squeeze()
            group = Group()
            xi = XX[index]
            XX = np.delete(XX, index, axis=0)  # 删除XX中的一行并重新赋值
            nearest_indexes = self._find_nearst_indexes(xi, XX)
            nearest_datas = None
            mean_vector = None
            if len(nearest_indexes) != 0:
                nearest_datas = None
                # 2.不断进行漂移，中心点达到稳定值
                epos = 1.0
                while (True):
                    nearest_datas = XX[nearest_indexes]
                    mean_vector = self._compute_mean_vector(xi, nearest_datas)
                    xi = mean_vector + xi
                    nearest_indexes = self._find_nearst_indexes(xi, XX)
                    epos = np.abs(sum(mean_vector))
                    if epos < 0.00001: break
                    if len(nearest_indexes) == 0: break
                # 有些博客说在一次漂移过程中 每个漂移点周边的点都需要纳入该类簇中，我觉得不妥，此处不是这样实现的，
                # 只把稳定点周边的数据纳入该类簇中
                group.members = nearest_datas.tolist()
                group.center = xi
                XX = np.delete(XX, nearest_indexes, axis=0)
            else:
                group.center = xi
            # 3.与历史类簇进行距离计算，若小于阈值则加入历史类簇，并更新类簇中心及成员
            for i in range(len(self._groups)):
                h_group = self._groups[i]
                distance = eculide(h_group.center, group.center)
                print(h_group.center, group.center)
                if distance <= self._distance_between_groups:
                    h_group.members = group.members
                    h_group.center = (h_group.center + group.center) / 2
                else:
                    group.name = len(self._groups) + 1
                    self._groups.append(group)
                    break
            if len(self._groups) == 0:
                group.name = len(self._groups) + 1
                self._groups.append(group)
            # 4.从余下的点中重复1-3的计算，直到所有数据完成选取

    def plot_example(self):
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_title("MeanShift Iris Example")
        plt.xlabel("first dim")
        plt.ylabel("third dim")
        legends = []
        cxs = []
        cys = []
        for i in range(len(self._groups)):
            group = self._groups[i]
            members = group.members
            x = [member[0] for member in members]
            y = [member[2] for member in members]
            cx = group.center[0]
            cy = group.center[2]
            cxs.append(cx)
            cys.append(cy)
            ax.scatter(x, y, marker='o')
            # ax.scatter(cx,cy,marker='+',c='r')
            legends.append(group.name)
        plt.scatter(cxs, cys, marker='+', c='k')
        plt.legend(legends, loc="best")
        plt.show()

    def gaussian_kernel(self, distances):
        """
        高斯核函数
        :param distances:
        :param h:
        :return:
        """
        left = 1 / (self._bandwidth * np.sqrt(2 * np.pi))
        right = np.exp(-np.power(distances, 2) / (2 * np.power(self._bandwidth, 2)))
        return left * right


def test_meanshift(use_gk=False):
    data, t, tn = load_data()
    ms = MeanShift(radius=0.66, distance_between_groups=1.4, use_gk=use_gk)
    ms.fit(data)
    ms.plot_example()


test_meanshift(use_gk=True)