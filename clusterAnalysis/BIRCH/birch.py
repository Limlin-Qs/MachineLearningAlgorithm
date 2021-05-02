# -*- coding: UTF-8 -*-
from sklearn.cluster import Birch
from sklearn.metrics import calinski_harabasz_score
from DBSCAN import DBscan1 as clu_data_helper
import matplotlib.pyplot as plt
import time

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2020/12/1 21:02
  Software: PyCharm
  Profile: 在微簇上宏聚类；利用层次结构的平衡迭代规约和聚类，为大量数值数据聚类设计
  https://github.com/dylgithub/cluster
"""


def get_right_num(y, y_pred):
    num = 0
    for i, j in zip(y, y_pred):
        if i == j:
            num = num + 1
    return num


X = clu_data_helper.get_sen_vec(90, 50)
_, y = clu_data_helper.get_content()
start = time.time()
# for i in [2,3,4,5]:
#     for data in [0.2,0.3,0.4,0.5]:
#         for j in [30,40,50,60,70]:
y_pred = Birch(n_clusters=2, threshold=0.2, branching_factor=70).fit_predict(X)
# end=time.time()
# print(end-start)
print("Calinski-Harabasz Score", calinski_harabasz_score(X, y_pred))
# print('i',i,'data','j',j,data,"Calinski-Harabasz Score",calinski_harabaz_score(X, y_pred))
# print(get_right_num(y,y_pred))
# plt.scatter(X[:,0], X[:,1],c=y_pred)
# plt.show()
