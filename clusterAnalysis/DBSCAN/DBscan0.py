# -*- coding: UTF-8 -*-
# 用calinski_harabaz_score方法评价聚类效果的好坏。大概是类间距除以类内距，这个值越大越好
from sklearn.metrics import calinski_harabasz_score
import DBSCAN.DBscan1 as clu_data_helper
from sklearn.cluster import DBSCAN
import time

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2020/12/1 21:25
  Software: PyCharm
  Profile: 基于高密度连通区域的基于密度的聚类
  https://github.com/dylgithub/cluster
"""

X = clu_data_helper.get_sen_vec(90, 50)
_, y = clu_data_helper.get_content()
start = time.time()
# for i in [11,12,13,14,15]:
#     for data in [0.9,0.8,1.0]:
y_pred = DBSCAN(eps=0.8, min_samples=12).fit_predict(X)
n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
# print("eps",data,"min_samples",i,'n_clusters_',n_clusters_,"Calinski-Harabasz Score",calinski_harabaz_score(X,
# y_pred))
end = time.time()
print(end - start)
print("Calinski-Harabasz Score", calinski_harabasz_score(X, y_pred))
