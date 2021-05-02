# -*- coding: UTF-8 -*-
from sklearn.cluster import AgglomerativeClustering
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/3/14 11:15
  Software: PyCharm
  Profile:
"""
n_clusters = 2
scale_features = 4
model = AgglomerativeClustering(n_clusters)
model.fit_predict(scale_features)