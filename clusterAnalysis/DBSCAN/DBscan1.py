from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2020/12/1 21:25
  Software: PyCharm
  Profile: 基于高密度连通区域的基于密度的聚类
"""

# 运行正常
spiral = pd.read_table("test1", delimiter=' ')
# X = wine.data[:, 3:5]  # 只要后两个维度
X = spiral.iloc[:, [1, 2]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# cluster the data into five clusters
dbscan = DBSCAN(eps=0.123, min_samples=2)
clusters = dbscan.fit_predict(X_scaled)

# plot the cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
