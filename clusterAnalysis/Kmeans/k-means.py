# 步骤：
# 1、首先随机选取k个中心点
# 2、然后每个点分配到最近的类中心点（欧式距离），形成k个类，然后重新计算每个类的中心点（均值）
# 3、重复第二步，直到类不发生变化，或者你也可以设置最大迭代次数，这样即使类中心点发生变化，但是只要达到最大迭代次数就会结束。
# 此原始数据没有结果数据。


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2020/12/1 20:37
  Software: PyCharm
  Profile: 划分；基于形心的技术
"""

# 没有找到对应数据，独立的程序，更换了数据集，发现里面的数据必须能转化成float类型，于是换了cvs
bankData = pd.read_csv('银行数据', sep=',')
print(bankData)
x_train = bankData[['openPrice']]
print(x_train)
# 数据归一化
ss = MinMaxScaler()
x_train = ss.fit_transform(x_train)
# 构造聚类器。
# precompute_distances: 是否需要提前计算距离，这个参数会在空间和时间之间做权衡，如果是True 会把整个距离矩阵都放到内存中，
# auto 会默认在数据样本大于featurs*samples 的数量大于12e6 的时候False,False 时核心实现的方法是利用Cpython 来实现的
# init: 初始簇中心的获取方法
teamclf = KMeans(n_clusters=3, init='k-means++', precompute_distances='auto')
# 聚类
teamclf.fit(x_train)

y_predict = teamclf.predict(x_train)
print(y_predict)
print(pd.Series(y_predict, name='echelon'))
df_x_train = pd.DataFrame(x_train)
teamResult = pd.merge(df_x_train, pd.Series(y_predict, name='聚类'), how='left', left_index=True, right_index=True)
print(teamResult)
# 聚类结果评估，用Calinski-Harabasz Index评估的聚类分数
metrics.calinski_harabaz_score(X, y_pred)
# 为什么输出全是小数？？
# 未能画出图
