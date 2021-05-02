# -*- coding: UTF-8 -*-
# 运行正常，可以采纳参考
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/3/11 20:06
  Software: PyCharm
  Profile: https://zhuanlan.zhihu.com/p/44870620
"""

from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target # 有三个类别，多分类问题

# split data into training and test sets
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',
                               n_estimators=10,
                               random_state=1,
                               n_jobs=2)
forest.fit(X_train, y_train)
print(forest.score(X_train, y_train)) # 注意这里自带score函数的传参
print(forest.score(X_test, y_test))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, forest.predict(X_train))) # 这是metrics包的分数函数，注意传参
print(accuracy_score(y_test, forest.predict(X_test)))