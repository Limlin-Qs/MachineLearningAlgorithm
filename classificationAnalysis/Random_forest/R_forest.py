# -*- coding: UTF-8 -*-
# 待调整
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import datetime

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/3/11 17:18
  Software: PyCharm
  Profile: https://blog.csdn.net/cxmscb/article/details/53541224
"""

estimators = {}

# criterion: 分支的标准(gini/entropy)
estimators['tree'] = tree.DecisionTreeClassifier(criterion='gini', random_state=8)  # 决策树

# n_estimators: 树的数量
# bootstrap: 是否随机有放回
# n_jobs: 可并行运行的数量
estimators['forest'] = RandomForestClassifier(n_estimators=20, criterion='gini', bootstrap=True, n_jobs=2,
                                              random_state=8)  # 随机森林

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df['species'] = pd.Categorical(iris.target, iris.target_names)
df.head()

train, test = df[df['is_train'] == True], df[df['is_train'] == False]
print(train, test)
# for k in estimators.keys():
#     start_time = datetime.datetime.now()
#     print('----%s----' % k)
#     estimators[k] = estimators[k].fit(X_train, y_train)
#     pred = estimators[k].predict(X_test)
#     print(pred[:10])
#     print("%s Score: %0.2f" % (k, estimators[k].score(X_test, y_test)))
#     scores = cross_val_score(estimators[k], X_train, y_train, scoring='accuracy', cv=10)
#     print("%s Cross Avg. Score: %0.2f (+/- %0.2f)" % (k, scores.mean(), scores.std() * 2))
#     end_time = datetime.datetime.now()
#     time_spend = end_time - start_time
#     print("%s Time: %0.2f" % (k, time_spend.total_seconds()))
