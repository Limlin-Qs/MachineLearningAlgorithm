# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/9/18 23:09
  Software: PyCharm
  Profile:
"""

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
LinearDiscriminantAnalysis()
print(clf.predict([[0.8, 1]]))
