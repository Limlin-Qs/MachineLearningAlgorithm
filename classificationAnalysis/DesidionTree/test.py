import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# 一般情况将K折交叉验证用于模型调优，找到使得模型泛化性能最优的超参值。


X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 2, 3, 4])

data = pd.read_csv(r"dataset.txt")
list2 = np.array(data)
# print(list2)
kf = KFold(n_splits=10)
# 2折交叉验证，将数据分为两份即前后对半分，每次取一份作为test集
for train_index, test_index in kf.split(list2):
    print('train_index', train_index, 'test_index', test_index)
    # train_index与test_index为下标
    train_X = list2[train_index]
    test_X = list2[test_index]
print("train_X", train_X)
print("test_X", test_X)