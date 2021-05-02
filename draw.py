from sklearn.datasets import make_regression, load_boston
import matplotlib.pyplot as mp
import pandas as pd


# 创建数据
X, y, coef = make_regression(n_features=1, noise=9, coef=True)
x = X.reshape(-1)
# 可视化
mp.scatter(x, y, c='g', alpha=0.3)
mp.plot(x, coef * x)
mp.show()
