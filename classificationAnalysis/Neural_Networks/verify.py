# -*- coding: UTF-8 -*-
from Neural_Networks.ANN import NeuralNetwork
import numpy as np

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/5/21 21:28
  Software: PyCharm
  Profile: https: // blog.csdn.net / jianfpeng241241 / article / details / 88881208
"""

layers = [3, 2, 1]
nn = NeuralNetwork(layers)

weights = np.array([[[0.2, -0.3], [0.4, 0.1], [-0.5, 0.2]], [[-0.3], [-0.2]]])
nn.set_weights(weights)

bias = np.array([[[-0.4], [0.2]], [[0.1]]])
nn.set_bias(bias)

data = [
    [1, 0, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 1, 1],
    [0, 0, 0, 0],
    [1, 0, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 1, 1],
    [0, 0, 0, 0],
    [1, 0, 1, 1],
    [0, 0, 0, 0],
    [1, 0, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
]
nn.fit(data)

predict_data = [[1, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]]
predict_result = nn.predict(predict_data)
