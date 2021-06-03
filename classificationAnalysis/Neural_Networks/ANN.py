# -*- coding: UTF-8 -*-

import numpy as np

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/5/21 21:27
  Software: PyCharm
  Profile: https: // blog.csdn.net / jianfpeng241241 / article / details / 88881208
"""
# 双曲正切函数
def tanh(x):
    return np.tanh(x)


# 双曲正切函数导数
def tanh_derivative(x):
    return 1 - np.tanh(x) * np.tanh(x)


# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid函数导数
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 神经网络实现类
class NeuralNetwork():

    # layers:神经网络
    # activation:激励函数
    # learning_rate 学习率
    # loss_threshold:损失阀值
    # epoch:最大训练次数
    def __init__(self, layers=[], activation="sigmoid",
                 learning_rate=0.1, epoch=1000, loss_threshold=0.01):
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == "tanh":
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        self.layers = layers
        self.init_weights(layers)
        self.init_bias(layers)
        self.init_nodes()
        self.init_errors()
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.loss_threshold = loss_threshold

    # 校验二维数组
    def valiad_two_array(self, data):
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list) == False or len(data[0]) == 0:
                raise RuntimeError("参数错误,请传一个不为空的二维数组")
        else:
            raise RuntimeError("参数错误,请传一个不为空的二维数组")

    # 校验一维数组
    def valid_one_array(self, data):
        if isinstance(data, list) == False or len(data) == 0:
            raise RuntimeError("参数错误,请传入一个不为空的一维数组")

    # 初始化权重
    def init_weights(self, layers):
        self.weights = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.random((layers[i - 1], layers[i])))

    # 初始化偏向
    def init_bias(self, layers):
        self.bias = []
        for i in range(1, len(layers)):
            self.bias.append(np.random.random((layers[i], 1)))

    # 训练模型
    def fit(self, data):
        self.valiad_two_array(data)
        self.counter = 0
        for i in range(len(data)):
            self.training_data(data[i], i)

    # 预测数据
    def predict(self, data):
        self.valiad_two_array(data)
        counter = 0
        for one_data in data:
            self.forward_propagation(one_data)
            predict = self.nodes[len(self.layers) - 1]
            for i in range(len(predict)):
                predict[i] = self.handle_by_threshold(predict[i])
            print("predict[{}] = {} ".format(counter, predict))
            counter += 1

    # 根据阀值处理数据
    def handle_by_threshold(self, data):
        if data >= 0.5:
            return 1
        else:
            return 0

    # 一次训练流程
    def training_data(self, one_data, number):
        self.loss = 1
        one_training_counter = 0
        while self.loss > self.loss_threshold and one_training_counter < self.epoch:
            self.counter += 1
            one_training_counter += 1
            self.forward_propagation(one_data)
            self.back_propagation_error(one_data)
            self.back_propagation_update_weights()
            self.back_propagation_update_bias()
            # print("总次数{},第{}行数据,当前次数:{},\n{}".
            #       format(self.counter, number, one_training_counter, self.get_obj_info()))

    # 获取对象信息
    def get_obj_info(self):
        info = "\n\n weights: " + str(self.weights) \
               + "\n\n bais: " + str(self.bias) \
               + "\n\n nodes: " + str(self.nodes) \
               + "\n\n errors: " + str(self.errors) \
               + "\n\n loss: " + str(self.loss)
        return info

    # 输出层错误计算
    # out:经过激励函数计算后的结果
    # predict:原始预测的结果
    def calculate_out_layer_error(self, out, predict):
        return out * (1 - out) * (predict - out)

    # 隐藏层错误计算
    # out:经过激励函数计算后的结果
    # errors:下一层所有节点的损失合计
    def calculate_hidden_layer_error(self, out, errors):
        return out * (1 - out) * errors

    # 前向传播,递归得到每一个节点的值
    # one_row_data:一行数据
    # counter: 计数器
    def forward_propagation(self, one_row_data, counter=0):
        if counter == 0:
            input = self.get_input(one_row_data)
            self.input = input
            for i in range(len(self.input)):
                self.nodes[0][i] = self.input[i]
            counter += 1
        if counter == len(self.layers):
            return
        current_nodes = self.nodes[counter]
        pre_nodes = self.nodes[counter - 1]
        for i in range(len(current_nodes)):
            current_value = 0
            for j in range(len(pre_nodes)):
                pre_node = pre_nodes[j]
                pre_weights = self.weights[counter - 1][j][i]
                current_value += pre_node * pre_weights
            current_bias = self.bias[counter - 1][i][0]
            current_value = (current_value + current_bias)[0]
            current_node = self.activation(current_value)
            current_nodes[i] = current_node
        self.forward_propagation(one_row_data, counter + 1)

    # 得到特征值
    def get_input(self, one_row_data):
        return one_row_data[:self.layers[0]]

    # 根据特征值真实结果
    def get_out(self, one_row_data):
        return one_row_data[self.layers[0]:]

    # 后向传播,得到误差
    def back_propagation_error(self, one_row_data, counter=-1):
        if counter == -1:  # 第一次进入方法，初始化
            counter = len(self.layers) - 1
            out = self.get_out(one_row_data)
            self.out = out
        if counter == 0:  # 遍历集合(第一层输入层不计算损失)
            return
        current_nodes = self.nodes[counter]
        if counter == len(self.layers) - 1:  # 输出层损失计算
            loss = 0
            for i in range(len(current_nodes)):
                current_node = current_nodes[i]
                predict = self.out[i]
                error_value = self.calculate_out_layer_error(current_node, predict)
                self.errors[counter][i] = error_value
                loss += pow(predict - current_node, 2)
            self.loss = loss
        else:  # 隐藏层损失计算
            next_errors = self.errors[counter + 1]
            for i in range(len(current_nodes)):
                current_node = current_nodes[i]
                errors = 0
                for j in range(len(next_errors)):
                    error = next_errors[j]
                    weight = self.weights[counter][i]
                    errors += error * weight
                error_value = self.calculate_hidden_layer_error(current_node, errors)
                self.errors[counter][i] = error_value
        self.back_propagation_error(one_row_data, counter - 1)

    # 后向传播,更新权重
    def back_propagation_update_weights(self):
        for i in reversed(range(len(self.layers) - 1)):
            current_nodes = self.nodes[i]
            errors = self.errors[i + 1]
            for j in range(len(current_nodes)):
                for m in range(len(errors)):
                    error = errors[m]
                    current_node = current_nodes[j]
                    weight = self.weights[i][j][m]
                    weight_delta = self.learning_rate * error * current_node
                    update_weight = weight + weight_delta
                    self.weights[i][j][m] = update_weight

    # 后向传播,更新偏向
    def back_propagation_update_bias(self):
        for i in reversed(range(len(self.layers) - 1)):
            bias = self.bias[i]
            for j in range(len(bias)):
                error = self.errors[i + 1][j]
                bias_delta = self.learning_rate * error
                bias[j] += bias_delta

    # 设置权重
    def set_weights(self, weights):
        self.weights = weights

    # 设置偏向
    def set_bias(self, bias):
        self.bias = bias

    # 初始化所有节点（节点值设置为一个随机数)
    def init_nodes(self):
        self.nodes = []
        for i in range(len(self.layers)):
            self.nodes.append(np.random.random((self.layers[i], 1)))

    # 初始化所有节点损失值(损失值设置为一个随机数)
    def init_errors(self):
        self.errors = []
        for i in range(len(self.layers)):
            self.errors.append(np.random.random((self.layers[i], 1)))
