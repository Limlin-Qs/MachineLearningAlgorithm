# -*- coding: UTF-8 -*-
import numpy as np
from collections import defaultdict
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/9/6 21:52
  Software: PyCharm
  Profile: https://www.cnblogs.com/zyb993963526/p/13778442.html#autoid-0-3-0 
  GloVe fastText
"""

settings = {'window_size': 2,
            'n': 3,
            'epochs': 500,
            'learning_rate': 0.01}


class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

    def generate_training_data(self, corpus):
        '''
        :param settings: 超参数
        :param corpus: 语料库
        :return: 训练样本
        '''
        word_counts = defaultdict(int)  # 当字典中不存在时返回0
        for row in corpus:
            for word in row.split(' '):
                word_counts[word] += 1
        self.v_count = len(word_counts.keys())  # v_count:不重复单词数
        self.words_list = list(word_counts.keys())  # words_list:单词列表
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))  # {单词:索引}
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))  # {索引:单词}

        training_data = []
        for sentence in corpus:
            tmp_list = sentence.split(' ')  # 语句单词列表
            sent_len = len(tmp_list)  # 语句长度
            for i, word in enumerate(tmp_list):  # 依次访问语句中的词语
                w_target = self.word2onehot(tmp_list[i])  # 中心词ont-hot表示
                w_context = []  # 上下文
                for j in range(i - self.window, i + self.window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(tmp_list[j]))
                training_data.append([w_target, w_context])  # 对应了一个训练样本

        return training_data

    def word2onehot(self, word):
        """
        :param word: 单词
        :return: ont-hot
        """
        word_vec = [0 for i in range(0, self.v_count)]  # 生成v_count维度的全0向量
        word_index = self.word_index[word]  # 获得word所对应的索引
        word_vec[word_index] = 1  # 对应位置位1
        return word_vec

    def train(self, training_data):
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))  # 随机生成参数矩阵
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
        for i in range(self.epochs):
            self.loss = 0

            for data in training_data:
                w_t, w_c = data[0], data[1]  # w_t是中心词的one-hot，w_c是window范围内所要预测此的one-hot
                y_pred, h, u = self.forward_pass(w_t)

                train_loss = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)  # 每个预测词都是一对训练数据，相加处理
                self.back_prop(train_loss, h, w_t)

                for word in w_c:
                    self.loss += - np.dot(word, np.log(y_pred))

            print('Epoch:', i, "Loss:", self.loss)

    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_pred = self.softmax(u)
        return y_pred, h, u

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # 防止上溢和下溢。减去这个数的计算结果不变
        return e_x / e_x.sum(axis=0)

    def back_prop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.dot(self.w2, e.T).reshape(-1)
        self.w1[x.index(1)] = self.w1[x.index(1)] - (self.lr * dl_dw1)  # x.index(1)获取x向量中value=1的索引，只需要更新该索引对应的行即可
        self.w2 = self.w2 - (self.lr * dl_dw2)


if __name__ == '__main__':
    # epochs指的就是训练过程接中数据将被“轮”多少次”;训练过程中当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一个epoch，网络会在每个epoch结束时报告关于模型学习进度的调试信息。
    corpus = ['There are many mangoes.']
    w2v = word2vec()
    training_data = w2v.generate_training_data(corpus)
    w2v.train(training_data)

'''
train loss 不断下降，test loss不断下降，说明网络仍在学习;（最好的）

train loss 不断下降，test loss趋于不变，说明网络过拟合;（max pool或者正则化）

train loss 趋于不变，test loss不断下降，说明数据集100%有问题;（检查dataset）

train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;（减少学习率）

train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。（最不好的情况）
'''