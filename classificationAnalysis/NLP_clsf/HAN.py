# -*- coding: UTF-8 -*-
import os
import numpy as np
import tensorflow as tf
from eval.evaluate import accuracy, Macro_F1
from tensorflow.contrib import slim
from loss.loss import cross_entropy_loss
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/9/6 22:15
  Software: PyCharm
  Profile: 多层注意力模型算法，该模型确实能够捕获到有助于对文本进行分类的词汇。
"""



class HAN(object):
    def __init__(self,
                 num_classes,
                 seq_length,
                 vocab_size,
                 embedding_dim,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 epoch,
                 dropout_keep_prob,
                 rnn_type,
                 hidden_dim,
                 num_sentences):
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.learning_decay_rate = learning_decay_rate
        self.learning_decay_steps = learning_decay_steps
        self.epoch = epoch
        self.dropout_keep_prob = dropout_keep_prob
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_sentences = num_sentences
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.model()

    def model(self):
        # 词向量映射
        with tf.name_scope("embedding"):
            input_x = tf.split(self.input_x, self.num_sentences, axis=1)
            # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
            input_x = tf.stack(input_x, axis=1)
            embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim])
            # [None,num_sentences,sentence_length,embed_size]
            embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)
            # [batch_size*num_sentences,sentence_length,embed_size]
            sentence_len = int(self.seq_length / self.num_sentences)
            embedding_inputs_reshaped = tf.reshape(embedding_inputs, shape=[-1, sentence_len, self.embedding_dim])

        # 词汇层
        with tf.name_scope("word_encoder"):
            (output_fw, output_bw) = self.bidirectional_rnn(embedding_inputs_reshaped, "word_encoder")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            word_hidden_state = tf.concat((output_fw, output_bw), 2)

        with tf.name_scope("word_attention"):
            # [batch_size*num_sentences, hidden_size * 2]
            sentence_vec = self.attention(word_hidden_state, "word_attention")

        # 句子层
        with tf.name_scope("sentence_encoder"):
            # [batch_size,num_sentences,hidden_size*2]
            sentence_vec = tf.reshape(sentence_vec, shape=[-1, self.num_sentences, self.hidden_dim * 2])
            output_fw, output_bw = self.bidirectional_rnn(sentence_vec, "sentence_encoder")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            sentence_hidden_state = tf.concat((output_fw, output_bw), 2)

        with tf.name_scope("sentence_attention"):
            # [batch_size, hidden_size * 2]
            doc_vec = self.attention(sentence_hidden_state, "sentence_attention")

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(doc_vec, self.dropout_keep_prob)

        # 输出层
        with tf.name_scope("output"):
            # 分类器
            self.logits = tf.layers.dense(h_drop, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")  # 预测类别

        # 损失函数
        self.loss = cross_entropy_loss(logits=self.logits, labels=self.input_y)

        # 优化函数
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.learning_decay_steps, self.learning_decay_rate,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.optim = slim.learning.create_train_op(total_loss=self.loss, optimizer=optimizer, update_ops=update_ops)

        # 准确率
        self.acc = accuracy(logits=self.logits, labels=self.input_y)
        # self.acc = Macro_F1(logits=self.logits,labels=self.input_y)

    def rnn_cell(self):
        """获取rnn的cell，可选RNN、LSTM、GRU"""
        if self.rnn_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim)
        elif self.rnn_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        elif self.rnn_type == "gru":
            return tf.nn.rnn_cell.GRUCell(self.hidden_dim)
        else:
            raise Exception("rnn_type must be vanilla、lstm or gru!")

    def bidirectional_rnn(self, inputs, name):
        with tf.variable_scope(name):
            fw_cell = self.rnn_cell()
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = self.rnn_cell()
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                             cell_bw=bw_cell,
                                                                             inputs=inputs,
                                                                             dtype=tf.float32)
        return output_fw, output_bw

    def attention(self, inputs, name):
        with tf.variable_scope(name):
            # 采用general形式计算权重
            hidden_vec = tf.layers.dense(inputs, self.hidden_dim * 2, activation=tf.nn.tanh, name='w_hidden')
            u_context = tf.Variable(tf.truncated_normal([self.hidden_dim * 2]), name='u_context')
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(hidden_vec, u_context),
                                                axis=2, keep_dims=True), dim=1)

            # 对隐藏状态进行加权
            attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)

        return attention_output

    def fit(self, train_x, train_y, val_x, val_y, batch_size):
        # 创建模型保存路径
        if not os.path.exists('./saves/han'): os.makedirs('./saves/han')
        if not os.path.exists('./train_logs/han'): os.makedirs('./train_logs/han')

        # 开始训练
        train_steps = 0
        best_val_acc = 0
        # summary
        tf.summary.scalar('val_loss', self.loss)
        tf.summary.scalar('val_acc', self.acc)
        merged = tf.summary.merge_all()

        # 初始化变量
        sess = tf.Session()
        writer = tf.summary.FileWriter('./train_logs/han', sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())

        for i in range(self.epoch):
            batch_train = self.batch_iter(train_x, train_y, batch_size)
            for batch_x, batch_y in batch_train:
                train_steps += 1
                feed_dict = {self.input_x: batch_x, self.input_y: batch_y}
                _, train_loss, train_acc = sess.run([self.optim, self.loss, self.acc], feed_dict=feed_dict)

                if train_steps % 1000 == 0:
                    feed_dict = {self.input_x: val_x, self.input_y: val_y}
                    val_loss, val_acc = sess.run([self.loss, self.acc], feed_dict=feed_dict)

                    summary = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(summary, global_step=train_steps)

                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        saver.save(sess, "./saves/han/", global_step=train_steps)

                    msg = 'epoch:%d/%d,train_steps:%d,train_loss:%.4f,train_acc:%.4f,val_loss:%.4f,val_acc:%.4f'
                    print(msg % (i, self.epoch, train_steps, train_loss, train_acc, val_loss, val_acc))

        sess.close()

    def batch_iter(self, x, y, batch_size=32, shuffle=True):
        """
        生成batch数据
        :param x: 训练集特征变量
        :param y: 训练集标签
        :param batch_size: 每个batch的大小
        :param shuffle: 是否在每个epoch时打乱数据
        :return:
        """
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_len))
            x_shuffle = x[shuffle_indices]
            y_shuffle = y[shuffle_indices]
        else:
            x_shuffle = x
            y_shuffle = y
        for i in range(num_batch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_len)
            yield (x_shuffle[start_index:end_index], y_shuffle[start_index:end_index])

    def predict(self, x):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./saves/han/')
        saver.restore(sess, ckpt.model_checkpoint_path)

        feed_dict = {self.input_x: x}
        logits = sess.run(self.logits, feed_dict=feed_dict)
        y_pred = np.argmax(logits, 1)
        return y_pred