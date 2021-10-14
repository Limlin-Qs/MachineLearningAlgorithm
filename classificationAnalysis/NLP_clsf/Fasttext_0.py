# -*- coding: UTF-8 -*-
import logging
import fasttext
import pandas as pd
import codecs

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/9/18 23:22
  Software: PyCharm
  Profile: https://blog.csdn.net/asd136912/article/details/80068241?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link
"""

basedir = '/Users/derry/Desktop/Data/'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 训练
classifier = fasttext.supervised(basedir + "news.data.seg.train", basedir + "news.dat.seg.model", label_prefix="__label__", word_ngrams=3, bucket=2000000)

# 测试并输出 F-score
result = classifier.test(basedir + "news.data.seg.test")
print(result.precision * result.recall * 2 / (result.recall + result.precision))

# 读取验证集
validate_texts = []
with open(basedir + 'news.data.seg.validate', 'r', encoding='utf-8') as infile:
    for line in infile:
        validate_texts += [line]

# 预测结果
labels = classifier.predict(validate_texts)

# 结果文件
result_file = codecs.open(basedir + "result.txt", 'w', 'utf-8')

validate_data = pd.read_table(basedir + 'News_info_validate.txt', header=None, error_bad_lines=False)
validate_data.drop([2], axis=1, inplace=True)
validate_data.columns = ['id', 'text']

# 写入
for index, row in validate_data.iterrows():
    outline = row['id'] + '\t' + labels[index][0] + '\tNULL\tNULL\n'
    result_file.write(outline)
    result_file.flush()

result_file.close()
