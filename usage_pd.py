import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# 在处理数据时遇到NAN值的几率还是比较大的，有的时候需要对数据值是否为nan值做判断 pd.isnull(np.nan)
# s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)

# 用含日期时间索引与标签的 Numpy数组生成Dataframe。np.random.randn(天数/条数，数据的列数），index索引列为日期，columns为数据列名
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
df0 = df.iloc[0].div(0.0001)  # 第二行数据同除0.1，放大10倍，问题是不能输出矩阵
print(df)
print(df0)
# df.info()

# df.to_numpy()
# df.head()
# df.tail(3)
# df.describe()  # 统计摘要
# df.T  # 转置数据
# df.sort_index(axis=1, ascending=False)  # 按轴排序
# df.sort_values(by='B')  # 按值排序
# df['A'] # 选择单列，与df.A同效
# df[0:3]  # 行切片取数据
# df['20130102':'20130104']
# df.loc[dates[0]]  # 用标签提取一行数据，提取第一行
# df.loc['20130102':'20130104', ['A', 'B']]  # 用标签切片，包括行与列
# df.loc['20130102', ['A', 'B']]  # 返回对象降维
# df.loc[dates[0], 'A']  # 提取标量值
# df.at[dates[0], 'A']
# df.iloc[3]  # 按照整数位置选择
# df.iloc[3:5, 0:2]  # 用整数切片
# df.iloc[[1, 2, 4], [0, 2]]  # 用整数按位置切片
# df.iloc[1:3, :]  # 显示整行切片
# df.iloc[:, 1:3]  # 显示整列切片
# df.iloc[1, 1]  # 显示提取值
# df.iat[1, 1]
# df[df.A > 0]  # 布尔索引，用单列的值选择数据
# df[df > 0]  # 选择dataframe中符合条件的值，所有大于0的
# df2 = df.copy()  # 将df数据复制到df2
# df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']  # 添加一列数据
# df2[df2['E'].isin(['two', 'four'])]  # isin（）筛选两行数据
# s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6)  #用索引自动对齐新增列的数据，加了第七列
# df.at[dates[0], 'A'] = 0  # 按标签赋值
# df.iat[0, 1] = 0  # 按位置赋值
# df.loc[:, 'D'] = np.array([5] * len(df))  # 按Numpy数组赋值
# df2[df2 > 0] = -df2  # 按where条件赋值
# df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])  # reindex可以更改、添加、删除指定轴的索引，并返回数据副本，即不更改元数据
# df1.loc[dates[0]:dates[1], 'E'] = 1
# # Pandas主要用np.nan表示缺失数据。计算时，默认不包含空值
# df1.dropna(how='any')  # 删除所有含缺失值的行
# df1.fillna(value=5)  # 填充缺失值
# pd.isna(df1) # 提取nan值的布尔掩码
# df.mean()  # 描述性统计
# df.mean(1)  # 在另一个轴上执行同样的操作
# df.apply(np.cumsum)  # 不同行的数据运算
# df.apply(lambda x: x.max() - x.min())
# s = pd.Series(np.random.randint(0, 7, size=10))  # 随机生成十个数，范围0-7
# s.value_counts()  # 直方图统计，看看有多少个y值达到4，或者更多更少
# s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])  #
# s.str.lower()  # 模式匹配将过滤所有字符转小写，默认正则
# df = pd.DataFrame(np.random.randn(10, 4))  # 来个十行四列的df，数字随机小数
# pieces = [df[:3], df[3:7], df[7:]]  # 分解为多组
# pd.concat(pieces)  # 合并多组
# left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})  # 数据库风格的互联，全部的排列组合
# right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})  # 还有就是生成数据的方式，学习一下
# pd.merge(left, right, on='key')
# left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})  # 另一组数据，其中key值互斥看一下。
# right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})

# 用 Series 字典对象生成 DataFrame
# df2 = pd.DataFrame({'A': 1.,
#                     'B': pd.Timestamp('20130102'),
#                     'C': pd.Series(1, index=list(range(4)), dtype='float32'),
#                     'D': np.array([3] * 4, dtype='int32'),
#                     'E': pd.Categorical(["test", "train", "test", "train"]),
#                     'F': 'foo'})
# print(df2)
# df2.dtypes   # 输出每列数据类型

# 从cvs表读数据，读取开始的五条

# data = pd.read_csv('BreadBasket_DMS.csv')
# df = data.loc[:, ['Transaction', 'Item']]
# print(df)
# df['name'] = df['name'].fillna('WU')  # 内容替换

# set a numeric id for use as an index for examples.
# data['Transaction'] = [random.randint(0, 1000) for x in range(data.shape[0])]
# data.head(5)
# print(data.head(5))

# plt.subplot(221)
# ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
# ts = ts.cumsum()
# ts.plot()  # 用dataframe.plot画完图还要用plt.show打开视窗查看，他不会自动弹
# plt.show()

# data1 = pd.read_excel(r'apriori算法实现.xlsx', index=False)  # 用pandas从excel中读数据

# 如何将dataframe的文件转化为txt和csv，csv只能转化有规律的表，完整列的那种
# data = pd.read_csv('goods.csv')
# # print(dataa)
# data.to_csv('test1.txt', sep='\t', index=False)
# data.to_csv('threecolumn1.csv')

import numpy as np
import pandas as pd


submit1_path = "goods.csv"
df = pd.read_csv(submit1_path)
df.to_csv('a.txt', sep='\t', index=False)