# 运行卡住了，还需要继续修改代码
import itertools

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/5/30 18:26
  Software: PyCharm
  Profile: https://zhuanlan.zhihu.com/p/118439868
"""

def load_data(path):
    result = []
    with open(path) as f:
        for line in f:
            line = line.strip('\n')
            # split处理之后返回list，相当于二维数组的形式
            result.append(line.split(","))
    return result


dataset = load_data("Relevance_exp_result_group.txt")
print(len(dataset))
print(dataset[:10])
for i in range(10):
    print(i + 1, dataset[i], sep="->")


items = set(itertools.chain(*dataset))
# 用来保存字符串到编号的映射
str_to_index = {}
# 用来保存编号到字符串的映射
index_to_str = {}
for index, item in enumerate(items):
    str_to_index[item] = index
    index_to_str[index] = item
# 输出结果
print("字符串到编号:", list(str_to_index.items())[:5])
print("编号到字符串:", list(index_to_str.items())[:5])

# 将原始数据进行转换，由字符串映射为数值索引
for i in range(len(dataset)):
    for j in range(len(dataset[i])):
        dataset[i][j] = str_to_index[dataset[i][j]]
for i in range(10):
    print(i + 1, dataset[i], sep="->")


# 生成候选1项集
def buildC1(dataset):
    item1 = set(itertools.chain(*dataset))
    return [frozenset([i]) for i in item1]


c1 = buildC1(dataset)


# 根据候选k项集和最小支持度，生成频繁k项集
def ck_to_lk(dataset, ck, min_support):
    support = {}  # 定义项集-频数字典，用来存储每个项集key对应的频数value
    for row in dataset:
        for item in ck:
            # 判断项集是否在记录中出现
            if item.issubset(row):
                support[item] = support.get(item, 0) + 1
    total = len(dataset)
    return {k: v / total for k, v in support.items() if v / total >= min_support}


L1 = ck_to_lk(dataset, c1, 0.05)


# 频繁K项集组合生成候选K+1项集
def lk_to_ck(lk_list):
    # 保存所有组合之后的候选k+1项集
    ck = set()
    lk_size = len(lk_list)
    if lk_size > 1:
        k = len(lk_list[0])
        for i, j in itertools.combinations(range(lk_size), 2):
            t = lk_list[i] | lk_list[j]
            if len(t) == k + 1:
                ck.add(t)
    return ck


c2 = lk_to_ck(list(L1.keys()))

L2 = ck_to_lk(dataset, c2, 0.05)


# 生成所有频繁项集，从原始数据生成频繁项集
def get_L_all(dataset, min_support):
    c1 = buildC1(dataset)
    L1 = ck_to_lk(dataset, c1, min_support)
    L_all = L1
    Lk = L1
    while len(Lk) > 1:
        lk_key_list = list(Lk.keys())
        ck = lk_to_ck(lk_key_list)
        Lk = ck_to_lk(dataset, ck, min_support)
        if len(Lk) > 0:
            L_all.update(Lk)
        else:
            break
    return L_all


L_all = get_L_all(dataset, 0.05)


# 从频繁项集生成关联规则
def rules_from_item(item):
    # 定义规则左侧的列表
    left = []
    for i in range(1, len(item)):
        left.extend(itertools.combinations(item, i))
    return [(frozenset(le), frozenset(item.difference(le))) for le in left]


rules_from_item(frozenset({1, 2, 3}))


# 根据关联规则，计算置信度，保留符合最小置信度的关联规则
def rules_from_L_all(L_all, min_confidence):
    # 保存所有候选的关联规则
    rules = []
    for lk in L_all:
        if len(lk) > 1:
            rules.extend(rules_from_item(lk))
    result = []
    for left, right in rules:
        support = L_all[left | right]
        confidence = support / L_all[left]
        lift = confidence / L_all[right]
        if confidence >= min_confidence:
            result.append({"左侧": left, "右侧": right, "支持度": support, "置信度": confidence, "提升度": lift})
    return result


rules_from_L_all(L_all, 0.3)


# 最终程序,从原始数据生成关联规则
def apriori(dataset, min_support, min_confidence):
    L_all = get_L_all(dataset, min_support)
    rules = rules_from_L_all(L_all, min_confidence)
    return rules


# rules = apriori(dataset, 0.05, 0.3)
rules = apriori(dataset, 0.5, 0.3)

# 最后我们把代码转换成真实的对象名称，同时为了能清晰呈现数据，用dataframe对象进行展示
import pandas as pd


def change(item):
    li = list(item)
    for i in range(len(li)):
        li[i] = index_to_str[li[i]]
    return li


df = pd.DataFrame(rules)
df = df.reindex(["左侧", "右侧", "支持度", "置信度", "提升度"], axis=1)
df["左侧"] = df["左侧"].apply(change)
df["右侧"] = df["右侧"].apply(change)
