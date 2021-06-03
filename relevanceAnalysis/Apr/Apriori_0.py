# 小论文实验比较用，调用算法库中已有的算法，可以输出scl三个关联度指标
from apyori import apriori
import time

data = [['豆奶', '莴苣'],
        ['莴苣', '尿布', '葡萄酒', '甜菜'],
        ['豆奶', '尿布', '葡萄酒', '橙汁'],
        ['莴苣', '豆奶', '尿布', '葡萄酒'],
        ['莴苣', '豆奶', '尿布', '橙汁']]
start = time.time()
# 提升度，即 P（B | A） / P(B)，称之为 A 条件对 B事务的提升度，即有 A 作为前提，对 B 出现的概率有什么样的影响，
result = apriori(transactions=data)
end =time.time()
print(end - start)
print(result)