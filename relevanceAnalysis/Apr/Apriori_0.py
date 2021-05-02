from apyori import apriori


data = [['豆奶', '莴苣'],
        ['莴苣', '尿布', '葡萄酒', '甜菜'],
        ['豆奶', '尿布', '葡萄酒', '橙汁'],
        ['莴苣', '豆奶', '尿布', '葡萄酒'],
        ['莴苣', '豆奶', '尿布', '橙汁']]

# 提升度，即 P（B | A） / P(B)，称之为 A 条件对 B事务的提升度，即有 A 作为前提，对 B 出现的概率有什么样的影响，
result = list(apriori(transactions=data))
print(result)
