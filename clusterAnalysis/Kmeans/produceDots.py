import numpy as np
import copy

# 自动生成数据并存储，但是生成的数据太方了，发散性不好
choosetype = 1  # 1 表示有序点 其他表示随机点
# data = [[3.5, -3.5], [3.5, 3.5], [-3.5, 3.5], [-3.5, -3.5]]  # 四类点的中心
data = [[7.5, -0.5]]  # 四类点的中心

totalnum = 3000  # 产生点的个数
file_path = "test3"  # 保存路径

fopen = open(file_path, 'w')  # 追加的方式读写打开

for i in range(totalnum):
    if choosetype == 1:
        datatemp = copy.deepcopy(data)
        choose = datatemp[i % 1]
        n1 = 2 * np.random.random(1) - 1
        n2 = 2 * np.random.random(1) - 1
        # n1 = 3 * np.random.random(1) - 1
        # n2 = 5 * np.random.random(1) - 1
        choose[0] = choose[0] + n1
        choose[1] = choose[1] + n2
        # str(round(choose[0][0], 3)), 3表示保留小数位，\t为制表符
        fopen.writelines(str(round(choose[0][0], 3)) + "\t" + str(round(choose[1][0], 3)) + "\n")
    else:
        n1 = 4 * np.random.random(1) - 2
        n2 = 4 * np.random.random(1) - 2

        fopen.writelines(str(round(n1[0], 3)) + "\t" + str(round(n2[0], 3)) + "\n")
fopen.close()
