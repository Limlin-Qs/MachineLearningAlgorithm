import re


# re的文本替换功能，此间手动转存.csv为.txt文件，放于本项目目录下，
# 然后由本例转换格式并存入maba.txt，便于之后用Apriori算法做关联性分析
with open('mba.txt', 'r+') as f:
    line = re.sub(',', ' ', f.readline())
    # print(line)
    with open('maba.txt', 'a') as fa:
        fa.write(line)
    fa.close()
    while line:
        line = f.readline()
        a = re.sub(',', ' ', line)
        with open('maba.txt', 'a') as fa:
            fa.write(a)
# 注意：多次执行本程序会不断往maba.txt中添加数据，造成数据重复，影响下一步支持度计算
f.close()
fa.close()