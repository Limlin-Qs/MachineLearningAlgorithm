from pymysql import *
import re
import csv

# 年龄分段统计，性别统计，销售数据统计，生成mba.csv
# 用于存放统计需要的顾客数据
LIST_COSTOMER = []


def get_data():
    # 共有三十九名顾客，整理每人的购物篮数据（MBA_dataset)，放入文件mba.csv
    csvfile = open('mba.csv', 'w', newline='')
    writer = csv.writer(csvfile)

    cnn = Connect(
        host='127.0.0.1',
        port=3306,
        user='lin',
        passwd='123456',
        db='links',
        charset='utf8'  # 不能加-，如utf-8就会出错
    )
    cur = cnn.cursor()
    for x in range(9900000984, 9900001023):
        x = str(x)
        sql = "SELECT 商品码 FROM mba200 where 小票号=%s" % x
        cur.execute(sql)
        cnn.commit()
        data = cur.fetchall()

        name = re.findall(r'[0-9]+', str(data))
        # print(name)
        writer.writerow(name)
    csvfile.close()

    # 从顾客信息表中，查询顾客数据并统计总数
    for x in range(9900000984, 9900001023):
        x = str(x)
        sql = "SELECT DISTINCT 顾客年龄,顾客性别 FROM gk200 where 小票号=%s" % x
        cur.execute(sql)
        cnn.commit()
        data = cur.fetchall()
        LIST_COSTOMER.extend(data)
    n = len(LIST_COSTOMER)
    print("顾客数：", n)

    cur.close()
    cnn.close()


# 概念分层-->年龄分层，27以下青年+少年统称青少年，27-55中年，55以上老年
# 顾客，按性别属性，数量统计
def divid_age():
    teens, middle, old, man, wom = 0, 0, 0, 0, 0
    for i in LIST_COSTOMER:
        if int(i[0]) < 27:
            teens = teens + 1
        elif int(i[0]) > 55:
            old = old + 1
        else:
            middle = middle + 1
        if i[1] == '男':
            man = man + 1
        else:
            wom = wom + 1
    print('少', teens, '中', middle, '老', old)
    print('男', man, '女', wom)


if __name__ == '__main__':
    get_data()
    divid_age()
