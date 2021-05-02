import numpy
import matplotlib.pyplot as plt


# 目前只能是二维的聚类
# 每次运行聚类中心三角形都会变化，但是数据圆形表示位置没变，感觉整体是一个用来生成聚类中心的程序
# 不小心误删了test.txt，但是有K-means1.py可以用，不过数完全不一样
# 能画散点图，输入文档格式类型如test.txt
def load_data(filename):  # 读入数据
    re_save = []
    ff = open(filename)
    for i in ff.readlines():
        temp1 = i.split('\n')[0].split('\t')
        # 关于数据转换，这两种写法都是可以的
        # 写法一：
        temp = list(map(float, temp1))
        # 写法二：
        # re_save.append(temp1)
        # re_save=[list(map(float,line)) for line in re_save]
        re_save.append(temp)
    return numpy.array(re_save)


def creat_cluter(dataset, dataset_lie_shu, tezhengshu):
    temp_cluter = numpy.array(numpy.zeros([tezhengshu, dataset_lie_shu]))
    for i in range(dataset_lie_shu):
        min_lie = numpy.min(dataset[:, i])
        max_lie = numpy.max(dataset[:, i])
        kuadu_lie = max_lie - min_lie
        ran_lie = kuadu_lie * (numpy.random.random()) + min_lie
        temp_cluter[:, i] = ran_lie
    return numpy.array(temp_cluter)


# 计算距离
def distance(date1, date2):
    return numpy.sqrt(sum(pow(date1 - date2, 2)))


# 聚类算法
def k_means1(dataset, tezhengshu):
    dataset_lie_shu = numpy.shape(dataset)[1]
    dataset_hang_shu = numpy.shape(dataset)[0]
    # 这里非常关键，一定要将其设为1的矩阵，否则造成一次就退出了
    tongji_matrix = numpy.array(numpy.ones([dataset_hang_shu, 2]))
    # ll=numpy.ones([dataset_hang_shu, 1])
    # print(type(ll))
    # print_data(dataset,numpy.array(numpy.ones([dataset_hang_shu,1])).flatten())
    creat_random_matrix = creat_cluter(dataset, dataset_lie_shu, tezhengshu)
    clu_change = True
    lun_index = 1
    while clu_change:
        # 这句目的是一次迭代之后则改变标志位，如果中途没有哪个位分类错误了，则会导致整个迭代过程退出
        clu_change = False
        for i in range(dataset_hang_shu):
            juli_matrix = [(distance(creat_random_matrix[j, :], dataset[i, :])) for j in range(4)]
            zhi_min_juli_matrix = min(juli_matrix)
            index_min_juli_matrix = juli_matrix.index(zhi_min_juli_matrix)
            # 如果这个点分类还是会改变说明还要继续分类，即继续迭代一次就是为了为了这个点，即采用新的簇之后,这是这个函数非常关键的地方
            if tongji_matrix[i, 0] != index_min_juli_matrix:
                clu_change = True
            # 将这个点的分类结果及距离他所属簇距离写入统计矩阵
            tongji_matrix[i, :] = index_min_juli_matrix, zhi_min_juli_matrix
        for i in range(4):
            belong_matrix = numpy.nonzero(tongji_matrix[:, 0] == i)[0]
            print("所属矩阵", belong_matrix)
            belong_dataset = dataset[belong_matrix]
            if len(belong_matrix) != 0:
                creat_random_matrix[i, ] = numpy.mean(belong_dataset, axis=0)
                print("第%d轮第%d个簇点改变：" % (lun_index, i))
                print(creat_random_matrix)
        lun_index += 1
    return creat_random_matrix, tongji_matrix


def print_data(dataset, lables):
    color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
    kong_temp = {}
    for i in lables:
        if i not in kong_temp.keys():
            kong_temp[i] = 0
        kong_temp[i] += 1
    kong_temp[5] = 90
    print(kong_temp)
    print("*******")
    print(list(kong_temp.keys()))
    for index, ky in enumerate(list(kong_temp.keys())):
        kong_temp[ky] = color[index]
    print(kong_temp)
    # print(kong_temp)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('The third graph')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.scatter(dataset[:, 0], dataset[:, 1], s=10 * lables, c='g')
    plt.show()


def print_data_test(dataset, tongji_matrix, cluter_matrix):
    color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('The third graph')
    plt.xlabel('X')
    plt.ylabel('Y')
    print(tongji_matrix[:, 0])
    print_matrix = numpy.zeros([numpy.shape(dataset)[0], 1])
    #
    # 这里也是非常关键的，由于要将每个数据的归类转换为颜色标签，这样写非常简便
    #
    print_matrix = numpy.array([list(map(str, matrix)) for matrix in print_matrix]).flatten()
    #
    for i in range(4):
        print_matrix[tongji_matrix[:, 0] == i] = color[i]
    print(print_matrix)
    print(type(tongji_matrix[:, 0]))
    ax.scatter(dataset[:, 0], dataset[:, 1], c=print_matrix)
    ax.scatter(cluter_matrix[:, 0], cluter_matrix[:, 1], s=120, c='r', marker='<')
    plt.show()


if __name__ == '__main__':
    # 同一个算法的测试数据集标志“文件名+xn”
    # dataset = load_data('k-means10')
    data = load_data('test3')
    cluter_matrix, tongji_matrix = k_means1(data, 4)
    print_data_test(data, tongji_matrix, cluter_matrix)
