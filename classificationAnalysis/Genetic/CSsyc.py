from ga import Ga as ga
import matplotlib.pyplot as plt
import numpy as np


def multiply(coding):
    #
    # 繁衍一代
    # 传入参数,上一代编码coding
    # 返回,新一代编码mutationcoding,新一代解码值decoding,新一代适应度值fitnessvalues
    #
    inipol = coding
    decoding = ga.decodeit(inipol, boudary[0], brange)
    fitnessvalues = ga.fitnessevaluations(decoding)
    copycoding = ga.copyoperator(inipol, fitnessvalues)
    crossovercoding = ga.crossoveroprator(copycoding, crossoverpossibility)
    mutationcoding = ga.mutationoprator(crossovercoding, mutationpossibility, boudary[0], brange)
    return mutationcoding, decoding, fitnessvalues


def generateit():
    #
    # 繁衍多代
    # 返回,最终编码coding,每次繁衍的编码decodings,每次繁衍的适应度值fitnessvalues,初始编码iniplot,每代的适应度值fitness
    #
    inipol = ga.codeit(n, length)
    decodings = []
    fitnessvalues = []
    fitness = []
    iniplot = np.copy(inipol).tolist()
    for i in range(g):
        inipol = multiply(inipol)[0]
        decodings.append(multiply(inipol)[1])
        fitnessvalues.append(multiply(inipol)[2])
        fitness.append(np.mean(multiply(inipol)[2]))
    return inipol, decodings, fitnessvalues, iniplot, fitness


def plotit():
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    inicoding = generateit()[3]
    inidecoding = ga.decodeit(inicoding, boudary[0], brange)
    initfittness = ga.fitnessevaluations(inidecoding)
    decoding = generateit()[1][-1]
    fitnessvalue = generateit()[2][-1]
    fitness = generateit()[4]
    x = np.linspace(boudary[0], boudary[1], 1000)
    fits = x + 10*np.sin(5*x) + 7*np.cos(4*x)
    plt.suptitle("简单遗传算法")
    plt.subplot(121)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(x, fits)                                  # 适应度曲线
    plt.scatter(inidecoding, initfittness, c='blue')  # 初始编码
    plt.scatter(decoding, fitnessvalue, c='red')      # 进化后的编码
    plt.xlabel("编码")
    plt.ylabel("适应度值")
    plt.title("适应度曲线")
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(122)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(range(g), [j for j in fitness])         # 适应度值变化曲线
    plt.xlabel("进化次数")
    plt.ylabel("适应度值")
    plt.title("适应度值变化曲线")
    plt.savefig("简单遗传算法")


if __name__ == "__main__":
    boudary = [0, 10]  # 变量范围
    brange = boudary[1] - boudary[0]
    accuracy = 1 / 1000  # 精度
    n = 50  # 初始群体数量
    g = 100  # 代
    crossoverpossibility = 0.65  # 杂交概率
    mutationpossibility = 0.05  # 变异概率
    length = int(np.rint(np.log2(brange / accuracy)))  # 编码长度
    with open("初始编码.txt", 'w', encoding="utf-8") as f:
        for i in generateit()[3]:
            f.writelines(i+'\n')
    with open("进化编码.txt", 'w', encoding="utf-8") as f:
        for i in generateit()[0]:
            f.writelines(i + '\n')
    with open("进化解码.txt", 'w', encoding="utf-8") as f:
        for i in generateit()[1][-10:]:
            f.writelines(str(i)+'\n')
    with open("进化适应度值.txt", 'w', encoding="utf-8") as f:
        for i in generateit()[2][-10:]:
            f.writelines(str(i)+'\n')
    plotit()