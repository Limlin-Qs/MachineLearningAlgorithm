# -*- coding: UTF-8 -*-
import time
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/5/30 21:28
  Software: PyCharm
  Profile: https://www.cnblogs.com/xyou/p/10068955.html
"""

def loadDataSet():
    result = []
    with open('Relevance_exp_result_group.txt') as f:
        for line in f:
            line = line.strip('\n')
            # split处理之后返回list，相当于二维数组的形式
            result.append(line.split(","))
    print(result)
    return result
    # return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, CK, minSupport):
    ssCnt = {}
    for tid in D:
        for can in CK:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


# 频繁项集两两组合
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2];
            L2 = list(Lk[j])[:k - 2]
            L1.sort();
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.4):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        CK = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, CK, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# 规则计算的主函数
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '--->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


if __name__ == '__main__':
    dataSet = loadDataSet()
    start = time.time()
    L, supportData = apriori(dataSet)
    rules = generateRules(L, supportData, minConf=0.4)
    end = time.time()
    print(end - start)