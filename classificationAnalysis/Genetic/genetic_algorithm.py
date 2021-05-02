import numpy as np


class Ga:
    @classmethod
    def codeit(cls, n, length):
        #
        # 创建初始群体
        # 传入参数, 初始群体数量n, 编码长度length
        # 返回初始群体编码列表
        #
        inipol = []
        for i in range(n):
            tempb = ""
            for j in range(length):
                data = np.random.choice([0, 1])
                tempb += str(data)
            inipol.append(tempb)
        return inipol

    @classmethod
    def decodeit(cls, coding, lowest, ranges):
        #
        # 解码
        # 传入参数, 编码coding, 可行域最小值lowest, 取值长度ranges
        # 返回解码列表decoding
        #
        decoding = np.copy(coding).tolist()
        length = len(decoding[0])
        for i in range(len(decoding)):
            decoding[i] = np.round(lowest + int(decoding[i], base=2)*ranges/(2**length-1), 3)
        return decoding

    @staticmethod
    def fits(x):
        #
        # 适应度函数
        #
        return x + 10*np.sin(5*x) + 7*np.cos(4*x)

    @classmethod
    def fitnessevaluations(cls, decoding):
        #
        # 个体适应度评价
        # 传入参数,解码decoding
        # 返回个体适应度值列表fitnessvalues
        #
        fitnessvalues = []
        for i in decoding:
            fitnessvalues.extend([np.round(Ga.fits(i), 3)])
        return fitnessvalues

    @classmethod
    def copyoperator(cls, coding, fitnessvalues):
        #
        # 复制算子
        # 传入参数,编码coding,适应度值fitnessvalues
        # 返回复制算子编码copycoding
        #
        copycoding = np.copy(coding).tolist()
        minindex = fitnessvalues.index(min(fitnessvalues))
        maxindex = fitnessvalues.index(max(fitnessvalues))
        copycoding[minindex] = copycoding[maxindex]
        copycoding[minindex] = copycoding[maxindex]
        return copycoding

    @classmethod
    def crossoveroprator(cls, copycoding, crossoverposssibility):
        #
        # 杂交算子
        # 参入参数,复制算子编码copycoding,杂交概率crossoverposibility
        # 返回杂交算子编码crossovercoding
        #
        crossovercoding = np.copy(copycoding).tolist()
        copysave = []
        crossover = []
        for i in range(int(len(crossovercoding)*crossoverposssibility)):
            copysave.append(crossovercoding[i])
        for i in range(int(len(crossovercoding) * (1-crossoverposssibility))):
            crossover.append(crossovercoding[-i-1])
        crossoverpoit = len(crossovercoding)-1
        for i in range(int(len(crossover)/2)):
            crossover[i], crossover[-i-1] = \
                crossover[i][:crossoverpoit]+crossover[-i-1][crossoverpoit:], \
                crossover[-i - 1][:crossoverpoit] + crossover[-i][crossoverpoit:]
        crossovercoding = copysave + crossover
        return crossovercoding

    @classmethod
    def mutationoprator(cls, crossovercoding, mutationpossibility, lowest, ranges):
        #
        # 变异算子
        # 传入参数, 杂交算子编码crossovercoding,变异概率crossoverpossibility
        # 可行域最小值lowest, 取值长度ranges,
        # 返回变异算子编码
        #
        mutationcoding = np.copy(crossovercoding).tolist()
        crossoverfitnessvalues = Ga.fitnessevaluations(Ga.decodeit(mutationcoding, lowest, ranges))
        tempfitvaleus = np.copy(crossoverfitnessvalues).tolist()
        tempfitvaleus.sort()
        mutationvalues = []
        for i in range(int(len(tempfitvaleus)*mutationpossibility)):
            mutationvalues.append(tempfitvaleus[i])
        mutaionindex = []
        for i in mutationvalues:
            mutaionindex.append(crossoverfitnessvalues.index(i))
        mutationposition = np.random.randint(0, len(mutationcoding[0]))
        for i in mutaionindex:
            if int(mutationcoding[i][mutationposition]) == 1:
                mutationcoding[i] = mutationcoding[i][:mutationposition] + '0' + mutationcoding[i][mutationposition+1:]
            else:
                mutationcoding[i] = mutationcoding[i][:mutationposition] + '1' + mutationcoding[i][mutationposition + 1:]
        return mutationcoding