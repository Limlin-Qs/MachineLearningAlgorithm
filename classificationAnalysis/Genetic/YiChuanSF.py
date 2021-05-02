import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10           #序列长度
POP_SIZE = 100          #种群的个体数目
CROSS_RATE = 0.8        #选择多少个体进行交叉配对
MUTATION_RATE = 0.003   #变异的概率/强度
N_GENERATIONS = 100     #有多少代（主循环的迭代次数）
X_BOUND = [0, 5]         #输入数据的范围

#需要要找到哪个函数的最大值
def F(x): return np.sin(10*x)*x + np.cos(2*x)*x  #返回y值（此例中为高度）

#用0-1按照定义的规模表示一代种群
pop = np.random.randint(1, size=(POP_SIZE, DNA_SIZE))  #默认从0开始，随机（提供选择的数值个数，重复次数（纵向，横向））
#print(pop)

#适应度函数(在本例中直接使用F函数就好，但是要处理返回值为负数的情况)
def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)

#DNA的翻译规则
def translateDNA(pop):
    #把二进制翻译成十进制,并将其归一化至一个范围（0,5）
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


#适者生存不适者淘汰
#idx:种群中个体的编号
#p参数定义：按什么规格来选择（此例中按照比例来选择，适应度得分高的p越大）
def select(pop,fitness_score):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness_score/fitness_score.sum())
    return pop[idx]


#繁衍
#父母的DNA交叉配对(定义到底怎么个交叉法)
def crosscover(parent,pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size = 1)
        cross_points = np.random.randint(0, 2, size = DNA_SIZE).astype(np.bool)
        parent[cross_points] = pop[i_, cross_points]
    return parent

#变异（定义如何随机挑选0-1互换的位置）
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


#画图
#plt.ion()       # something about plotting
#x = np.linspace(*X_BOUND, 200)
#plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))
    print(translateDNA(pop))
    #print(F_values)

    #计算适应度得分，适应度越高的越好（本例中，适应度就用y轴-高度，也就是F_value体现）
    fitness_score = get_fitness(F_values)

    #print("Most fitted DNA: ", pop[np.argmax(fitness_score), :])

    #进行适者生存的选择（把种群和得分传入)
    pop = select(pop, fitness_score)
    #print(pop)

    pop_copy = pop.copy()
    #对于种群中所有个体，在种群中挑选另一个体进行配对:
    for parent in pop:
        child = crosscover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child

#plt.ioff();plt.show(
