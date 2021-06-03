# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import warnings # for hiding warnings
warnings.filterwarnings('ignore')
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/5/31 7:56
  Software: PyCharm
  Profile: github
"""
# base_path = r'C:\Users\wayiqin\Downloads\Data_mining\Assignment2'
# path to the folder that contains the dataset.csv.
dataset_name = 'retail.txt'
# dataset_path = rf'{base_path}/{dataset_name}'
dataset_path = dataset_name
dataset = pd.read_csv(dataset_path, sep='\t',header=None,names=["transaction"]) # extract each trasaction in each line
dataset["transaction"] = dataset.transaction.apply(lambda x: x.split()) # split each items by blank space
dataset = dataset.transaction.values.tolist()
lendata = len(dataset)


def findfrequent_item(data, min_support_count):
    invert_data = {}
    candidates = []
    frequent_item = []
    support = []
    trans = 0
    # convert to the vertical format
    for row in data:
        trans += 1
        for item in row:
            if item not in invert_data:
                invert_data[item] = set()
            else:
                invert_data[item].add(trans)

    # find the 1-frequent patterns:
    start_time = time.time()
    for item in invert_data.keys():
        candidates.append([item])
        # if satisfied this condition, the item will be added into the frequent_item set
        if len(invert_data[item]) >= min_support_count:
            frequent_item.append([item])
            support.append(invert_data[item])
    end_time = time.time()
    itertime = end_time - start_time
    return frequent_item, support, candidates, itertime

def getIntersection(frequent_item, support, min_support_count):
    sub_frequent_item = []
    sub_support = []
    candidates = []
    k = len(frequent_item[0])
    for i in range(len(frequent_item)):
        for j in range(i+1, len(frequent_item)):
    # enumerate all the frequent items
            L1 = list(frequent_item[i])[:k-1]
            L2 = list(frequent_item[j])[:k-1]
    # judge the top k-1 elements are or not the same, if same, then can calculate the intersection of two support list
            if L1 == L2:
                flag = len(list(set(support[i]).intersection(set(support[j]))))
                if flag >= 1:
                    candidates.append(list(set(frequent_item[i]) | set(frequent_item[j])))
                if flag >= min_support_count:
                    sub_frequent_item.append(list(set(frequent_item[i]) | set(frequent_item[j])))
                    sub_support.append(list(set(support[i]).intersection(set(support[j]))))
    return sub_frequent_item, sub_support, candidates

def eclat(dataset, min_supportcount):
    starteclat = time.time()
    candidates_count = 0
    frequent_count = 0
    frequent_item, support, candidates, itertime = findfrequent_item(dataset, min_supportcount)
    candidates_count += len(candidates)
    frequent_count += len(frequent_item)
    frequent_pattern = [frequent_item[i][0] for i in range(len(frequent_item))]
    iteration = 1
    print(f"iteration{iteration}:\n", f"- execution time: {itertime} seconds \n", f"- candidates: {len(candidates)} \n", f"- frequent itemsets: {len(frequent_item)} \n")
    frequent_set = []
    support_set = []
    frequent_set.append(frequent_item)
    support_set.append(support)
    while len(frequent_item) >= 2:
        start_time = time.time()
        frequent_item, support, candidates = getIntersection(frequent_item, support, min_supportcount)
        if frequent_item:
            iteration += 1
            candidates_count += len(candidates)
            frequent_count += len(frequent_item)
            frequent_set.append(frequent_item)
            support_set.append(support)
            end_time = time.time()
            itertime = end_time - start_time
            print(f"iteration{iteration}:\n", f"- execution time: {itertime} seconds \n", f"- candidates: {len(candidates)} \n", f"- frequent itemsets: {len(frequent_item)} \n")
    endeclat = time.time()
    eclat_execution_time = endeclat - starteclat
    return frequent_set, support_set, frequent_pattern, candidates_count, frequent_count, eclat_execution_time
min_support = 0.0015
min_supportcount = min_support * len(dataset)
frequent_set, support_set, frequent_pattern, candidates_count, frequent_count, exetime= eclat(dataset, min_supportcount)

# output the pattern file
frequent_patterns = pd.DataFrame(columns = ["frequent pattern", "support count", "support"])
k = 0
for i in range(len(frequent_set)):
    for j in range(len(frequent_set[i])):
        frequent_patterns.loc[k] = [frequent_set[i][j],len(support_set[i][j]), len(support_set[i][j]) / len(dataset)]
        k += 1

patterns_name = 'patterns.csv'
frequent_patterns.to_csv(patterns_name, index=False)
def maximal_pattern(frequent_set, support_set, frequent_pattern):
    x = len(frequent_set)-1
    submax = []
    subsupport = []
    while frequent_pattern:
        while x >= 0:
            for i in range(len(frequent_set[x])-1,-1,-1):
                tempmax = frequent_set[x][i]
                tempsupport = support_set[x][i]
                inter = set(tempmax).intersection(set(frequent_pattern))
                if inter:
                    submax.append(tempmax)
                    subsupport.append(tempsupport)
                    frequent_pattern = [ele for ele in frequent_pattern if ele not in inter]
            x = x-1
    return submax, subsupport
maximal_patterns, maximum_support = maximal_pattern(frequent_set, support_set, frequent_pattern)
maxsupport_set = [len(i) for i in maximum_support]

# output the maximal file
maxpatterns = pd.DataFrame(columns = ["maximal pattern", "support count", "support"])
k = 0
for i in range(len(maximal_patterns)):
        maxpatterns.loc[k] = [maximal_patterns[i],len(maximum_support[i]), len(maximum_support[i]) / len(dataset)]
        k += 1
maximal_name = 'maximal.csv'
maxpatterns.to_csv(maximal_name, index=False)

compression_ratio = 1 - (len(maximal_patterns) / frequent_count)
print(f"Compression ratio:\n", f"- frequent itemsets: {frequent_count} \n", f"- maximal patterns: {len(maximal_patterns)} \n", f"- compression ratio: {compression_ratio} \n")
def PowerSetsBinary(items):
    N = len(items)
    com = []
    for i in range(2 ** N):
        combo = []
        for j in range(N):
            if(i>>j)%2:
                combo.append(items[j])
        com.append(combo)
    com.pop(0) # drop the empty set
    com.pop(-1) # drop the same set with its own
    return com
def find_in_list_of_list(mylist, char):
    for sub_list in mylist:
        if char in sub_list:
            return (mylist.index(sub_list), sub_list.index(char))
    raise ValueError("'{char}' is not in list".format(char = char))


def find_rule(actual_combo, frequent_set, support_set, support_item, frequent_item, min_confidence, lendata):
    rule = pd.DataFrame(columns=["antecedent", "consequent", "support count", "support", "confidence"])
    k = 0
    for i in range(len(actual_combo) - 1, -1, -1):
        for j in range(len(actual_combo[i]) - 1, -1, -1):
            combo = actual_combo[i][j]

            # for each subset, find out its index from the frequent set
            (p, q) = find_in_list_of_list(frequent_set, combo)

            # calculate the support of each subset and the support of the choosed frequent item
            support_X = len(support_set[p][q])  # the support of subset
            support_Z = len(support_item)  # the support of the choosed frequent item
            confidence = support_Z / support_X  # confidence

            if confidence >= min_confidence:
                X = [ele for ele in frequent_item if ele not in combo]
                relativesupport_Z = support_Z / lendata
                # save the rules between X and Z-X
                rule.loc[k] = [combo, X, support_Z, relativesupport_Z, confidence]
                k += 1
    return rule


def generateRules(frequent_set, support_set, min_confidence, lendata):
    rules = pd.DataFrame(columns=["antecedent", "consequent", "support count", "support", "confidence"])

    # from the last frequent pattern, begain to find its subsets
    for i in range(len(frequent_set) - 1, -1, -1):
        for j in range(len(frequent_set[i]) - 1, -1, -1):
            frequent_item = frequent_set[i][j]
            support_item = support_set[i][j]
            if len(frequent_item) > 1:
                combo = PowerSetsBinary(frequent_item)

                # judge the subsets is or not in the frequent pattern set
                actual_combo = [list(filter(lambda x: x in combo, sublist)) for sublist in frequent_set]
                while [] in actual_combo:
                    actual_combo.remove([])

                # then call the rule function to find the association rules between different subsets and taget patterns
                rule = find_rule(actual_combo, frequent_set, support_set, support_item, frequent_item, min_confidence,
                                 lendata)
                rules = pd.concat([rules, rule], axis=0, ignore_index=True)
    return rules
start_time = time.time()
min_confidence = 0.9
rules = generateRules(frequent_set, support_set, min_confidence, lendata)
end_time = time.time()
rule_time = end_time - start_time

print(f"Strong rules:\n", f"Execution time: {rule_time} seconds\n",f"Strong rules: {len(rules.index)} \n")

for i in range(len(rules.index)):
    print(f"Rule {i+1}: {rules.antecedent[i]} => {rules.consequent[i]} , conf = {rules.confidence[i]} \n")
    rules.antecedent[i] = f"{rules.antecedent[i]} => {rules.consequent[i]}" # change the rule format in rules.csv as required

# save as csv file
rules = rules.drop(['consequent'], axis=1)
rules.rename(columns={'antecedent':'rules'}, inplace = True)
rules_name = 'rules.csv'

rules.to_csv(rules_name, index=False)


minimum_support_set = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
times = []
number_candidates = []
number_frequent_itemsets = []
for i in range(len(minimum_support_set)):
    min_support = minimum_support_set[i]
    min_supportcount = min_support * lendata
    # call the eclat function
    frequent_set, support_set, frequent_pattern, candidates_count, frequent_count, exetime= eclat(dataset, min_supportcount)
    times.append(exetime)
    number_candidates.append(candidates_count)
    number_frequent_itemsets.append(frequent_count)

plt.figure(figsize=(10, 10))
plt.title("Runtime & Minimum_support", fontsize = 20.0)
plt.plot(minimum_support_set, times, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.xticks(fontsize = 15.0)
plt.yticks(fontsize = 15.0)
plt.xlabel('Minimum support', fontsize = 20.0)
plt.ylabel('Runtime', fontsize = 20.0)
plt.show()
plt.figure(figsize=(10, 10))
plt.title("Number of candidates & Minimum_support",fontsize = 20.0)
plt.plot(minimum_support_set, number_candidates, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.xticks(fontsize = 15.0)
plt.yticks(fontsize = 15.0)
plt.xlabel('Minimum support', fontsize = 20.0)
plt.ylabel('Number of candidates', fontsize = 20.0)
plt.show()

plt.figure(figsize=(10, 10))
plt.title("Number of frequent itemsets & Minimum_support", fontsize = 20.0)
plt.plot(minimum_support_set, number_frequent_itemsets, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.xticks(fontsize = 15.0)
plt.yticks(fontsize = 15.0)
plt.xlabel('Minimum support', fontsize = 20.0)
plt.ylabel('Number of frequent itemsets', fontsize = 20.0)
plt.show()