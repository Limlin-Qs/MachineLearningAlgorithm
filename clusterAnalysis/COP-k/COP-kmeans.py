# -*- coding: UTF-8 -*-
import numpy as np

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2020/12/18 14:53
  Software: PyCharm
  Profile:
"""


def COP_K_means(X, n_clusters=3, Con1=None, Con2=None):
    clusters = np.random.choice(len(X), n_clusters)
    clusters = X[clusters]
    labels = np.array([-1 for i in range(len(X))])

    def validata_constrained(d, c, Con1, Con2):
        for dm, value in enumerate(Con1[d]):  # should in the same group
            if value == 0:
                continue
            if labels[dm] == -1 or labels[dm] == c:  # has not allocated or ...
                continue
            if labels[dm] != -1 and labels[dm] != c:  # has allocated
                return False

        for dm, value in enumerate(Con2[d]):  # cannot in the same group
            if value == 0:
                continue
            if labels[dm] == -1 or labels[dm] != c:  # has not allocated or ...
                continue
            if labels[dm] != -1 and labels[dm] == c:  # has allocated
                return False

        return True

    while True:
        labels_new = np.array([-1 for i in range(len(X))])
        for i, xi in enumerate(X):
            close_list = np.argsort([np.linalg.norm(xi - cj) for cj in clusters])

            unexpect = True
            for index in close_list:
                if validata_constrained(i, index, Con1, Con2):
                    unexpect = False
                    labels_new[i] = index
                    break
            if unexpect:
                raise Exception("Can not utilize COP-k-Means algorithm inside the dataset.")

        if sum(labels != labels_new) == 0:
            break

        for j in range(n_clusters):
            clusters[j] = np.mean(X[np.where(labels_new == j)], axis=0)
        labels = labels_new.copy()
    return labels