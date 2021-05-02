# -*- coding: UTF-8 -*-
from kmodes.kprototypes import k_prototypes
"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2020/12/8 8:35
  Software: PyCharm
  Profile: K-Prototype算法是结合K-Means与K-modes算法，针对混合属性的，解决2个核心问题如下：
    1.度量具有混合属性的方法是，数值属性采用K-means方法得到P1，分类属性采用K-modes方法P2，那么D=P1+a*P2，
    a是权重，如果觉得分类属性重要，则增加a，否则减少a，a=0时即只有数值属性
    2.更新一个簇的中心的方法，方法是结合K-Means与K-modes的更新方法
"""

KP = k_prototypes(n_clusters=3, init='Cao').fit_predict(X, categorical=self.dis_col)